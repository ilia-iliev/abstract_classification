"""Shared PyTorch classifier used by training and inference."""

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


@dataclass(frozen=True)
class BackboneSpec:
    pooling: str


# Pooling follows each checkpoint family's embedding/classification convention.
BACKBONES = {
    "google-bert/bert-base-uncased": BackboneSpec("cls"),
    "answerdotai/ModernBERT-base": BackboneSpec("cls"),
    "google/embeddinggemma-300m": BackboneSpec("cls"),
    "Qwen/Qwen3-Embedding-0.6B": BackboneSpec("last_token"),
}
DEFAULT_BACKBONE = "google-bert/bert-base-uncased"


MISTRAL_PRETOKENIZER_REGEX = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|"
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|"
    r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def load_tokenizer(name_or_path):
    """Load tokenizers with the corrected Mistral-family pre-tokenizer regex."""
    config = AutoConfig.from_pretrained(name_or_path)
    if config.model_type == "qwen3":
        return AutoTokenizer.from_pretrained(name_or_path, fix_mistral_regex=True)
    if config.model_type == "gemma3_text":
        tokenizer = AutoTokenizer.from_pretrained(name_or_path, fix_mistral_regex=False)
        import tokenizers

        tokenizer.backend_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Split(
            pattern=tokenizers.Regex(MISTRAL_PRETOKENIZER_REGEX), behavior="isolated"
        )
        return tokenizer
    return AutoTokenizer.from_pretrained(name_or_path)


def backbone_spec(name):
    try:
        return BACKBONES[name]
    except KeyError as error:
        raise ValueError(f"No documented pooling strategy registered for {name}") from error


def pool_hidden_states(hidden_states, attention_mask, strategy):
    if strategy == "cls":
        return hidden_states[:, 0]
    if strategy == "pooler":
        raise ValueError("pooler output must be selected from model outputs")
    if strategy == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    if strategy == "last_token":
        last_indices = attention_mask.size(1) - attention_mask.flip(1).long().argmax(dim=1) - 1
        return hidden_states[
            torch.arange(hidden_states.size(0), device=hidden_states.device), last_indices
        ]
    raise ValueError(f"Unknown pooling strategy: {strategy}")


class MultilabelClassifier(nn.Module):
    """Backbone, documented pooling, dropout, and one five-logit linear head."""

    def __init__(self, backbone, pooling, num_labels, dropout=None):
        super().__init__()
        self.backbone = backbone
        self.pooling = pooling
        hidden_size = backbone.config.hidden_size
        dropout = getattr(backbone.config, "classifier_dropout", None) if dropout is None else dropout
        self.dropout = nn.Dropout(0.1 if dropout is None else dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    @classmethod
    def from_pretrained(cls, name_or_path, num_labels, pooling=None):
        backbone = AutoModel.from_pretrained(name_or_path)
        configured_pooling = pooling or backbone_spec(getattr(backbone.config, "_name_or_path", name_or_path)).pooling
        return cls(backbone, configured_pooling, num_labels)

    def forward(self, **inputs):
        outputs = self.backbone(**inputs)
        if self.pooling == "pooler":
            if outputs.pooler_output is None:
                raise ValueError("Backbone does not provide a pooler output")
            embedding = outputs.pooler_output
        else:
            embedding = pool_hidden_states(outputs.last_hidden_state, inputs["attention_mask"], self.pooling)
        return self.classifier(self.dropout(embedding))

    def save(self, output):
        self.backbone.save_pretrained(output)
        torch.save(
            {"classifier": self.classifier.state_dict(), "pooling": self.pooling},
            output / "classifier.pt",
        )

    @classmethod
    def load(cls, path, num_labels, map_location="cpu"):
        payload = torch.load(path / "classifier.pt", map_location=map_location, weights_only=True)
        config = AutoConfig.from_pretrained(path)
        # ModernBERT enables torch.compile when Triton is detected. Inference must
        # remain usable when that optional compiler cannot run on the host GPU.
        if hasattr(config, "reference_compile"):
            config.reference_compile = False
        model = cls(AutoModel.from_pretrained(path, config=config), payload["pooling"], num_labels)
        model.classifier.load_state_dict(payload["classifier"])
        return model


def weighted_multilabel_loss(logits, weighted_labels, positive_weights):
    """Downweight secondary positives while keeping them positive labels."""
    labels = weighted_labels.ceil()
    weights = torch.as_tensor(positive_weights, dtype=logits.dtype, device=logits.device)
    values = nn.functional.binary_cross_entropy_with_logits(
        logits, labels, pos_weight=weights, reduction="none"
    )
    observation_weights = torch.where(
        labels > 0, weighted_labels, torch.ones_like(weighted_labels)
    )
    return (values * observation_weights).mean()
