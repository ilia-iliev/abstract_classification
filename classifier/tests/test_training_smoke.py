import json
from argparse import Namespace

import pytest
import torch
from torch import nn

from classifier.labels import LABELS
from classifier.modeling import BACKBONES
from scripts.data import _partition
from scripts.train import train


class TinyTokenizer:
    unk_token_id = 0

    def __init__(self):
        self.size = 16

    def add_special_tokens(self, _tokens):
        self.size += 1

    def convert_tokens_to_ids(self, _token):
        return self.size - 1

    def __len__(self):
        return self.size

    def __call__(self, texts, **_kwargs):
        batch_size = len(texts)
        return {
            "input_ids": torch.ones((batch_size, 2), dtype=torch.long),
            "attention_mask": torch.ones((batch_size, 2), dtype=torch.long),
        }

    def save_pretrained(self, output):
        (output / "tokenizer.json").write_text("{}", encoding="utf-8")


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = Namespace(hidden_size=8)
        self.embedding = nn.Embedding(32, 8)

    def resize_token_embeddings(self, _size):
        pass

    def forward(self, input_ids, **_kwargs):
        return Namespace(last_hidden_state=self.embedding(input_ids))


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.pooling = "cls"
        self.classifier = nn.Linear(8, len(LABELS))

    def forward(self, input_ids, **kwargs):
        return self.classifier(self.backbone(input_ids, **kwargs).last_hidden_state[:, 0])

    def save(self, output):
        torch.save(self.classifier.state_dict(), output / "classifier.pt")


def smoke_dataset(path):
    categories = ["q-bio.BM", "physics.chem-ph", "cs.AI", "hep-th", "econ.EM"]
    records = []
    for partition in ("training", "validation", "test"):
        target = {"training": range(2, 10), "validation": [1], "test": [0]}[partition]
        for category in categories:
            identifier = next(
                f"{partition}-{category}-{index}"
                for index in range(1000)
                if _partition(f"{partition}-{category}-{index}") in target
            )
            records.append(
                {
                    "id": identifier,
                    "categories": category,
                    "abstract": f"{identifier}: {category} studies $H_2$O and $x^2 + y^2$.",
                }
            )
    path.write_text("".join(f"{json.dumps(record)}\n" for record in records), encoding="utf-8")


@pytest.mark.parametrize("sampling", ["tag_aware", "category_aware", "uniform"])
def test_training_smoke_runs_all_backbones_and_sampling_modes(tmp_path, monkeypatch, sampling):
    dataset = tmp_path / "arxiv.jsonl"
    smoke_dataset(dataset)
    monkeypatch.setattr("scripts.train.load_tokenizer", lambda _name: TinyTokenizer())
    monkeypatch.setattr(
        "scripts.train.MultilabelClassifier.from_pretrained",
        lambda *_args, **_kwargs: TinyClassifier(),
    )

    for index, model_name in enumerate(BACKBONES):
        output = tmp_path / f"{sampling}-{index}"
        train(
            Namespace(
                dataset=dataset,
                output=output,
                model_name=model_name,
                sampling=sampling,
                per_label=1,
                validation_limit=5,
                test_limit=5,
                limit=5,
                train_limit=5,
                minimum_label_examples=1,
                class_weighting="none",
                batch_size=5,
                epochs=1,
                learning_rate=1e-3,
                weight_decay=0.0,
                warmup_ratio=0.0,
                device="cpu",
                thresholds=[0.5],
            )
        )
        metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["base_model"] == model_name
        assert metadata["training_size"] == 5
        assert metadata["preprocessing"] == "inline_math_24_formula_token_v2"
