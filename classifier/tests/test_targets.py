import numpy as np
import torch

from classifier.modeling import weighted_multilabel_loss
from classifier.preprocessing import (
    FORMULA_TOKEN,
    PRIMARY_LABEL_LOSS_WEIGHT,
    SECONDARY_LABEL_LOSS_WEIGHT,
    register_formula_token,
)
from scripts.data import build_weighted_labels


class Tokenizer:
    unk_token_id = 0

    def __init__(self):
        self.tokens = {"[UNK]": 0}

    def add_special_tokens(self, values):
        for token in values["additional_special_tokens"]:
            self.tokens.setdefault(token, len(self.tokens))

    def convert_tokens_to_ids(self, token):
        return self.tokens.get(token, self.unk_token_id)


def test_labels_use_fixed_primary_and_secondary_loss_weights():
    labels = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
    primary = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32)

    assert build_weighted_labels(labels, primary).tolist() == [
        [PRIMARY_LABEL_LOSS_WEIGHT, SECONDARY_LABEL_LOSS_WEIGHT, 0.0],
        [0.0, SECONDARY_LABEL_LOSS_WEIGHT, PRIMARY_LABEL_LOSS_WEIGHT],
    ]


def test_secondary_label_is_a_half_weighted_positive_not_a_soft_target():
    logit = torch.tensor([[1.0]], requires_grad=True)

    loss = weighted_multilabel_loss(logit, torch.tensor([[0.5]]), np.ones(1))
    expected = 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(
        logit, torch.ones_like(logit)
    )
    loss.backward()

    assert torch.isclose(loss, expected)
    assert logit.grad.item() < 0


def test_formula_token_is_registered_as_one_special_token():
    tokenizer = Tokenizer()

    token_id = register_formula_token(tokenizer)

    assert token_id == tokenizer.convert_tokens_to_ids(FORMULA_TOKEN)
    assert token_id != tokenizer.unk_token_id
