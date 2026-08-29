import numpy as np

from classifier.preprocessing import (
    FORMULA_TOKEN,
    PRIMARY_LABEL_TARGET,
    SECONDARY_LABEL_TARGET,
    register_formula_token,
)
from scripts.data import build_targets


class Tokenizer:
    unk_token_id = 0

    def __init__(self):
        self.tokens = {"[UNK]": 0}

    def add_special_tokens(self, values):
        for token in values["additional_special_tokens"]:
            self.tokens.setdefault(token, len(self.tokens))

    def convert_tokens_to_ids(self, token):
        return self.tokens.get(token, self.unk_token_id)


def test_targets_use_fixed_primary_and_secondary_values():
    labels = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.float32)
    primary = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.float32)

    assert build_targets(labels, primary).tolist() == [
        [PRIMARY_LABEL_TARGET, SECONDARY_LABEL_TARGET, 0.0],
        [0.0, SECONDARY_LABEL_TARGET, PRIMARY_LABEL_TARGET],
    ]


def test_formula_token_is_registered_as_one_special_token():
    tokenizer = Tokenizer()

    token_id = register_formula_token(tokenizer)

    assert token_id == tokenizer.convert_tokens_to_ids(FORMULA_TOKEN)
    assert token_id != tokenizer.unk_token_id
