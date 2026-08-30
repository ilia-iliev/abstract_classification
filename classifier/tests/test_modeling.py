from types import SimpleNamespace
from unittest.mock import patch

from classifier.modeling import load_tokenizer


@patch("classifier.modeling.AutoTokenizer.from_pretrained")
@patch("classifier.modeling.AutoConfig.from_pretrained")
def test_qwen_tokenizer_enables_mistral_regex_fix(config, tokenizer):
    config.return_value = SimpleNamespace(model_type="qwen3")

    load_tokenizer("qwen")

    tokenizer.assert_called_once_with("qwen", fix_mistral_regex=True)


@patch("classifier.modeling.AutoTokenizer.from_pretrained")
@patch("classifier.modeling.AutoConfig.from_pretrained")
def test_gemma_tokenizer_applies_the_regex_without_transformers_broken_patch(config, tokenizer):
    config.return_value = SimpleNamespace(model_type="gemma3_text")
    loaded_tokenizer = SimpleNamespace(backend_tokenizer=SimpleNamespace())
    tokenizer.return_value = loaded_tokenizer

    assert load_tokenizer("gemma") is loaded_tokenizer

    tokenizer.assert_called_once_with("gemma", fix_mistral_regex=False)
    assert loaded_tokenizer.backend_tokenizer.pre_tokenizer is not None
