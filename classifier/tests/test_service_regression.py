import json

import pytest

from classifier.services import AbstractClassifier, LABELS, ModelUnavailableError


def test_threshold_selection_uses_highest_scoring_label():
    service = AbstractClassifier()
    service._threshold = .9
    result = service._format_scores([.2, .4, .3, .1, .05], "pytorch")
    assert result["categories"] == ["chemistry"]
    assert list(result["scores"]) == LABELS


def test_missing_artifact_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    service = AbstractClassifier()

    assert service.backend is None
    assert "missing" in service.load_error
    with pytest.raises(ModelUnavailableError, match="missing"):
        service.predict(["abstract"])


def test_incompatible_artifact_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"preprocessing": "wrong"}), encoding="utf-8"
    )
    service = AbstractClassifier()

    assert service.backend is None
    with pytest.raises(ModelUnavailableError, match="incompatible preprocessing"):
        service.predict(["abstract"])
