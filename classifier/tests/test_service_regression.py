import json

from classifier.services import AbstractClassifier, LABELS


def test_threshold_fallback_uses_highest_scoring_label():
    service = AbstractClassifier()
    service._threshold = .9
    result = service._format_scores([.2, .4, .3, .1, .05], "pytorch")
    assert result["categories"] == ["chemistry"]
    assert list(result["scores"]) == LABELS


def test_missing_or_incompatible_artifacts_fall_back(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    assert AbstractClassifier().backend == "keyword_fallback"

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "metadata.json").write_text(json.dumps({"preprocessing": "wrong"}), encoding="utf-8")
    assert AbstractClassifier().backend == "keyword_fallback"
