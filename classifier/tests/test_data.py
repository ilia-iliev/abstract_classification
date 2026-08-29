import json

from scripts.data import deduplicated_records, load_examples


def write_records(path, values):
    path.write_text("\n".join(json.dumps(value) for value in values), encoding="utf-8")


def test_deduplication_discards_conflicts_and_keeps_one_copy(tmp_path):
    dataset = tmp_path / "records.json"
    write_records(dataset, [
        {"id": "1", "abstract": " H$_2$O ", "categories": "physics.chem-ph"},
        {"id": "2", "abstract": "h$_2$o", "categories": "physics.chem-ph"},
        {"id": "3", "abstract": "Same abstract", "categories": "cs.LG"},
        {"id": "4", "abstract": " same  abstract ", "categories": "physics.atom-ph"},
    ])

    unique = list(deduplicated_records(dataset))

    assert [(record["id"], labels) for record, labels, _ in unique] == [
        ("1", {"chemistry"})
    ]


def test_loader_returns_deduplicated_preprocessed_examples(tmp_path):
    dataset = tmp_path / "records.json"
    write_records(dataset, [
        {"id": "1", "abstract": "Atomic spectroscopy", "categories": "physics.atom-ph"},
        {"id": "2", "abstract": "H$_2$O reaction", "categories": "physics.chem-ph"},
    ])

    examples, eligible, _ = load_examples(dataset, limit=10)
    labels_by_text = {text: labels for text, labels in examples}

    assert eligible == 2
    assert labels_by_text == {
        "Atomic spectroscopy": [0, 0, 0, 1, 0],
        "H2O reaction": [0, 1, 0, 0, 0],
    }
