import json

from dejargonizer.similarity import check_labeled_similarity


def test_similarity_rejects_empty_labeled_dataset(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps({"source": "a", "target": "", "issues": ["empty_target"]}) + "\n")
    try:
        check_labeled_similarity(path)
    except ValueError as error:
        assert "No labeled" in str(error)
    else:
        raise AssertionError("Expected empty labels to be rejected")


def test_similarity_validates_threshold(tmp_path):
    path = tmp_path / "labels.jsonl"
    path.write_text("")
    try:
        check_labeled_similarity(path, threshold=1.1)
    except ValueError as error:
        assert "threshold" in str(error)
    else:
        raise AssertionError("Expected invalid threshold to be rejected")
