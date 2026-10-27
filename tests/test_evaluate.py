import pytest

from dejargonizer.evaluate import evaluate_file


def test_evaluate_rejects_invalid_progress_interval(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="progress_interval"):
        evaluate_file(path, "checkpoint", progress_interval=0)
