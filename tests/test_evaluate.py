import pytest

from dejargonizer.evaluate import _evaluation_quality, evaluate_file


def test_evaluate_rejects_invalid_progress_interval(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match="progress_interval"):
        evaluate_file(path, "checkpoint", progress_interval=0)


def test_evaluation_quality_reports_readability_reduction_and_numbers():
    class TextStat:
        @staticmethod
        def flesch_reading_ease(text):
            return 20.0 if "technical" in text else 60.0

    result = _evaluation_quality(
        ["The technical study found 42% improvement."],
        ["The study found 42% improvement."],
        ["The study found a 42% improvement."],
        TextStat,
    )

    assert result["readability_improvement"] == 40.0
    assert result["source_numbers_preserved_rate"] == 1.0
    assert result["prediction_word_reduction_percent"] > 0
    assert result["empty_prediction_rate"] == 0.0


def test_evaluation_quality_detects_empty_and_unchanged_predictions():
    class TextStat:
        @staticmethod
        def flesch_reading_ease(text):
            return 50.0

    result = _evaluation_quality(
        ["A technical sentence.", "Another source."],
        ["", "Another source."],
        ["A simple sentence.", "Another source."],
        TextStat,
    )

    assert result["empty_prediction_rate"] == 0.5
    assert result["unchanged_prediction_rate"] == 0.5
