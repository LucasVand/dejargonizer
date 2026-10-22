from dejargonizer.quality import validate_target
from dejargonizer.labeling import _status_message


def test_quality_flags_missing_numbers():
    issues = validate_target("The trial found 42% improvement.", "The trial found improvement.")
    assert "number_changed_or_missing" in issues


def test_quality_accepts_plain_rewrite():
    assert validate_target("The study tests a new method.", "The study tests a new method.") == ["unchanged"]


def test_labeling_status_reports_completion():
    assert _status_message(0, 100) == "Labeling complete: 100/100 records already labeled"
