import hashlib
import json

from dejargonizer.verify_pairs import repair_copy, verify_pairs


def make_pair(source="Technical source.", target="Simple source."):
    return {
        "id": "pair-1",
        "split": "train",
        "source": source,
        "target": target,
        "source_hash": hashlib.sha256(source.encode()).hexdigest(),
        "label_provider": "ollama",
        "label_model": "llama3.1:8b",
        "prompt_version": "v1",
        "issues": [],
    }


def test_verify_pairs_accepts_valid_record(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(json.dumps(make_pair()) + "\n")
    report = verify_pairs(path)
    assert report["valid"] is True
    assert report["records"] == 1


def test_verify_pairs_reports_bad_hash_and_malformed_line(tmp_path):
    path = tmp_path / "pairs.jsonl"
    row = make_pair()
    row["source_hash"] = "wrong"
    path.write_text(json.dumps(row) + "\nnot json\n")
    report = verify_pairs(path)
    assert report["valid"] is False
    assert any("source_hash" in error for error in report["errors"])
    assert any("invalid JSON" in error for error in report["errors"])


def test_repair_copy_removes_only_exact_duplicates_and_preserves_input(tmp_path):
    source = tmp_path / "pairs.jsonl"
    repaired = tmp_path / "pairs.repaired.jsonl"
    row = make_pair()
    source.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    original = source.read_text()
    result = repair_copy(source, repaired)
    assert source.read_text() == original
    assert len(result["removed"]) == 1
    assert len(repaired.read_text().splitlines()) == 1


def test_repair_copy_does_not_remove_conflicting_duplicate(tmp_path):
    source = tmp_path / "pairs.jsonl"
    repaired = tmp_path / "pairs.repaired.jsonl"
    first = make_pair()
    second = {**first, "target": "Different target."}
    source.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    result = repair_copy(source, repaired)
    assert result["removed"] == []
    assert len(repaired.read_text().splitlines()) == 2
