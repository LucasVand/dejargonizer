import hashlib
import json

from dejargonizer.verify_pairs import verify_pairs


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
