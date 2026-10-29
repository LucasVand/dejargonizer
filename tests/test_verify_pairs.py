import hashlib
import json

from dejargonizer.verify_pairs import repair_copy, verify_pairs
from dejargonizer.combine_pairs import combine_pairs


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


def test_combine_pairs_preserves_order_and_skips_exact_duplicates(tmp_path):
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    output_path = tmp_path / "combined.jsonl"
    first = make_pair()
    second = {**first, "id": "pair-2", "source": "Another technical source."}
    second["source_hash"] = hashlib.sha256(second["source"].encode()).hexdigest()
    first_path.write_text(json.dumps(first) + "\n")
    second_path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")

    result = combine_pairs([first_path, second_path], output_path)

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [row["id"] for row in rows] == ["pair-1", "pair-2"]
    assert result["duplicates_skipped"] == 1


def test_combine_pairs_prefers_first_conflicting_duplicate_source(tmp_path):
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    output_path = tmp_path / "combined.jsonl"
    first = make_pair()
    second = {**first, "target": "Different target."}
    first_path.write_text(json.dumps(first) + "\n")
    second_path.write_text(json.dumps(second) + "\n")

    result = combine_pairs([first_path, second_path], output_path)

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows == [first]
    assert result["duplicates_skipped"] == 1


def test_combine_pairs_prefers_first_duplicate_id(tmp_path):
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    output_path = tmp_path / "combined.jsonl"
    first = make_pair()
    second = {
        **first,
        "source": "Different source.",
        "target": "Different target.",
        "source_hash": hashlib.sha256("Different source.".encode()).hexdigest(),
    }
    first_path.write_text(json.dumps(first) + "\n")
    second_path.write_text(json.dumps(second) + "\n")

    result = combine_pairs([first_path, second_path], output_path)

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert rows == [first]
    assert result["duplicates_skipped"] == 1


def test_combine_pairs_deduplicates_records_within_one_input(tmp_path):
    first_path = tmp_path / "first.jsonl"
    second_path = tmp_path / "second.jsonl"
    output_path = tmp_path / "combined.jsonl"
    first = make_pair()
    second = {**first, "target": "A later target for the same pair."}
    first_path.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    other = {
        **first,
        "id": "pair-2",
        "source": "Another technical source.",
        "source_hash": hashlib.sha256("Another technical source.".encode()).hexdigest(),
    }
    second_path.write_text(json.dumps(other) + "\n")

    result = combine_pairs([first_path, second_path], output_path)

    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [row["id"] for row in rows] == ["pair-1", "pair-2"]
    assert rows[0]["target"] == first["target"]
    assert result["duplicates_skipped"] == 1
