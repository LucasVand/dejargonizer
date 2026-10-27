import json

from dejargonizer.labeling import label_records
from dejargonizer.labeling import _ollama_target, _performance_message


def test_split_filter_only_labels_requested_records(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "labels.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": split,
                    "split": split,
                    "source": f"Source text for {split}.",
                }
            )
            for split in ("train", "validation")
        )
        + "\n"
    )
    monkeypatch.setattr(
        "dejargonizer.labeling._cloudflare_target",
        lambda source, labeling: f"Simple {source}",
    )
    label_records(
        source,
        output,
        {"labeling": {"provider": "cloudflare", "model": "test", "prompt_version": "v1"}},
        split_filter="validation",
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["split"] for row in rows] == ["validation"]


def test_ollama_target_parses_chat_response(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "A simple explanation."}}

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: Response())
    assert _ollama_target(
        "A technical sentence.",
        {"endpoint": "http://ollama.test", "model": "test-model"},
    ) == "A simple explanation."


def test_performance_message_reports_rate_and_eta():
    message = _performance_message(processed=25, pending=100, elapsed=50)
    assert "0.50 queries/s" in message
    assert "ETA 150s" in message


def test_performance_message_includes_existing_records():
    message = _performance_message(
        processed=8,
        pending=16351,
        elapsed=4,
        existing=3649,
        total=20000,
    )
    assert "Progress: 3657/20000 records" in message
    assert "8/16351 new requests" in message


def test_labeling_ignores_malformed_existing_output(tmp_path, monkeypatch):
    source = tmp_path / "source.jsonl"
    output = tmp_path / "labels.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "validation",
                "split": "validation",
                "source": "Source text for validation.",
            }
        )
        + "\n"
    )
    output.write_text("\nnot valid json\n")
    monkeypatch.setattr(
        "dejargonizer.labeling._cloudflare_target",
        lambda source, labeling: f"Simple {source}",
    )
    label_records(
        source,
        output,
        {"labeling": {"provider": "cloudflare", "model": "test", "prompt_version": "v1"}},
    )
    rows = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    assert rows[-1]["target"] == "Simple Source text for validation."
