import json

from dejargonizer.labeling import label_records


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
