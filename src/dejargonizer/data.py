"""Bounded, streaming-only acquisition and preparation of arXiv abstracts."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable



def load_config(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open() as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML object")
    return config


def _text(record: dict[str, Any]) -> str:
    for key in ("abstract", "text", "content"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())
    raise KeyError("Could not find a non-empty abstract/text/content field")


def _stable_id(record: dict[str, Any], text: str) -> str:
    for key in ("id", "paper_id", "arxiv_id"):
        if record.get(key) is not None:
            return str(record[key])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def _eligible(text: str, config: dict[str, Any]) -> bool:
    words = text.split()
    dataset_config = config["dataset"]
    return (
        dataset_config["min_words"] <= len(words) <= dataset_config["max_words"]
        and len(words) <= int(dataset_config["max_source_tokens"])
    )


def _sentences(text: str) -> list[str]:
    """Split ordinary prose without downloading or requiring a full NLP model."""
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text) if part.strip()]


def _quota_targets(sample_size: int, percentages: list[float]) -> list[int]:
    if len(percentages) != 9 or any(value < 0 for value in percentages):
        raise ValueError("chunks.quota_percentages must contain 9 non-negative values")
    if abs(sum(percentages) - 100) > 0.01:
        raise ValueError("chunks.quota_percentages must total 100")
    raw = [sample_size * value / 100 for value in percentages]
    targets = [int(value) for value in raw]
    for index in sorted(range(9), key=lambda item: raw[item] - targets[item], reverse=True):
        if sum(targets) >= sample_size:
            break
        targets[index] += 1
    return targets


def _select_quota_chunk(
    record: dict[str, Any],
    config: dict[str, Any],
    remaining: list[int],
) -> dict[str, Any] | None:
    chunks = config["dataset"].get("chunks", {})
    if not chunks.get("enabled", False):
        return record
    sentences = _sentences(record["source"])
    max_sentences = int(chunks.get("max_sentences", 8))
    eligible_lengths = [length for length in range(1, max_sentences + 1) if len(sentences) >= length]
    eligible_categories = [length - 1 for length in eligible_lengths] + [8]
    category = max(eligible_categories, key=lambda item: remaining[item])
    if remaining[category] <= 0:
        return None
    if category == 8:
        return {
            **record,
            "id": f"{record['id']}:full",
            "metadata": {
                **record.get("metadata", {}),
                "parent_id": record["id"],
                "chunk_start": 1,
                "chunk_end": len(sentences),
                "chunk_sentences": len(sentences),
                "quota_category": 8,
            },
        }
    length = category + 1
    start = (int(hashlib.sha256(record["id"].encode()).hexdigest(), 16) % (len(sentences) - length + 1))
    return {
        **record,
        "id": f"{record['id']}:s{start + 1}-{start + length}",
        "source": " ".join(sentences[start:start + length]),
        "metadata": {
            **record.get("metadata", {}),
            "parent_id": record["id"],
            "chunk_start": start + 1,
            "chunk_end": start + length,
            "chunk_sentences": length,
            "quota_category": category,
        },
    }


def stream_pilot(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Read at most max_scan_records remotely and return at most sample_size records.

    This function deliberately rejects non-streaming access and requires both bounds.
    """
    dataset_config = config["dataset"]
    sample_size = int(dataset_config["sample_size"])
    max_scan = int(dataset_config["max_scan_records"])
    if sample_size < 1 or max_scan < sample_size:
        raise ValueError("sample_size must be positive and no greater than max_scan_records")
    if not dataset_config.get("max_source_tokens"):
        raise ValueError("dataset.max_source_tokens is required")

    from datasets import load_dataset

    stream = load_dataset(
        dataset_config["name"],
        split=dataset_config["split"],
        revision=dataset_config.get("revision"),
        streaming=True,
    )
    if hasattr(stream, "shuffle"):
        stream = stream.shuffle(
            seed=int(dataset_config["seed"]),
            buffer_size=int(dataset_config["shuffle_buffer"]),
        )

    selected: list[dict[str, Any]] = []
    chunking_enabled = config["dataset"].get("chunks", {}).get("enabled", False)
    remaining = _quota_targets(
        sample_size,
        config["dataset"].get("chunks", {}).get("quota_percentages", [100]),
    ) if chunking_enabled else []
    eligible_count = 0
    progress_interval = max(1, min(100, max_scan // 20))
    print(
        f"Streaming dataset: selecting {sample_size} records; "
        f"scan limit={max_scan}",
        flush=True,
    )
    for scanned, record in enumerate(itertools.islice(stream, max_scan), start=1):
        try:
            text = _text(record)
        except KeyError:
            if scanned % progress_interval == 0:
                print(
                    f"Progress: scanned {scanned}/{max_scan}; "
                    f"eligible {eligible_count}; selected {len(selected)}/{sample_size}",
                    flush=True,
                )
            continue
        if not _eligible(text, config):
            if scanned % progress_interval == 0:
                print(
                    f"Progress: scanned {scanned}/{max_scan}; "
                    f"eligible {eligible_count}; selected {len(selected)}/{sample_size}",
                    flush=True,
                )
            continue
        eligible_count += 1
        base_record = {
            "id": _stable_id(record, text),
            "source": text,
            "metadata": {
                key: record[key]
                for key in ("title", "categories", "authors")
                if key in record
            },
        }
        if chunking_enabled:
            chunk = _select_quota_chunk(base_record, config, remaining)
            if chunk is None:
                continue
            selected.append(chunk)
            category = chunk["metadata"]["quota_category"]
            remaining[category] -= 1
        else:
            selected.append(base_record)
        if scanned % progress_interval == 0 or len(selected) >= sample_size:
            print(
                f"Progress: scanned {scanned}/{max_scan}; "
                f"eligible {eligible_count}; selected {len(selected)}/{sample_size}",
                flush=True,
            )
        if len(selected) >= sample_size:
            break
    if not selected:
        raise RuntimeError("No eligible records found within max_scan_records")
    print(
        f"Stream complete: scanned {scanned} records; "
        f"selected {len(selected)} records",
        flush=True,
    )
    return selected


def split_records(records: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Assign deterministic splits without duplicating source records."""
    records = list(records)
    random.Random(seed).shuffle(records)
    n = len(records)
    for index, record in enumerate(records):
        fraction = index / n
        record["split"] = "train" if fraction < 0.8 else "validation" if fraction < 0.9 else "test"
    return records


def write_pilot(records: Iterable[dict[str, Any]], output_dir: str | Path, config: dict[str, Any]) -> Path:
    records = list(records)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "interim" / "pilot.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    record_count = len(records)
    manifest = output / "interim" / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset": config["dataset"]["name"],
                "revision": config["dataset"].get("revision"),
                "sample_size": record_count,
                "max_scan_records": config["dataset"]["max_scan_records"],
                "seed": config["dataset"]["seed"],
            },
            indent=2,
        )
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    records = split_records(stream_pilot(config), int(config["dataset"]["seed"]))
    path = write_pilot(records, config["dataset"]["output_dir"], config)
    print(f"Wrote {len(records)} bounded pilot records to {path}")


if __name__ == "__main__":
    main()
