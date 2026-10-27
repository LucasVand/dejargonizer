"""Evaluate simplification quality with overlap and readability metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .inference import Dejargonizer


def evaluate_file(
    input_path: str | Path,
    checkpoint: str | Path,
    progress_interval: int = 10,
) -> dict[str, Any]:
    import evaluate
    import textstat

    if progress_interval < 1:
        raise ValueError("progress_interval must be positive")
    all_rows = [
        json.loads(line)
        for line in Path(input_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in all_rows if row.get("split") == "test" and not row.get("issues")]
    if not rows:
        raise ValueError("No accepted test examples found")
    print(
        f"Evaluation set: {len(rows)} accepted test pairs "
        f"from {len(all_rows)} labeled records",
        flush=True,
    )
    print(f"Loading model checkpoint: {checkpoint}", flush=True)
    model = Dejargonizer(checkpoint)
    predictions: list[str] = []
    for index, row in enumerate(rows, start=1):
        prediction = model.simplify(row["source"])
        predictions.append(prediction)
        if index == 1 or index % progress_interval == 0 or index == len(rows):
            print(
                f"Generated predictions: {index}/{len(rows)} "
                f"(latest pair: {row.get('id', '<unknown>')})",
                flush=True,
            )
    references = [row["target"] for row in rows]
    print("Calculating SARI and readability metrics...", flush=True)
    sari = evaluate.load("sari").compute(
        sources=[row["source"] for row in rows],
        predictions=predictions,
        references=[[reference] for reference in references],
    )
    result = {
        "sari": sari,
        "source_readability": sum(textstat.flesch_reading_ease(row["source"]) for row in rows) / len(rows),
        "prediction_readability": sum(textstat.flesch_reading_ease(text) for text in predictions) / len(predictions),
    }
    print(
        f"Evaluation complete: SARI={sari.get('sari', sari)}, "
        f"source readability={result['source_readability']:.2f}, "
        f"prediction readability={result['prediction_readability']:.2f}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/labeled/pilot.jsonl")
    parser.add_argument("--checkpoint", default="artifacts/t5-small-pilot")
    parser.add_argument("--progress-interval", type=int, default=10)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate_file(args.input, args.checkpoint, args.progress_interval),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
