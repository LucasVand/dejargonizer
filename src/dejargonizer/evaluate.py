"""Evaluate simplification quality with overlap and readability metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .inference import Dejargonizer


def evaluate_file(input_path: str | Path, checkpoint: str | Path) -> dict[str, Any]:
    import evaluate
    import textstat

    rows = [json.loads(line) for line in Path(input_path).read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row.get("split") == "test" and not row.get("issues")]
    if not rows:
        raise ValueError("No accepted test examples found")
    model = Dejargonizer(checkpoint)
    predictions = [model.simplify(row["source"]) for row in rows]
    references = [row["target"] for row in rows]
    sari = evaluate.load("sari").compute(
        sources=[row["source"] for row in rows],
        predictions=predictions,
        references=[[reference] for reference in references],
    )
    return {
        "sari": sari,
        "source_readability": sum(textstat.flesch_reading_ease(row["source"]) for row in rows) / len(rows),
        "prediction_readability": sum(textstat.flesch_reading_ease(text) for text in predictions) / len(predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/labeled/pilot.jsonl")
    parser.add_argument("--checkpoint", default="artifacts/t5-small-pilot")
    args = parser.parse_args()
    print(json.dumps(evaluate_file(args.input, args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
