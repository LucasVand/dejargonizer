"""Evaluate simplification quality with overlap and readability metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .inference import Dejargonizer

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[./%-][A-Za-z0-9]+)*")
_NUMBER_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?%?")


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_PATTERN.findall(text)}


def _numbers(text: str) -> list[str]:
    return _NUMBER_PATTERN.findall(text)


def _jaccard(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens and not right_tokens:
        return 1.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _evaluation_quality(
    sources: list[str],
    predictions: list[str],
    references: list[str],
    textstat: Any,
) -> dict[str, float]:
    source_words = [len(_tokens(text)) for text in sources]
    prediction_words = [len(_tokens(text)) for text in predictions]
    reference_words = [len(_tokens(text)) for text in references]
    source_readability = [
        textstat.flesch_reading_ease(text) for text in sources
    ]
    prediction_readability = [
        textstat.flesch_reading_ease(text) for text in predictions
    ]
    number_scores = []
    for source, prediction in zip(sources, predictions):
        source_numbers = _numbers(source)
        prediction_numbers = _numbers(prediction)
        prediction_counts = {number: prediction_numbers.count(number) for number in set(prediction_numbers)}
        preserved = sum(
            min(source_numbers.count(number), prediction_counts.get(number, 0))
            for number in set(source_numbers)
        )
        number_scores.append(preserved / len(source_numbers) if source_numbers else 1.0)

    return {
        "source_readability": sum(source_readability) / len(sources),
        "prediction_readability": sum(prediction_readability) / len(predictions),
        "reference_readability": sum(
            textstat.flesch_reading_ease(text) for text in references
        ) / len(references),
        "readability_improvement": (
            sum(prediction_readability) - sum(source_readability)
        ) / len(sources),
        "average_source_words": sum(source_words) / len(sources),
        "average_prediction_words": sum(prediction_words) / len(predictions),
        "average_reference_words": sum(reference_words) / len(references),
        "prediction_word_reduction_percent": (
            sum(
                (source - prediction) / source * 100
                for source, prediction in zip(source_words, prediction_words)
                if source
            ) / sum(bool(source) for source in source_words)
        ),
        "average_source_prediction_overlap": sum(
            _jaccard(source, prediction)
            for source, prediction in zip(sources, predictions)
        ) / len(sources),
        "average_reference_prediction_overlap": sum(
            _jaccard(reference, prediction)
            for reference, prediction in zip(references, predictions)
        ) / len(references),
        "source_numbers_preserved_rate": sum(number_scores) / len(number_scores),
        "empty_prediction_rate": sum(not prediction.strip() for prediction in predictions)
        / len(predictions),
        "unchanged_prediction_rate": sum(
            source.strip().casefold() == prediction.strip().casefold()
            for source, prediction in zip(sources, predictions)
        ) / len(predictions),
    }


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
    sources = [row["source"] for row in rows]
    quality = _evaluation_quality(sources, predictions, references, textstat)
    result = {"sari": sari, **quality}
    print(
        f"Evaluation complete: SARI={sari.get('sari', sari)}, "
        f"source readability={quality['source_readability']:.2f}, "
        f"prediction readability={quality['prediction_readability']:.2f}, "
        f"readability improvement={quality['readability_improvement']:.2f}, "
        f"number preservation={quality['source_numbers_preserved_rate']:.1%}",
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
