"""Check semantic similarity between labeled source and target text pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def check_labeled_similarity(
    input_path: str | Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    threshold: float = 0.70,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Score labeled source/target pairs, not tuned-model inference output."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    from sentence_transformers import SentenceTransformer, util

    rows = [
        json.loads(line)
        for line in Path(input_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    valid_rows = [
        row for row in rows
        if row.get("source") and row.get("target")
        and "labeling_error" not in row.get("issues", [])
    ]
    if not valid_rows:
        raise ValueError("No labeled source/target pairs available for similarity checking")

    print(f"Loading similarity model: {model_name}", flush=True)
    model = SentenceTransformer(model_name)
    scores: list[float] = []
    for start in range(0, len(valid_rows), batch_size):
        batch = valid_rows[start:start + batch_size]
        source_embeddings = model.encode(
            [row["source"] for row in batch],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        target_embeddings = model.encode(
            [row["target"] for row in batch],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores.extend(util.cos_sim(source_embeddings, target_embeddings).diagonal().tolist())
        print(
            f"Scored {min(start + batch_size, len(valid_rows))}/{len(valid_rows)} pairs",
            flush=True,
        )

    results = [
        {
            "id": row["id"],
            "split": row.get("split"),
            "source": row["source"],
            "target": row["target"],
            "similarity": round(float(score), 6),
            "below_threshold": float(score) < threshold,
        }
        for row, score in zip(valid_rows, scores)
    ]
    return {
        "input": str(input_path),
        "model": model_name,
        "threshold": threshold,
        "pairs_scored": len(results),
        "below_threshold": sum(item["below_threshold"] for item in results),
        "mean_similarity": sum(item["similarity"] for item in results) / len(results),
        "pairs": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check semantic similarity of labeled source/target pairs."
    )
    parser.add_argument("--input", default="data/labeled/pilot.jsonl")
    parser.add_argument("--report", default="data/labeled/similarity.json")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    report = check_labeled_similarity(
        args.input,
        model_name=args.model,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    below_threshold = [pair for pair in report["pairs"] if pair["below_threshold"]]
    if below_threshold:
        print(
            f"\nPairs below similarity threshold ({args.threshold:.2f}):",
            flush=True,
        )
        for pair in below_threshold:
            print(f"\n[{pair['id']}] similarity={pair['similarity']:.6f}", flush=True)
            print(f"Source: {pair['source']}", flush=True)
            print(f"Target: {pair['target']}", flush=True)
    else:
        print(f"\nAll pairs meet the similarity threshold ({args.threshold:.2f}).", flush=True)
    print(
        f"Similarity check complete: {report['pairs_scored']} pairs, "
        f"{report['below_threshold']} below threshold; report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
