"""Validate the structure and integrity of generated JSONL training pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "id",
    "split",
    "source",
    "target",
    "source_hash",
    "label_provider",
    "label_model",
    "prompt_version",
    "issues",
}
VALID_SPLITS = {"train", "validation", "test"}


def verify_pairs(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Pair file not found: {path}")

    errors: list[str] = []
    warnings: list[str] = []
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line.strip():
            errors.append(f"line {line_number}: blank line")
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as error:
            errors.append(f"line {line_number}: invalid JSON: {error.msg}")
            continue
        if not isinstance(row, dict):
            errors.append(f"line {line_number}: record must be a JSON object")
            continue

        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            errors.append(f"line {line_number}: missing fields: {', '.join(sorted(missing))}")
            continue
        if not all(isinstance(row[field], str) for field in (
            "id", "split", "source", "target", "source_hash",
            "label_provider", "label_model", "prompt_version",
        )):
            errors.append(f"line {line_number}: string fields have invalid types")
        if row["split"] not in VALID_SPLITS:
            errors.append(f"line {line_number}: invalid split {row['split']!r}")
        if not isinstance(row["issues"], list) or not all(
            isinstance(issue, str) for issue in row["issues"]
        ):
            errors.append(f"line {line_number}: issues must be a list of strings")
        if not row["source"].strip():
            errors.append(f"line {line_number}: source is empty")
        if not row["target"].strip():
            errors.append(f"line {line_number}: target is empty")
        expected_hash = hashlib.sha256(row["source"].encode()).hexdigest()
        if row["source_hash"] != expected_hash:
            errors.append(f"line {line_number}: source_hash does not match source")
        if row["id"] in seen_ids:
            errors.append(f"line {line_number}: duplicate id {row['id']!r}")
        if row["source_hash"] in seen_hashes:
            warnings.append(f"line {line_number}: duplicate source_hash {row['source_hash']}")
        seen_ids.add(row["id"])
        seen_hashes.add(row["source_hash"])
        rows.append(row)

    return {
        "path": str(path),
        "records": len(rows),
        "errors": errors,
        "warnings": warnings,
        "valid": not errors and bool(rows),
    }


def repair_copy(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Write a repaired copy, removing only exact duplicate records.

    The input is never modified. Malformed, incomplete, and conflicting records are
    retained so they remain visible to the follow-up integrity check.
    """
    source = Path(input_path).resolve()
    destination = Path(output_path).resolve()
    if source == destination:
        raise ValueError("Repair output must be different from the original input")
    if not source.is_file():
        raise FileNotFoundError(f"Pair file not found: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    seen_fingerprints: set[str] = set()
    kept_lines: list[str] = []
    removed: list[str] = []
    for line_number, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            kept_lines.append(raw_line)
            continue
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError:
            kept_lines.append(raw_line)
            continue
        if not isinstance(row, dict):
            kept_lines.append(raw_line)
            continue
        fingerprint = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if fingerprint in seen_fingerprints:
            removed.append(
                f"line {line_number}: removed exact duplicate of an earlier complete record "
                f"(id={row.get('id', '<missing>')!r})"
            )
            continue
        seen_fingerprints.add(fingerprint)
        kept_lines.append(raw_line)

    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return {
        "input": str(source),
        "output": str(destination),
        "removed": removed,
        "records_written": len(kept_lines),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify generated training-pair integrity.")
    parser.add_argument("--input", default="data/labeled/pilot.jsonl")
    parser.add_argument(
        "--repair-copy",
        help="Write a repaired copy; never modifies --input.",
    )
    args = parser.parse_args()
    if args.repair_copy:
        try:
            repair = repair_copy(args.input, args.repair_copy)
        except (FileNotFoundError, OSError, ValueError) as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print(f"Original unchanged: {repair['input']}")
        print(f"Repaired copy written: {repair['output']}")
        if repair["removed"]:
            print("Repairs made:")
            for action in repair["removed"]:
                print(f"  {action}")
        else:
            print("Repairs made: none; no exact duplicate records found.")
        print(
            "No malformed, incomplete, or conflicting records were modified; "
            "run integrity verification on the copy."
        )
        args.input = args.repair_copy
    try:
        report = verify_pairs(args.input)
    except (FileNotFoundError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print(f"Checked {report['records']} records in {report['path']}")
    for warning in report["warnings"]:
        print(f"WARNING: {warning}")
    for error in report["errors"]:
        print(f"ERROR: {error}", file=sys.stderr)
    if report["valid"]:
        print("Pair integrity check passed.")
        return 0
    print("Pair integrity check failed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
