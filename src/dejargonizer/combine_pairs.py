"""Combine generated JSONL training-pair files without creating duplicate records."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from .verify_pairs import verify_pairs

_DUPLICATE_ID_ERROR = re.compile(r"^line \d+: duplicate id ")


def combine_pairs(
    input_paths: list[str | Path], output_path: str | Path
) -> dict[str, Any]:
    """Combine valid pair files, with records from earlier inputs taking precedence."""
    paths = [Path(path).resolve() for path in input_paths]
    output = Path(output_path).resolve()
    if output in paths:
        raise ValueError("Output file must be different from every input file")
    if len(paths) < 2:
        raise ValueError("At least two input files are required")

    for path in paths:
        report = verify_pairs(path)
        blocking_errors = [
            error
            for error in report["errors"]
            if not _DUPLICATE_ID_ERROR.match(error)
        ]
        if blocking_errors:
            details = "; ".join(blocking_errors[:3])
            if len(blocking_errors) > 3:
                details += f"; and {len(blocking_errors) - 3} more error(s)"
            raise ValueError(f"Cannot combine invalid pair file {path}: {details}")

    output.parent.mkdir(parents=True, exist_ok=True)
    seen_ids: dict[str, str] = {}
    seen_hashes: dict[str, str] = {}
    records_written = 0
    duplicates_skipped = 0
    temporary = output.with_suffix(output.suffix + ".tmp")

    try:
        with temporary.open("w", encoding="utf-8") as destination:
            for path in paths:
                for line_number, raw_line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    row = json.loads(raw_line)
                    fingerprint = json.dumps(
                        row, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                    )
                    row_id = row["id"]
                    source_hash = row["source_hash"]
                    previous_id = seen_ids.get(row_id)
                    previous_hash = seen_hashes.get(source_hash)
                    if previous_id is not None or previous_hash is not None:
                        duplicates_skipped += 1
                        continue
                    destination.write(json.dumps(row, ensure_ascii=False) + "\n")
                    seen_ids[row_id] = fingerprint
                    seen_hashes[source_hash] = fingerprint
                    records_written += 1
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    return {
        "inputs": [str(path) for path in paths],
        "output": str(output),
        "records_written": records_written,
        "duplicates_skipped": duplicates_skipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Combine two or more generated JSONL training-pair files."
    )
    parser.add_argument("inputs", nargs="+", help="Input pair JSONL files")
    parser.add_argument("--output", required=True, help="Combined output JSONL file")
    args = parser.parse_args()
    try:
        result = combine_pairs(args.inputs, args.output)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Combined {len(result['inputs'])} files into {result['output']}")
    print(f"Records written: {result['records_written']}")
    print(
        f"Duplicate records skipped (earlier input preferred): "
        f"{result['duplicates_skipped']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
