"""Generate auditable silver labels with Cloudflare Workers AI or Ollama."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from .quality import clean_target, validate_target

SYSTEM_PROMPT = """Rewrite the technical English aggressively for a general audience with no
specialist background. Prefer familiar everyday words over technical vocabulary. Replace jargon
with plain explanations whenever possible, and briefly explain technical terms that must remain.
Break long or complicated sentences into short, direct sentences. Make the result substantially
easier to read while preserving the complete original meaning, facts, numbers, units, names,
uncertainty, comparisons, and limitations. Do not weaken, strengthen, or invent claims. Do not
summarize away important details.

Your response must contain ONLY the rewritten plain-English text. Do not include an introduction,
preface, explanation, heading, label, quotation marks, Markdown, code fences, or any statement
such as "Here is the simplified version." Say nothing before or after the rewrite."""


def _status_message(pending_count: int, total_count: int) -> str:
    if pending_count == 0:
        return f"Labeling complete: {total_count}/{total_count} records already labeled"
    completed_count = total_count - pending_count
    return (
        f"Starting labeling: {completed_count}/{total_count} records already labeled; "
        f"{pending_count} pending"
    )


def _performance_message(
    processed: int,
    pending: int,
    elapsed: float,
    *,
    existing: int = 0,
    total: int | None = None,
) -> str:
    rate = processed / elapsed if elapsed > 0 else 0.0
    remaining = max(pending - processed, 0)
    eta_seconds = remaining / rate if rate > 0 else None
    eta = "unknown" if eta_seconds is None else f"{eta_seconds:.0f}s"
    total = existing + pending if total is None else total
    completed = min(existing + processed, total)
    return (
        f"Progress: {completed}/{total} records; "
        f"{processed}/{pending} new requests; "
        f"{rate:.2f} queries/s; elapsed {elapsed:.0f}s; ETA {eta}"
    )


def _cloudflare_target(source: str, labeling: dict[str, Any]) -> str:
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not account_id or not api_token:
        raise RuntimeError(
            "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are required for labeling"
        )
    import httpx

    response = httpx.post(
        f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{labeling['model']}",
        headers={"Authorization": f"Bearer {api_token}"},
        json={
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": source},
            ],
            "temperature": float(labeling["temperature"]),
            "max_tokens": int(labeling["max_tokens"]),
        },
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success", False):
        raise RuntimeError(f"Cloudflare Workers AI request failed: {payload.get('errors', payload)}")
    target = payload.get("result", {}).get("response")
    if not isinstance(target, str) or not target.strip():
        raise RuntimeError(f"Cloudflare Workers AI returned no text: {payload}")
    return target.strip()


def _ollama_target(source: str, labeling: dict[str, Any]) -> str:
    import httpx

    endpoint = labeling.get("endpoint", "http://100.97.198.56:11434").rstrip("/")
    response = httpx.post(
        f"{endpoint}/api/chat",
        json={
            "model": labeling["model"],
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": source},
            ],
            "stream": False,
            "options": {
                "temperature": float(labeling.get("temperature", 0.2)),
                "num_predict": int(labeling.get("max_tokens", 300)),
            },
        },
        timeout=float(labeling.get("timeout", 300)),
    )
    response.raise_for_status()
    payload = response.json()
    message = payload.get("message", {})
    target = message.get("content") if isinstance(message, dict) else None
    if not isinstance(target, str) or not target.strip():
        raise RuntimeError(f"Ollama returned no text: {payload}")
    return target.strip()


def label_records(
    input_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any],
    split_filter: str | None = None,
) -> None:
    labeling = config["labeling"]
    provider = labeling.get("provider", "cloudflare").lower()
    if provider not in {"cloudflare", "ollama"}:
        raise ValueError(f"Unsupported labeling provider: {provider}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    valid_existing_lines: list[str] = []
    if output.exists():
        for line_number, line in enumerate(
            output.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                completed.add(item["source_hash"])
                valid_existing_lines.append(line)
            except (json.JSONDecodeError, KeyError) as error:
                print(
                    f"Warning: ignoring malformed labeled-output line {line_number}: "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )

    records = [
        json.loads(line)
        for line in Path(input_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if split_filter is not None:
        records = [record for record in records if record.get("split") == split_filter]
    pending = [
        record
        for record in records
        if hashlib.sha256(record["source"].encode()).hexdigest() not in completed
    ]
    existing = len(records) - len(pending)
    print(_status_message(len(pending), len(records)), flush=True)
    started_at = time.monotonic()
    processed = 0

    with output.open("w", encoding="utf-8") as target_file:
        for line in valid_existing_lines:
            target_file.write(line + "\n")
        for record in records:
            source_hash = hashlib.sha256(record["source"].encode()).hexdigest()
            if source_hash in completed:
                continue
            progress_index = existing + processed + 1
            print(
                f"[{progress_index}/{len(records)}] Labeling {record['id']}...",
                flush=True,
            )
            try:
                target = (
                    _ollama_target(record["source"], labeling)
                    if provider == "ollama"
                    else _cloudflare_target(record["source"], labeling)
                )
                target = clean_target(target)
                issues = validate_target(record["source"], target)
            except Exception as error:
                print(
                    f"[{progress_index}/{len(records)}] ERROR for {record['id']}: "
                    f"{type(error).__name__}: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                item = {
                    "id": record["id"],
                    "split": record["split"],
                    "source": record["source"],
                    "target": "",
                    "source_hash": source_hash,
                    "label_provider": provider,
                    "label_model": labeling["model"],
                    "prompt_version": labeling["prompt_version"],
                    "issues": ["labeling_error"],
                    "error": f"{type(error).__name__}: {error}",
                }
                target_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                target_file.flush()
                processed += 1
                print(
                    _performance_message(
                        processed,
                        len(pending),
                        time.monotonic() - started_at,
                        existing=existing,
                        total=len(records),
                    ),
                    flush=True,
                )
                continue
            item = {
                "id": record["id"],
                "split": record["split"],
                "source": record["source"],
                "target": target,
                "source_hash": source_hash,
                "label_provider": provider,
                "label_model": labeling["model"],
                "prompt_version": labeling["prompt_version"],
                "issues": issues,
            }
            target_file.write(json.dumps(item, ensure_ascii=False) + "\n")
            target_file.flush()
            completed.add(source_hash)
            processed += 1
            print(f"[{progress_index}/{len(records)}] Complete", flush=True)
            print(
                _performance_message(
                    processed,
                    len(pending),
                    time.monotonic() - started_at,
                    existing=existing,
                    total=len(records),
                ),
                flush=True,
            )
    elapsed = time.monotonic() - started_at
    print(
        f"Labeling finished: {processed} requests in {elapsed:.0f}s; "
        f"{processed / elapsed if elapsed > 0 else 0.0:.2f} queries/s",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/interim/pilot.jsonl")
    parser.add_argument("--output", default="data/labeled/pilot.jsonl")
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--provider", choices=("cloudflare", "ollama"))
    args = parser.parse_args()
    import yaml

    with Path(args.config).open() as file:
        config = yaml.safe_load(file)
    if args.provider:
        config["labeling"]["provider"] = args.provider
    label_records(args.input, args.output, config, split_filter=args.split)


if __name__ == "__main__":
    main()
