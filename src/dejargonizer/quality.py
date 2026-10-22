"""Validation of generated simplification labels."""

from __future__ import annotations

import re

NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")


def validate_target(source: str, target: str) -> list[str]:
    issues: list[str] = []
    target = target.strip()
    if not target:
        return ["empty_target"]
    if target == source.strip():
        return ["unchanged"]
    if target.lower().startswith(("here is", "sure,", "simplified version")):
        issues.append("meta_commentary")
    if len(target.split()) < max(8, len(source.split()) // 10):
        issues.append("possible_omission")
    source_numbers = set(NUMBER_RE.findall(source))
    target_numbers = set(NUMBER_RE.findall(target))
    if not source_numbers.issubset(target_numbers):
        issues.append("number_changed_or_missing")
    return issues
