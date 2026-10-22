"""Validation of generated simplification labels."""

from __future__ import annotations

import re

NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
WRAPPER_PREFIXES = (
    re.compile(r"^\s*here(?:'s| is) a rewritten version of the text in simpler language\s*:?\s*", re.I),
    re.compile(r"^\s*here(?:'s| is) the simplified version\s*:?\s*", re.I),
    re.compile(r"^\s*simplified version\s*:?\s*", re.I),
)


def clean_target(target: str) -> str:
    """Remove provider framing so only the generated simplification is trained."""
    cleaned = target.strip()
    for prefix in WRAPPER_PREFIXES:
        cleaned = prefix.sub("", cleaned, count=1).strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = re.sub(r"^```(?:text|markdown)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return cleaned


def validate_target(source: str, target: str) -> list[str]:
    issues: list[str] = []
    target = clean_target(target)
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
