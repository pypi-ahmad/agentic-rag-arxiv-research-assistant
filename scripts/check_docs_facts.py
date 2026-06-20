#!/usr/bin/env python3
"""Validate docs claims against artifacts and scan for unsafe text patterns."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = ROOT / "docs"
CLAIMS_FILE = DOCS_ROOT / "08_reference" / "claims_traceability.md"

DEPRECATED_PATTERNS: dict[str, str] = {
    r"granite4\.1-guardian": "Deprecated judge model reference found.",
    r"_pending run_": "Placeholder marker found.",
}

SECRET_PATTERNS: dict[str, str] = {
    r"sk-[A-Za-z0-9]{20,}": "Potential OpenAI-style API key found.",
    r"hf_[A-Za-zA-Z0-9]{20,}": "Potential Hugging Face token found.",
    r"AKIA[0-9A-Z]{16}": "Potential AWS access key found.",
    r"AIza[0-9A-Za-z\-_]{35}": "Potential Google API key found.",
    r"xox[baprs]-[A-Za-z0-9-]{10,}": "Potential Slack token found.",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----": "Private key header found.",
}

TABLE_HEADER = "| Claim ID | Claim | Artifact | Check values |"


@dataclass(frozen=True)
class ClaimRow:
    claim_id: str
    artifact: str
    checks: tuple[str, ...]


def markdown_files() -> list[Path]:
    files = [ROOT / "README.md"]
    files.extend(sorted(DOCS_ROOT.rglob("*.md")))
    return files


def strip_ticks(value: str) -> str:
    return value.strip().strip("`").strip()


def parse_claim_rows(text: str) -> list[ClaimRow]:
    lines = text.splitlines()
    start_idx = -1
    for idx, line in enumerate(lines):
        if line.strip() == TABLE_HEADER:
            start_idx = idx
            break
    if start_idx == -1:
        raise ValueError(f"Could not find claims table header in {CLAIMS_FILE}")

    rows: list[ClaimRow] = []
    for line in lines[start_idx + 2 :]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            break
        parts = [p.strip() for p in stripped.strip("|").split("|")]
        if len(parts) != 4:
            continue
        claim_id = parts[0]
        artifact = strip_ticks(parts[2])
        checks_text = strip_ticks(parts[3])
        checks = tuple(c.strip() for c in checks_text.split(";") if c.strip())
        rows.append(ClaimRow(claim_id=claim_id, artifact=artifact, checks=checks))
    return rows


def try_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def lookup_path(data: Any, dotted_path: str) -> Any:
    current = data
    for key in dotted_path.split("."):
        if not isinstance(current, dict) or key not in current:
            raise KeyError(dotted_path)
        current = current[key]
    return current


def validate_claim_rows(rows: list[ClaimRow]) -> list[str]:
    failures: list[str] = []
    json_cache: dict[Path, Any] = {}

    for row in rows:
        artifact_path = (ROOT / row.artifact).resolve()
        if not artifact_path.exists():
            failures.append(f"{row.claim_id}: missing artifact `{row.artifact}`")
            continue

        artifact_is_json = artifact_path.suffix.lower() == ".json"
        artifact_text: str | None = None
        artifact_data: Any = None

        if artifact_is_json:
            if artifact_path not in json_cache:
                json_cache[artifact_path] = json.loads(
                    artifact_path.read_text(encoding="utf-8")
                )
            artifact_data = json_cache[artifact_path]

        for raw_check in row.checks:
            if "=" not in raw_check:
                failures.append(f"{row.claim_id}: malformed check `{raw_check}`")
                continue

            key, expected_str = raw_check.split("=", maxsplit=1)
            key = key.strip()
            expected_str = expected_str.strip()

            if key == "contains":
                if artifact_text is None:
                    artifact_text = artifact_path.read_text(
                        encoding="utf-8", errors="ignore"
                    )
                if expected_str not in artifact_text:
                    failures.append(
                        f"{row.claim_id}: expected substring `{expected_str}` not found in `{row.artifact}`"
                    )
                continue

            if not artifact_is_json:
                failures.append(
                    f"{row.claim_id}: key-path check `{key}` requires JSON artifact, got `{row.artifact}`"
                )
                continue

            try:
                observed = lookup_path(artifact_data, key)
            except KeyError:
                failures.append(
                    f"{row.claim_id}: missing key path `{key}` in `{row.artifact}`"
                )
                continue

            expected_float = try_float(expected_str)
            if expected_float is not None:
                if not isinstance(observed, (int, float)):
                    failures.append(
                        f"{row.claim_id}: `{key}` expected numeric {expected_float}, observed non-numeric `{observed}`"
                    )
                    continue
                if abs(float(observed) - expected_float) > 1e-4:
                    failures.append(
                        f"{row.claim_id}: `{key}` expected {expected_float}, observed {observed}"
                    )
                continue

            if str(observed) != expected_str:
                failures.append(
                    f"{row.claim_id}: `{key}` expected `{expected_str}`, observed `{observed}`"
                )

    return failures


def scan_docs_text() -> list[str]:
    failures: list[str] = []
    for file_path in markdown_files():
        rel = file_path.relative_to(ROOT)
        text = file_path.read_text(encoding="utf-8", errors="ignore")

        for pattern, message in DEPRECATED_PATTERNS.items():
            if re.search(pattern, text):
                failures.append(f"{rel}: {message} (pattern: {pattern})")

        for pattern, message in SECRET_PATTERNS.items():
            if re.search(pattern, text):
                failures.append(f"{rel}: {message} (pattern: {pattern})")

    return failures


def main() -> int:
    failures: list[str] = []
    if not CLAIMS_FILE.exists():
        print(f"Claims file not found: {CLAIMS_FILE}")
        return 1

    claims_text = CLAIMS_FILE.read_text(encoding="utf-8")
    claim_rows = parse_claim_rows(claims_text)
    if not claim_rows:
        print(f"No claim rows parsed from: {CLAIMS_FILE}")
        return 1

    failures.extend(validate_claim_rows(claim_rows))
    failures.extend(scan_docs_text())

    if failures:
        print("Docs facts check FAILED")
        for item in failures:
            print(f"- {item}")
        return 1

    print("Docs facts check PASSED")
    print(f"- Validated {len(claim_rows)} claims")
    print("- No deprecated model markers found")
    print("- No secret-like tokens detected in README/docs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
