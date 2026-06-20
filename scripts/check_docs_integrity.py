#!/usr/bin/env python3
"""Docs integrity checks for README + MkDocs markdown pages.

Checks:
1. Stale patterns that should not appear in docs.
2. Referenced repo paths in code spans/markdown links exist (with allowlist exceptions).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = ROOT / "docs"

STALE_PATTERNS: dict[str, str] = {
    r"_pending run_": "Placeholder metric marker found.",
    r"artifacts/faiss_index\.bin": "Deprecated FAISS path. Use artifacts/faiss_index/index.bin.",
    r"artifacts/chunks\.pkl": "Deprecated chunk path. Use artifacts/faiss_index/chunks.pkl.",
    r"artifacts/eval/": "Deprecated eval path. Use artifacts/eval_results/.",
    r"granite4\.1-guardian": "Deprecated model in docs. Use granite4.1:8b for judge path.",
}

# Runtime-generated files that may be absent in clean clones.
OPTIONAL_PATH_PREFIXES: tuple[str, ...] = (
    "artifacts/eval_results/04_chromadb_graphrag_4000.json",
    "artifacts/eval_results/04_pinecone_graphrag_4000.json",
    "artifacts/eval_results/04_agent_graphrag_4000.json",
)

CODE_PAT = re.compile(r"`([^`\n]+)`")
LINK_PAT = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
SAFE_PATH_TOKEN = re.compile(r"^[A-Za-z0-9_./~-]+$")
ROOT_PATH_PREFIXES: tuple[str, ...] = (
    "docs/",
    "notebooks/",
    "src/",
    "scripts/",
    "artifacts/",
)
ROOT_FILES: tuple[str, ...] = ("README.md", "mkdocs.yml", "requirements.txt")


def markdown_files() -> list[Path]:
    files = [ROOT / "README.md"]
    files.extend(sorted(DOC_ROOT.rglob("*.md")))
    return files


def is_path_like(token: str) -> bool:
    if token.startswith(("http://", "https://", "mailto:", "#")):
        return False
    if token.startswith(("../", "./")):
        return True
    if not SAFE_PATH_TOKEN.match(token):
        return False
    if token.startswith(ROOT_PATH_PREFIXES) or token in ROOT_FILES:
        return True
    if "/" in token:
        return False
    if any(ch.isalpha() for ch in token) is False:
        return False
    return token.endswith((
        ".md",
        ".ipynb",
        ".json",
        ".pkl",
        ".bin",
        ".png",
        ".py",
        ".yml",
        ".txt",
    ))


def normalize_token(token: str) -> str:
    token = token.strip().strip("<>")
    token = token.split("#", maxsplit=1)[0]
    token = token.split("?", maxsplit=1)[0]
    token = token.rstrip(".,;:")
    return token


def iter_path_tokens(text: str) -> Iterable[str]:
    for match in LINK_PAT.finditer(text):
        tok = normalize_token(match.group(1))
        if tok and is_path_like(tok) and "*" not in tok:
            yield tok
    for match in CODE_PAT.finditer(text):
        tok = normalize_token(match.group(1))
        if tok and is_path_like(tok) and "*" not in tok:
            # Skip obvious pseudo-tokens from formulas/snippets.
            if tok.startswith(("1/", "b×", "(1/")):
                continue
            yield tok


def resolve_path(current_file: Path, token: str) -> Path:
    if token.startswith(("../", "./")):
        return (current_file.parent / token).resolve()
    if token.startswith(ROOT_PATH_PREFIXES) or token in ROOT_FILES:
        return (ROOT / token).resolve()
    if token.endswith(".ipynb") and "/" not in token:
        return (ROOT / "notebooks" / token).resolve()
    if token.endswith(".md") and "/" not in token:
        return (current_file.parent / token).resolve()
    return (ROOT / token).resolve()


def path_is_optional_missing(token: str) -> bool:
    return any(token.startswith(prefix) for prefix in OPTIONAL_PATH_PREFIXES)


def main() -> int:
    files = markdown_files()
    failures: list[str] = []

    for file_path in files:
        rel = file_path.relative_to(ROOT)
        text = file_path.read_text(encoding="utf-8", errors="ignore")

        for pattern, message in STALE_PATTERNS.items():
            if re.search(pattern, text):
                failures.append(f"{rel}: {message} (pattern: {pattern})")

        for token in iter_path_tokens(text):
            resolved = resolve_path(file_path, token)
            if resolved.exists():
                continue
            if path_is_optional_missing(token):
                continue
            # Ignore package names or command-ish tokens that look like paths.
            if token.startswith(("cross-encoder/", "ccdv/", "~/.")):
                continue
            failures.append(f"{rel}: Missing referenced path `{token}`")

    if failures:
        print("Docs integrity check FAILED")
        for item in failures:
            print(f"- {item}")
        return 1

    print("Docs integrity check PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
