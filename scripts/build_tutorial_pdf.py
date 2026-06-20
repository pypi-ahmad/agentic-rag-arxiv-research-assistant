#!/usr/bin/env python3
"""Build the full tutorial PDF from MkDocs print output."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = ROOT / "site"
DEFAULT_OUTPUT = ROOT / "docs" / "assets" / "agentic-rag-full-tutorial.pdf"


def run_mkdocs_build() -> None:
    cmd = [sys.executable, "-m", "mkdocs", "build", "--strict"]
    subprocess.run(cmd, cwd=ROOT, check=True)


def find_print_page() -> Path:
    candidates = [
        SITE_DIR / "print_page" / "index.html",
        SITE_DIR / "print_page.html",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    for pattern in ("**/print_page/index.html", "**/print_page.html", "**/*print*page*.html"):
        matches = sorted(SITE_DIR.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "Could not find MkDocs print page. Ensure `print-site` plugin is enabled in mkdocs.yml."
    )


def render_pdf(print_html: Path, output_pdf: Path) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    chrome_binary = next(
        (
            path
            for path in (
                shutil.which("google-chrome"),
                shutil.which("google-chrome-stable"),
                shutil.which("chromium"),
                shutil.which("chromium-browser"),
            )
            if path
        ),
        None,
    )
    if chrome_binary is None:
        raise RuntimeError(
            "No Chrome/Chromium binary found. Install google-chrome or chromium."
        )

    base_cmd = [
        chrome_binary,
        "--disable-gpu",
        "--no-sandbox",
        "--print-to-pdf-no-header",
        f"--print-to-pdf={output_pdf}",
        "--virtual-time-budget=20000",
        print_html.resolve().as_uri(),
    ]

    # Prefer the new headless mode, then fall back for older Chrome builds.
    errors: list[str] = []
    for headless_flag in ("--headless=new", "--headless"):
        cmd = [base_cmd[0], headless_flag, *base_cmd[1:]]
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if result.returncode == 0 and output_pdf.exists() and output_pdf.stat().st_size > 0:
            return
        stderr_tail = result.stderr.strip().splitlines()[-1] if result.stderr else "no stderr"
        errors.append(f"{headless_flag}: {stderr_tail}")

    raise RuntimeError(
        "Chrome PDF rendering failed. "
        + " | ".join(errors)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a full tutorial PDF from MkDocs content."
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip `mkdocs build --strict` and only render from existing site/ output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PDF path (default: {DEFAULT_OUTPUT.relative_to(ROOT)})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_pdf = args.output
    if not output_pdf.is_absolute():
        output_pdf = (ROOT / output_pdf).resolve()

    if not args.skip_build:
        run_mkdocs_build()

    print_html = find_print_page()
    render_pdf(print_html=print_html, output_pdf=output_pdf)

    print("Tutorial PDF build PASSED")
    print(f"- Source print page: {print_html.relative_to(ROOT)}")
    print(f"- Output: {output_pdf.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
