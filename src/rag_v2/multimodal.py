"""Multimodal helpers for notebook 09."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import ollama
import requests

ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def download_arxiv_pdf(arxiv_id: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    output_path.write_bytes(resp.content)
    return output_path


def pdf_first_page_to_png(pdf_path: Path, output_png: Path) -> Path:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    prefix = output_png.with_suffix("")
    cmd = [
        "pdftoppm",
        "-f",
        "1",
        "-singlefile",
        "-png",
        str(pdf_path),
        str(prefix),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    generated = output_png
    if not generated.exists():
        raise FileNotFoundError(f"Failed to render first page for {pdf_path}")
    return generated


def _clean_cli_output(text: str) -> str:
    cleaned = ANSI_RE.sub("", text)
    cleaned = cleaned.replace("\r", "\n")
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    # Drop spinner / divider noise from CLI output.
    lines = [ln for ln in lines if ln not in {"---", "```", "markdown"}]
    if not lines:
        return ""
    deduped: list[str] = []
    for line in lines:
        if deduped and line == deduped[-1]:
            continue
        deduped.append(line)
    return "\n".join(deduped[:120])


def run_glm_ocr_cli(image_path: Path) -> str:
    """
    OCR via CLI as requested by user: `ollama run glm-ocr`.
    """
    prompt = f"Extract all readable text from this image: {image_path}"
    cmd = ["ollama", "run", "glm-ocr", prompt]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    raw = proc.stdout.strip() or proc.stderr.strip()
    return _clean_cli_output(raw)


def run_qwen_vision(
    image_path: Path,
    question: str = "Summarize this first page and list key terms.",
    fallback_text: str | None = None,
) -> str:
    """
    Vision response from qwen3.5:4b.
    """
    if fallback_text and fallback_text.strip():
        prompt = (
            f"{question}\n\n"
            "Use this OCR text from the page and return a concise summary plus key terms.\n\n"
            f"OCR text:\n{fallback_text[:1200]}"
        )
        resp = ollama.chat(
            model="qwen3.5:4b",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_gpu": 0, "num_ctx": 4096, "num_predict": 100},
        )
        content = resp["message"]["content"].strip()
        if content:
            return content

    try:
        resp = ollama.chat(
            model="qwen3.5:4b",
            messages=[
                {
                    "role": "user",
                    "content": question,
                    "images": [str(image_path)],
                }
            ],
            options={"temperature": 0.0, "num_gpu": 0, "num_ctx": 8192, "num_predict": 80},
        )
        return resp["message"]["content"].strip()
    except Exception:
        return ""
