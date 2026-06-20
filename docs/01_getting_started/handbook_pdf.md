# Full Tutorial PDF

This repository ships a full handbook PDF for offline learning and recruiter review.

## Download

- [agentic-rag-full-tutorial.pdf](../assets/agentic-rag-full-tutorial.pdf)

## Rebuild Locally

```bash
uv run python scripts/build_tutorial_pdf.py
```

The script builds docs in strict mode, renders the print page, and writes:

- `docs/assets/agentic-rag-full-tutorial.pdf`

## Notes

- The PDF is generated from the MkDocs content, so markdown pages remain the single source of truth.
- If dependencies are missing, install with:

```bash
uv pip install -r requirements.txt
```

The renderer uses a local Chrome/Chromium binary in headless mode.
