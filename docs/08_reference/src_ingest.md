# `src/ingest.py` — Data Loading and FAISS Indexing

Source: [`src/ingest.py`](https://github.com/pypi-ahmad/agentic-rag-arxiv-research-assistant/blob/main/src/ingest.py)

This module owns the build-time ingestion pipeline:

1. load papers
2. chunk documents
3. embed chunks with Ollama
4. build FAISS index
5. save/load index artifacts

It is used by notebook 01 for index creation and by downstream notebooks for index loading.

---

## Key constants

```python
EMBED_MODEL_PRIMARY = "qwen3-embedding:4b"
EMBED_MODEL_LITE = "qwen3-embedding:0.6b"
TARGET_CATEGORIES = ["cs.CL", "cs.AI", "cs.LG"]
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
```

`embed_texts()` and `embed_query()` default to `EMBED_MODEL_LITE` unless overridden.

---

## Data loading

### `load_arxiv_papers(n_samples, categories)`

Attempts live ArXiv retrieval first and falls back to HuggingFace when unavailable.

### `load_hf_papers(n_samples, split, ml_filter, scan_multiplier)`

Loads from `ccdv/arxiv-summarization`, scans a larger slice, and keeps ML/AI-relevant rows when `ml_filter=True`.

Returns a `list[dict]` with keys:

- `id`
- `title`
- `abstract`
- `category`
- `url`

---

## Chunking

### `chunk_documents(papers, chunk_size=512, chunk_overlap=64)`

Splits abstracts into overlapping character windows.

Returns chunk dicts including:

- `chunk_id`
- `paper_id`
- `title`
- `text`
- `chunk_idx`
- `category`

---

## Embedding

### `embed_texts(texts, model=EMBED_MODEL_LITE, batch_size=32)`

Embeds batched text using Ollama and returns L2-normalized `float32` matrix.

### `embed_query(query, model=EMBED_MODEL_LITE)`

Embeds one query and returns shape `(1, dim)` for FAISS search compatibility.

---

## FAISS index

### `build_faiss_index(embeddings)`

Builds `faiss.IndexFlatIP` over normalized embeddings.

### `save_index_and_chunks(index, chunks, base_path)`

Persists synchronized artifacts under `base_path`:

- `artifacts/faiss_index/index.bin`
- `artifacts/faiss_index/chunks.pkl`

In this project, notebook 01 saves to `artifacts/faiss_index/`.

### `load_index_and_chunks(base_path)`

Loads the same pair and raises `FileNotFoundError` if notebook 01 has not been executed.

---

## Artifact contract

Downstream notebooks assume the pair below exists and remains position-aligned:

- `artifacts/faiss_index/index.bin`
- `artifacts/faiss_index/chunks.pkl`

If these files are missing, run `notebooks/01_naive_rag.ipynb` first.
