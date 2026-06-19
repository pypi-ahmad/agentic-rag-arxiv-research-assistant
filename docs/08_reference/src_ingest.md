# `src/ingest.py` — Data Loading & Indexing

Source: [`src/ingest.py`](https://github.com/pypi-ahmad/agentic-rag-arxiv-research-assistant/blob/main/src/ingest.py)

This module owns the entire **build-time pipeline**: loading papers, chunking text, embedding chunks, and building + persisting the FAISS index. It is called once in notebook 01; notebooks 02 and 03 load the saved artifacts from disk.

---

## Constants

```python
EMBED_MODEL_LITE    = "qwen3-embedding:0.6b"   # 1 024-dim, ~639 MB VRAM
EMBED_MODEL_PRIMARY = "qwen3-embedding:4b"      # 2 560-dim, ~2.5 GB VRAM
TARGET_CATEGORIES   = ["cs.CL", "cs.AI", "cs.LG"]
DEFAULT_INDEX_PATH  = Path("artifacts/faiss_index")
```

`EMBED_MODEL_PRIMARY` is the default used throughout. Switch to `EMBED_MODEL_LITE` to cut VRAM usage at the cost of retrieval quality (~15% MRR drop in our tests).

---

## `load_hf_papers(n_samples, ml_filter)`

```python
def load_hf_papers(n_samples: int = 500, ml_filter: bool = True) -> list[dict]:
```

Loads papers from the `ccdv/arxiv-summarization` HuggingFace dataset. This is the **primary data source** for all notebooks.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `n_samples` | `int` | `500` | How many papers to return after filtering |
| `ml_filter` | `bool` | `True` | If `True`, only keeps papers whose abstract contains ML keywords |

**Returns** — `list[dict]` where each dict has:

```python
{
    "id":       "2401.15884",          # arXiv paper ID
    "title":    "Corrective RAG ...",  # paper title
    "abstract": "We propose ...",      # full abstract text
    "category": "cs.CL",              # primary arXiv category
    "url":      "https://arxiv.org/abs/2401.15884"
}
```

**How it works:**

1. Streams `ccdv/arxiv-summarization` train split (avoids downloading all 203K papers at once)
2. Filters rows: abstract must contain at least one ML keyword (`transformer`, `attention`, `neural`, `bert`, `gpt`, `llm`, `language model`, `deep learning`, `reinforcement`, `fine-tun`, `embedding`)
3. Stops after collecting `n_samples` matching papers

!!! tip "Why streaming?"
    The full dataset is ~4 GB. Streaming lets us collect 2 000 papers in ~2 minutes without downloading the whole thing.

---

## `load_arxiv_papers(n_samples, categories)`

```python
def load_arxiv_papers(n_samples: int = 500, categories: list[str] | None = None) -> list[dict]:
```

**Legacy function** — tries the live arxiv.org API first, falls back to `load_hf_papers()` on failure. Prefer `load_hf_papers()` directly to avoid rate-limit delays.

---

## `chunk_documents(papers, chunk_size, chunk_overlap)`

```python
def chunk_documents(
    papers: list[dict],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[dict]:
```

Splits each paper's abstract into overlapping fixed-size chunks.

**Parameters**

| Name | Default | Description |
|------|---------|-------------|
| `chunk_size` | `512` | Characters per chunk |
| `chunk_overlap` | `64` | Characters shared between consecutive chunks |

**Returns** — `list[dict]` where each dict has:

```python
{
    "chunk_id": "2401.15884_chunk_0",  # unique identifier
    "paper_id": "2401.15884",          # parent paper
    "title":    "Corrective RAG ...",
    "text":     "We propose a ...",    # the chunk text
    "chunk_idx": 0,                    # position within paper
    "category": "cs.CL"
}
```

!!! note "Rule of thumb for chunk_size"
    512 characters ≈ 100 tokens ≈ 2–3 sentences. For abstracts of 800–2 000 characters this produces 2–4 chunks per paper. Larger chunks give more context but coarser retrieval; smaller chunks give finer retrieval but lose sentence context.

---

## `embed_texts(texts, model, batch_size)`

```python
def embed_texts(
    texts: list[str],
    model: str = EMBED_MODEL_PRIMARY,
    batch_size: int = 32,
) -> np.ndarray:
```

Calls `ollama.embed()` in batches and returns an **L2-normalised** float32 matrix.

**Returns** — `np.ndarray` of shape `(n_texts, embed_dim)`, L2-normalised (each row has unit norm).

!!! info "Why L2-normalise?"
    After normalisation, `dot(a, b) == cosine_similarity(a, b)`. This lets us use `faiss.IndexFlatIP` (inner product) as a cosine similarity index — no extra computation at query time.

---

## `embed_query(query, model)`

```python
def embed_query(query: str, model: str = EMBED_MODEL_PRIMARY) -> np.ndarray:
```

Embeds a single query string. Returns shape `(1, embed_dim)` — the extra dimension is required by `faiss.index.search()`.

---

## `build_faiss_index(embeddings)`

```python
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
```

Creates a `faiss.IndexFlatIP` index and adds all embeddings.

`IndexFlatIP` performs **exact** inner-product search — no approximation, no compression. For 2 000 papers × ~4 chunks = ~8 000 chunks, exact search completes in under 1 ms per query. Switch to `IndexIVFFlat` for corpora above ~100 K vectors.

---

## `save_index_and_chunks(index, chunks, base_path)`

```python
def save_index_and_chunks(
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    base_path: Path = DEFAULT_INDEX_PATH,
) -> None:
```

Saves two files that **must always be kept in sync**:

| File | Format | Contents |
|------|--------|----------|
| `index.bin` | FAISS binary | The vector index |
| `chunks.pkl` | Python pickle | The parallel list of chunk dicts |

The integer row `i` in the FAISS index corresponds to `chunks[i]`. Never save one without the other.

---

## `load_index_and_chunks(base_path)`

```python
def load_index_and_chunks(base_path: Path = DEFAULT_INDEX_PATH) -> tuple[faiss.IndexFlatIP, list[dict]]:
```

Loads and returns `(index, chunks)`. Raises `FileNotFoundError` with a helpful message if notebook 01 has not been run yet.
