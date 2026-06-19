# `src/retriever.py` — Retrieval Strategies

Source: [`src/retriever.py`](https://github.com/pypi-ahmad/agentic-rag-arxiv-research-assistant/blob/main/src/retriever.py)

Four classes, each representing a distinct retrieval paradigm. Notebook 02 instantiates them independently and measures each one's contribution to retrieval quality.

---

## Module-level helpers

### `BM25_STOP_WORDS`

```python
BM25_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", ...
})
```

30+ common English stop words removed during BM25 tokenisation. These words have near-zero IDF (they appear in almost every document) so removing them speeds up indexing and slightly improves BM25 precision.

### `_tokenise(text)`

```python
def _tokenise(text: str) -> list[str]:
```

The improved tokeniser used by `BM25Retriever`. Three steps:

1. `text.lower()` — case-normalise
2. `re.sub(r"[^a-z0-9]", " ", ...)` — strip all punctuation
3. Filter: remove stop words and single-character tokens

**Before (old):** `"Transformer-based model."` → `["Transformer-based", "model."]`  
**After (new):** `"Transformer-based model."` → `["transformer", "based", "model"]`

---

## `DenseRetriever`

```python
class DenseRetriever:
    def __init__(self, index, chunks, embed_model=EMBED_MODEL_PRIMARY): ...
    def retrieve(self, query: str, k: int = 5) -> list[dict]: ...
```

Retrieves by cosine similarity between query and chunk embeddings. Wraps the FAISS index built in notebook 01.

**`retrieve()` returns** — top-k chunk dicts, each augmented with:

```python
{"score": 0.847, "retriever": "dense", ...}  # score = cosine similarity
```

**When dense wins:** paraphrase queries, semantic questions, queries with no exact vocabulary match to the corpus.

**When dense struggles:** rare technical terms, model names (`"Qwen3"`), dataset names (`"SQuAD"`), acronyms.

---

## `BM25Retriever`

```python
class BM25Retriever:
    def __init__(self, chunks: list[dict]): ...
    def retrieve(self, query: str, k: int = 5) -> list[dict]: ...
```

Retrieves by BM25 keyword scoring. Builds the index at construction time from the full chunk list.

**Construction:** tokenises every chunk with `_tokenise()` and passes the result to `BM25Okapi`.

**`retrieve()` returns** — top-k chunk dicts sorted by raw BM25 score:

```python
{"score": 4.21, "retriever": "bm25", ...}  # raw BM25 score, NOT normalised to 0-1
```

!!! warning "Score scale"
    BM25 scores are unbounded positive floats (typically 0–15 for our corpus). They cannot be directly compared to cosine similarity scores (0–1) without normalisation — this is why alpha fusion requires min-max normalisation.

**Best result in our eval:** MRR **0.441** (highest of all strategies on the 20-query eval set).

---

## `HybridRetriever`

```python
class HybridRetriever:
    def __init__(self, dense, bm25, alpha=0.7, fusion="alpha", rrf_k=60): ...
    def retrieve(self, query: str, k: int = 10) -> list[dict]: ...
```

Combines `DenseRetriever` and `BM25Retriever` results via one of two fusion strategies.

**Parameters**

| Name | Default | Description |
|------|---------|-------------|
| `alpha` | `0.7` | Weight for dense scores in alpha fusion (1-alpha for BM25) |
| `fusion` | `"alpha"` | `"alpha"` or `"rrf"` |
| `rrf_k` | `60` | RRF smoothing constant (standard value from literature) |

**Internal strategy — `_alpha_fusion()`:**

1. Collect 2k candidates from each retriever
2. Min-max normalise both score sets to [0, 1]
3. Blend: `hybrid = alpha × dense_norm + (1-alpha) × bm25_norm`
4. Sort by hybrid score, return top-k

**Internal strategy — `_rrf_fusion()`:**

1. Collect 2k candidates from each retriever
2. For each document: `rrf_score += 1 / (60 + rank)`
3. Sort by rrf_score, return top-k

**Project finding:** alpha fusion (MRR 0.308) slightly outperformed RRF (MRR 0.287) on this corpus because the dense score magnitude was informative — high-confidence matches had large cosine gaps that RRF discarded.

---

## `Reranker`

```python
class Reranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"): ...
    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]: ...
```

Re-scores a candidate list using a cross-encoder. Intended as a second stage after `HybridRetriever.retrieve(k=20)`.

**`rerank()` steps:**

1. Build `(query, chunk_text)` pairs
2. Call `CrossEncoder.predict(pairs)` — batched, CPU-only
3. Sort descending by cross-encoder logit score
4. Return top-k with `"score"` overwritten and `"retriever"` set to `"reranked"`

**Latency:** ~8 ms per (query, chunk) pair on CPU. For 20 candidates: ~160 ms total — acceptable for non-interactive use.

**Impact in this project:** RRF MRR 0.287 → RRF+Rerank MRR 0.363 (+27%).
