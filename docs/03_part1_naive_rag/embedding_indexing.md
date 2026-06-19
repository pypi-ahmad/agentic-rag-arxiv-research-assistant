# Embedding and Indexing

After chunking, each chunk is a plain string. To make it searchable by meaning rather than
by keyword, we convert it to a dense vector — an embedding. This page covers the three
functions that turn a list of strings into a queryable FAISS index, and how the artifacts
are persisted so that notebooks 02 and 03 do not have to re-embed the corpus.

---

## `embed_texts()` — batched calls to Ollama

```python
def embed_texts(
    texts: list[str],
    model: str = EMBED_MODEL_LITE,   # "qwen3-embedding:0.6b"
    batch_size: int = 32,
) -> np.ndarray:
```

The function iterates over the chunk list in batches of 32, calls `ollama.embed()` for each
batch, collects the results, and returns a single numpy array:

```python
for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding ({model})"):
    batch = texts[i : i + batch_size]
    response = ollama.embed(model=model, input=batch)
    all_embeddings.extend(response["embeddings"])

embeddings = np.array(all_embeddings, dtype=np.float32)
```

**Why batch size 32?** A single `ollama.embed()` call has a fixed overhead for loading the
model into the inference engine. Processing 32 texts per call amortises that overhead without
asking the Ollama server to hold hundreds of inputs in memory at once. On an 8GB GPU
(RTX 4060), batch size 32 keeps VRAM usage comfortably below the limit for `qwen3-embedding:0.6b`.

**Why float32?** FAISS expects float32 arrays. Using float16 would halve memory but requires
an explicit cast before every FAISS call. Float32 keeps the code simple.

---

## L2 normalisation — why it matters

Immediately after collecting the raw embeddings, `embed_texts()` normalises every vector:

```python
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
embeddings = embeddings / np.maximum(norms, 1e-10)
```

This divides each vector by its own Euclidean magnitude, producing unit-length vectors.
The `np.maximum(..., 1e-10)` guard prevents division-by-zero on a zero vector (which should
never occur for real text, but defensive code is good practice).

After normalisation, the dot product (inner product) of two vectors equals their cosine
similarity:

```
dot(a, b) = |a| × |b| × cos(θ)
          = 1  × 1  × cos(θ)    # after L2 normalisation
          = cos(θ)
```

Cosine similarity measures the angle between two vectors, ignoring their magnitude. This is
what we want for semantic search: two chunks that discuss the same concept should have a high
cosine score regardless of whether one is 100 characters and the other is 400.

!!! info "Definition — Cosine similarity"
    Given two unit vectors, cosine similarity ranges from -1 (opposite directions) to +1
    (identical direction). For text embeddings, values above 0.7 typically indicate strong
    semantic overlap; values near 0 indicate unrelated content.

---

## `build_faiss_index()` — creating `IndexFlatIP`

```python
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    n, dim = embeddings.shape
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"Built FAISS index: {n} vectors, dimension {dim}")
    return index
```

`IndexFlatIP` is FAISS's exact inner-product index. "Flat" means all vectors are stored
sequentially in memory and a query triggers a brute-force scan over every stored vector.
For 700–900 vectors at 1024 dimensions, this scan completes in under 1 millisecond — exact
search is entirely practical at this scale.

The alternative — an approximate index like `IndexIVFFlat` or `IndexHNSW` — would be faster
at scale (millions of vectors) but adds a training step, introduces recall loss, and is harder
to reason about. We use the flat index because correctness and simplicity matter more here
than sub-millisecond latency.

!!! tip "When to switch to an approximate index"
    The rule of thumb: switch from `IndexFlatIP` to `IndexIVFFlat` or `IndexHNSW` when
    your corpus exceeds ~100,000 vectors and query latency becomes noticeable. For a
    600-paper corpus producing under 1,000 chunks, exact search is the right choice.

---

## `save_index_and_chunks()` — persisting to disk

```python
def save_index_and_chunks(
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    base_path: Path,
) -> None:
    faiss.write_index(index, str(index_path))
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
```

Two files are written together:

| File | What it contains |
|---|---|
| `artifacts/faiss_index/index.bin` | The FAISS binary — vectors only, no text |
| `artifacts/faiss_index/chunks.pkl` | The chunk dicts — text, metadata, IDs |

The FAISS index maps integer positions (0, 1, 2, …) to vectors. The chunks list maps those
same integer positions to chunk text and metadata. They are always saved and loaded together.
Saving one without the other would produce a retriever that returns vectors with no associated
text — it would crash on the first query.

Notebooks 02 and 03 call `load_index_and_chunks()` to skip the 60-second embedding step:

```python
index, chunks = load_index_and_chunks(base_path=ARTIFACTS_DIR / "faiss_index")
```

This is the payoff for keeping all ingestion logic in `src/ingest.py`: the index is built
exactly once and shared across the entire tutorial series.

---

## What's next

With the index on disk, we are ready to query it. The next page — [Retrieval and Generation](retrieval_generation.md) — covers the `DenseRetriever` class, how a user question becomes
a FAISS search call, and how the retrieved chunks are assembled into a prompt for `granite4.1:8b`.
