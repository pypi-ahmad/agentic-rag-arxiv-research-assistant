"""
src/ingest.py — Document ingestion pipeline (shared by all three notebooks)

What this file does:
  1. Downloads ArXiv ML/AI/NLP papers (via arxiv Python library or HuggingFace)
  2. Splits long paper abstracts into overlapping text chunks
  3. Embeds chunks using qwen3-embedding via Ollama
  4. Builds a FAISS index over the embeddings
  5. Saves/loads the index and chunk metadata to disk

Why a separate file instead of notebook code?
  All three notebooks (naive RAG, advanced RAG, agentic RAG) need the same
  indexed corpus. Keeping the logic here means we build the index once in
  notebook 01 and load it in notebooks 02 and 03 — no duplication.

Two data source options
  Option A — arxiv library (default):
    Fetches papers directly from arxiv.org using the official arxiv Python library.
    Papers are filtered to cs.CL / cs.AI / cs.LG so the corpus is topically
    coherent for ML/NLP/AI queries.
    Requires internet access; rate-limited to ~100 papers/request.

  Option B — HuggingFace ccdv/arxiv-summarization (fallback):
    200K papers from all of arxiv — not ML-filtered.
    Use load_hf_papers() if you need offline access or a larger corpus.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import arxiv as arxiv_lib
import faiss
import numpy as np
from loguru import logger
from tqdm import tqdm

import ollama


# Timeout-enabled Ollama client to prevent indefinite hangs during long runs.
# 300s is intentionally generous for larger batch requests.
OLLAMA_CLIENT = ollama.Client(timeout=300.0)


# ── Constants (override per notebook if needed) ───────────────────────────────

# Ollama embedding model names.
# LITE is faster (~60s for 500 docs); PRIMARY produces richer 2560-dim vectors.
# Notebooks 01-02 default to LITE; swap to PRIMARY for the best quality run.
EMBED_MODEL_PRIMARY = "qwen3-embedding:4b"
EMBED_MODEL_LITE = "qwen3-embedding:0.6b"

# ArXiv categories for the ML/AI/NLP corpus:
#   cs.CL = Computation and Language (NLP, LLMs, text generation)
#   cs.AI = Artificial Intelligence
#   cs.LG = Machine Learning
TARGET_CATEGORIES = ["cs.CL", "cs.AI", "cs.LG"]

# Chunk settings (characters, not tokens — roughly 1 token per 4 chars)
DEFAULT_CHUNK_SIZE = 512      # target characters per chunk
DEFAULT_CHUNK_OVERLAP = 64    # characters shared between consecutive chunks

# FAISS index stores L2-normalised vectors => inner product == cosine similarity
FAISS_METRIC = faiss.METRIC_INNER_PRODUCT


# ── Step 1: Load dataset ──────────────────────────────────────────────────────

def load_arxiv_papers(
    n_samples: int = 500,
    categories: list[str] | None = None,
) -> list[dict]:
    """
    Load ML/AI/NLP papers — tries arxiv.org first, falls back to HuggingFace.

    Primary path (arxiv.org):
        Fetches recent papers filtered by category (cs.CL, cs.AI, cs.LG).
        Requires internet access. Rate-limited to ~100 papers per request.

    Fallback path (HuggingFace ccdv/arxiv-summarization):
        Used automatically when arxiv.org is unreachable (offline, firewall).
        Scans the HuggingFace dataset and keeps abstracts containing ML keywords.

    Args:
        n_samples:  Number of papers to load (default 500).
        categories: ArXiv categories. Defaults to cs.CL, cs.AI, cs.LG.

    Returns:
        List of dicts: id, title, abstract, category, url.
    """
    if categories is None:
        categories = TARGET_CATEGORIES

    # Try arxiv.org (needs internet)
    try:
        import socket
        socket.setdefaulttimeout(5)
        socket.getaddrinfo("export.arxiv.org", 443)
        socket.setdefaulttimeout(None)

        category_query = " OR ".join(f"cat:{cat}" for cat in categories)
        logger.info(
            f"Fetching {n_samples} papers from arxiv.org ({', '.join(categories)}) ..."
        )
        client = arxiv_lib.Client(page_size=100, delay_seconds=3.0, num_retries=3)
        search = arxiv_lib.Search(
            query=category_query,
            max_results=n_samples,
            sort_by=arxiv_lib.SortCriterion.SubmittedDate,
            sort_order=arxiv_lib.SortOrder.Descending,
        )
        papers: list[dict] = []
        for result in tqdm(client.results(search), desc="Fetching from arxiv", total=n_samples):
            abstract = (result.summary or "").strip()
            if not abstract or len(abstract) < 80:
                continue
            papers.append(
                {
                    "id": result.entry_id.split("/")[-1],
                    "title": result.title.strip(),
                    "abstract": abstract,
                    "category": result.primary_category,
                    "url": result.entry_id,
                }
            )
            if len(papers) >= n_samples:
                break
        if papers:
            logger.info(f"Fetched {len(papers)} papers from arxiv.org.")
            return papers

    except (OSError, Exception) as e:
        logger.warning(f"arxiv.org unreachable ({e}), using HuggingFace fallback.")

    # Fallback: HuggingFace with ML keyword filter
    return load_hf_papers(n_samples=n_samples, ml_filter=True)


def load_hf_papers(
    n_samples: int = 500,
    split: str = "train",
    ml_filter: bool = True,
    scan_multiplier: int = 10,
) -> list[dict]:
    """
    Load from HuggingFace ccdv/arxiv-summarization — offline-friendly.

    The dataset contains 203K papers from ALL of arxiv. Because many papers
    are from physics, math, and chemistry, we scan a larger pool and keep only
    papers whose abstract contains ML/AI keywords. This gives a topically
    coherent corpus without requiring an internet connection to arxiv.org.

    Args:
        n_samples:        How many ML papers to return.
        split:            HuggingFace split — "train", "validation", or "test".
        ml_filter:        If True (default), keep only ML/AI-related abstracts.
                          Set to False for unrestricted sampling.
        scan_multiplier:  Load n_samples * scan_multiplier rows to have enough
                          candidates after filtering. Default 10 means scan 5000
                          rows to find 500 ML papers (~10-15% hit rate typical).

    Returns:
        List of dicts with: id, title, abstract, category, url.

    Note:
        Papers are sampled starting from offset 5000 (skipping the earliest
        papers which are heavily skewed toward math/stats) to get a broader
        cross-section of topics.
    """
    from datasets import load_dataset

    # ML/AI keyword filter — an abstract matching any of these is kept
    ML_KEYWORDS = [
        "neural network", "deep learning", "machine learning", "transformer",
        "attention mechanism", "language model", "reinforcement learning",
        "classification", "object detection", "generative model", "diffusion",
        "fine-tuning", "pre-training", "bert", "gpt", "llm", "embedding",
        "convolutional", "gradient descent", "backpropagation", "training",
        "inference", "benchmark", "dataset", "model architecture", "encoder",
        "decoder", "semantic", "representation learning", "self-supervised",
    ]

    scan_size = n_samples * scan_multiplier
    offset = 5000  # skip earliest rows (heavy math/stats bias)
    logger.info(
        f"Scanning {scan_size} rows from ccdv/arxiv-summarization ({split}) "
        f"to find {n_samples} ML/AI papers ..."
    )
    raw = load_dataset(
        "ccdv/arxiv-summarization",
        split=f"{split}[{offset}:{offset + scan_size}]",
    )

    papers: list[dict] = []
    for idx, row in enumerate(raw):
        abstract = (row.get("abstract") or "").strip()
        if not abstract or len(abstract) < 80:
            continue

        if ml_filter:
            abstract_lower = abstract.lower()
            if not any(kw in abstract_lower for kw in ML_KEYWORDS):
                continue

        first_sentence = abstract.split(".")[0].strip()
        title = first_sentence[:120] + ("..." if len(first_sentence) > 120 else "")
        paper_id = f"arxiv_{split}_{offset + idx:05d}"

        papers.append(
            {
                "id": paper_id,
                "title": title,
                "abstract": abstract,
                "category": "cs.AI",
                "url": f"https://arxiv.org/abs/{paper_id}",
            }
        )

        if len(papers) >= n_samples:
            break

    logger.info(f"Loaded {len(papers)} ML/AI papers (scanned up to {scan_size} rows).")
    return papers


# Alias for backward compatibility and notebook usage
load_arxiv_papers_hf = load_hf_papers


# ── Step 2: Chunk documents ───────────────────────────────────────────────────

def chunk_documents(
    papers: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
    """
    Split each paper's abstract into overlapping fixed-size character chunks.

    Why chunk at all?
        Embedding models have a maximum input length. More importantly, for
        long documents (full papers, reports, contracts) chunking ensures that
        each chunk is topically focused — you retrieve the specific paragraph
        that answers the question, not an entire 20-page paper.

        For abstracts specifically (150-400 words), each abstract produces
        1-3 chunks at chunk_size=512. This is intentionally coarse — we keep
        full semantic units intact rather than splitting mid-sentence.

    Why overlap?
        If a sentence spans two chunk boundaries, the overlap ensures neither
        chunk loses that sentence's context. Without overlap, relevant content
        that falls exactly on a boundary would be split and potentially missed.

    Args:
        papers:        Output of load_arxiv_papers() or load_hf_papers().
        chunk_size:    Target character count per chunk (~128 tokens).
        chunk_overlap: Characters shared between consecutive chunks.
                       Must be smaller than chunk_size.

    Returns:
        List of chunk dicts, each with:
            "chunk_id"    — unique key "<paper_id>_chunk_<n>"
            "paper_id"    — source paper ID (links back to papers list)
            "title"       — paper title (used in answer attribution)
            "category"    — ArXiv category label
            "text"        — chunk text (this string is passed to embed_texts)
            "chunk_index" — position of this chunk within its paper (0-indexed)

    Example:
        >>> chunks = chunk_documents(papers, chunk_size=256, chunk_overlap=32)
        >>> print(chunks[0]["chunk_id"])
        2310.01542v2_chunk_0
    """
    chunks: list[dict] = []
    for paper in tqdm(papers, desc="Chunking papers"):
        text = paper["abstract"].strip()
        if not text:
            continue

        step = chunk_size - chunk_overlap
        start = 0
        chunk_idx = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if len(chunk_text) > 50:  # skip tiny trailing fragments
                chunks.append(
                    {
                        "chunk_id": f"{paper['id']}_chunk_{chunk_idx}",
                        "paper_id": paper["id"],
                        "title": paper["title"],
                        "category": paper["category"],
                        "text": chunk_text,
                        "chunk_index": chunk_idx,
                    }
                )
                chunk_idx += 1
            start += step

    logger.info(f"Created {len(chunks)} chunks from {len(papers)} papers.")
    return chunks


# ── Step 3: Embed chunks ──────────────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    model: str = EMBED_MODEL_LITE,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Generate dense vector embeddings for a list of text strings via Ollama.

    What is an embedding?
        An embedding is a fixed-size numerical vector that represents the
        *meaning* of a piece of text. Similar texts produce vectors that point
        in similar directions in high-dimensional space. This enables semantic
        search: we compare query vectors to document vectors by angle
        (cosine similarity), not by keyword overlap.

    Why Ollama?
        qwen3-embedding runs entirely locally — no API keys, no network
        latency after the first model pull, and no data leaves your machine.

    Why L2 normalisation?
        After normalising each vector to unit length, the dot product (inner
        product) between two vectors equals their cosine similarity. This lets
        us use FAISS IndexFlatIP — which computes inner products — as a
        cosine-similarity index without any extra formula.

    Args:
        texts:      List of strings to embed.
        model:      Ollama model name (default: qwen3-embedding:0.6b — fast).
                    Use EMBED_MODEL_PRIMARY ("qwen3-embedding:4b") for higher quality.
        batch_size: Texts per Ollama API call. Larger = faster but more RAM.

    Returns:
        Float32 numpy array of shape (len(texts), embedding_dim).
        Vectors are L2-normalised so inner product == cosine similarity.
    """
    all_embeddings: list[list[float]] = []

    for i in tqdm(range(0, len(texts), batch_size), desc=f"Embedding ({model})"):
        batch = texts[i : i + batch_size]
        response = None
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                response = OLLAMA_CLIENT.embed(model=model, input=batch)
                break
            except Exception as exc:
                logger.warning(
                    f"Ollama embed failed (attempt {attempt}/{max_retries}) for batch "
                    f"{i//batch_size + 1}: {exc}"
                )
                if attempt == max_retries:
                    raise
                time.sleep(1.5 * attempt)

        all_embeddings.extend(response["embeddings"])

    embeddings = np.array(all_embeddings, dtype=np.float32)

    # L2-normalise: divide each vector by its own Euclidean magnitude.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-10)

    logger.info(f"Embedded {len(texts)} texts => shape {embeddings.shape}")
    return embeddings


# ── Step 4: Build FAISS index ─────────────────────────────────────────────────

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Create a FAISS flat inner-product index from an embedding matrix.

    What is FAISS?
        FAISS (Facebook AI Similarity Search) is a library for efficient
        nearest-neighbour search in high-dimensional vector spaces. "Flat"
        means it stores all vectors and performs exact (brute-force) search
        — no approximation. Exact search is fine for corpora up to ~100K vectors.
        For millions of vectors you would switch to an approximate index
        (e.g. IndexIVFFlat or IndexHNSW).

    Why IndexFlatIP (inner product) instead of IndexFlatL2 (L2 distance)?
        After L2 normalisation (done in embed_texts), inner product equals
        cosine similarity. Cosine similarity is the standard metric for
        semantic search because it measures the *angle* between two vectors,
        ignoring differences in vector magnitude caused by text length.

    Args:
        embeddings: L2-normalised float32 array of shape (n, dim).

    Returns:
        A FAISS IndexFlatIP with all n vectors already added.
        Query: index.search(query_vec, k) returns (scores, indices).
    """
    n, dim = embeddings.shape
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    logger.info(f"Built FAISS index: {n} vectors, dimension {dim}")
    return index


# ── Step 5: Save / load index and chunks ──────────────────────────────────────

def save_index_and_chunks(
    index: faiss.IndexFlatIP,
    chunks: list[dict],
    base_path: Path,
) -> None:
    """
    Persist the FAISS index and chunk metadata to disk.

    Two files are saved together:
        <base_path>/index.bin  — the FAISS binary index file
        <base_path>/chunks.pkl — the list of chunk dicts (metadata + text)

    The index maps integer positions (0, 1, 2, ...) to vectors.
    The chunks list maps those same integer positions to text and metadata.
    They must stay in sync — never save one without the other.

    Args:
        index:     FAISS index returned by build_faiss_index().
        chunks:    List of chunk dicts returned by chunk_documents().
        base_path: Directory where index.bin and chunks.pkl are saved.
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    index_path = base_path / "index.bin"
    chunks_path = base_path / "chunks.pkl"

    faiss.write_index(index, str(index_path))
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    logger.info(f"Saved FAISS index ({index.ntotal} vectors) => {index_path}")
    logger.info(f"Saved {len(chunks)} chunks => {chunks_path}")


def load_index_and_chunks(
    base_path: Path,
) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """
    Load a previously saved FAISS index and chunk list from disk.

    Used by notebooks 02 and 03 to avoid re-embedding the whole corpus
    (which takes ~60 seconds). Notebook 01 builds and saves; 02 and 03 load.

    Args:
        base_path: Directory containing index.bin and chunks.pkl.

    Returns:
        Tuple of (faiss_index, chunks_list).

    Raises:
        FileNotFoundError: If index files do not exist.
                           Run notebook 01_naive_rag.ipynb first.
    """
    base_path = Path(base_path)
    index_path = base_path / "index.bin"
    chunks_path = base_path / "chunks.pkl"

    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run notebook 01_naive_rag.ipynb first to build and save the index."
        )

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    logger.info(f"Loaded FAISS index ({index.ntotal} vectors) from {index_path}")
    logger.info(f"Loaded {len(chunks)} chunks from {chunks_path}")
    return index, chunks


# ── Step 6: Embed a single query (used at retrieval time) ─────────────────────

def embed_query(query: str, model: str = EMBED_MODEL_LITE) -> np.ndarray:
    """
    Embed a single query string and return a normalised 1D numpy vector.

    Separated from embed_texts() because:
      - Queries are embedded one at a time at retrieval time (not batched)
      - The FAISS search call expects shape (1, dim) not (dim,)

    IMPORTANT: Use the same model that was used to build the FAISS index.
    Mixing models produces garbage results because different models encode
    text into incompatible vector spaces.

    Args:
        query: The user's question or search string.
        model: Ollama model name — must match the model used in embed_texts().

    Returns:
        Float32 array of shape (1, embedding_dim), L2-normalised.
    """
    response = None
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = OLLAMA_CLIENT.embed(model=model, input=[query])
            break
        except Exception as exc:
            logger.warning(
                f"Ollama query embedding failed (attempt {attempt}/{max_retries}): {exc}"
            )
            if attempt == max_retries:
                raise
            time.sleep(1.0 * attempt)
    vec = np.array(response["embeddings"][0], dtype=np.float32)
    vec = vec / np.maximum(np.linalg.norm(vec), 1e-10)
    return vec.reshape(1, -1)
