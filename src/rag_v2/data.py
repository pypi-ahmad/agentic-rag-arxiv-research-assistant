"""Data loading utilities for new notebooks (05-09)."""

from __future__ import annotations

import pickle
from pathlib import Path

import faiss

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_PATH = PROJECT_ROOT / "artifacts" / "faiss_index" / "index.bin"
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "artifacts" / "faiss_index" / "chunks.pkl"


def load_base_corpus(
    index_path: Path = DEFAULT_INDEX_PATH,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
) -> tuple[faiss.Index, list[dict]]:
    """Load the existing 4,000-paper FAISS corpus built by prior notebooks."""
    if not index_path.exists():
        raise FileNotFoundError(f"Missing FAISS index: {index_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunk metadata: {chunks_path}")

    index = faiss.read_index(str(index_path))
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    return index, chunks


def load_papers_from_chunks(chunks: list[dict]) -> list[dict]:
    """
    Reconstruct paper-level records from chunk metadata.

    The project corpus stores many chunks per paper. For evaluation and
    graph-style algorithms, we often need one merged paper text.
    """
    bucket: dict[str, dict] = {}
    for chunk in chunks:
        paper_id = chunk.get("paper_id", "")
        if not paper_id:
            continue
        if paper_id not in bucket:
            bucket[paper_id] = {
                "id": paper_id,
                "title": chunk.get("title", ""),
                "category": chunk.get("category", ""),
                "chunks": [],
            }
        bucket[paper_id]["chunks"].append(chunk)

    papers: list[dict] = []
    for paper in bucket.values():
        ordered = sorted(
            paper["chunks"],
            key=lambda x: int(x.get("chunk_index", x.get("chunk_idx", 0))),
        )
        merged_text = "\n".join(c.get("text", "") for c in ordered)
        papers.append(
            {
                "id": paper["id"],
                "title": paper["title"],
                "category": paper["category"],
                "text": merged_text,
                "n_chunks": len(ordered),
            }
        )

    papers.sort(key=lambda x: x["id"])
    return papers

