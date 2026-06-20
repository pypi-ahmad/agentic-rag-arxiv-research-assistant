"""Retrieval primitives for Hybrid / Agentic / CRAG notebooks."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from src.ingest import embed_query

EMBED_MODEL_DIM_TO_NAME = {
    1024: "qwen3-embedding:0.6b",
    2560: "qwen3-embedding:4b",
}

STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "by",
    "as",
    "at",
    "from",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
}


def _tokenise(text: str) -> list[str]:
    tokens = re.sub(r"[^a-z0-9]", " ", text.lower()).split()
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


class DenseRetriever:
    """FAISS dense retrieval over existing chunk index."""

    def __init__(self, index, chunks: list[dict], embed_model: str | None = None) -> None:
        self.index = index
        self.chunks = chunks
        self.embed_model = embed_model or EMBED_MODEL_DIM_TO_NAME.get(index.d, "qwen3-embedding:0.6b")

    def retrieve(self, query: str, k: int = 10) -> list[dict]:
        qv = embed_query(query, model=self.embed_model)
        scores, indices = self.index.search(qv, k)
        out: list[dict] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = dict(self.chunks[idx])
            row["score"] = float(score)
            row["retriever"] = "dense"
            out.append(row)
        return out


class BM25Retriever:
    """Sparse lexical retrieval using BM25."""

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks
        self._tokens = [_tokenise(c.get("text", "")) for c in chunks]
        self._bm25 = BM25Okapi(self._tokens)

    def retrieve(self, query: str, k: int = 10) -> list[dict]:
        q_tokens = _tokenise(query)
        scores = self._bm25.get_scores(q_tokens)
        order = np.argsort(scores)[::-1][:k]
        out: list[dict] = []
        for idx in order:
            row = dict(self.chunks[int(idx)])
            row["score"] = float(scores[idx])
            row["retriever"] = "bm25"
            out.append(row)
        return out


class HybridRetriever:
    """Alpha fusion of dense and BM25 retrieval."""

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, alpha: float = 0.7) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.alpha = alpha

    @staticmethod
    def _minmax(values: list[float]) -> tuple[float, float]:
        if not values:
            return 0.0, 1.0
        return min(values), max(values)

    def retrieve(self, query: str, k: int = 10, fetch_k: int | None = None) -> list[dict]:
        fetch_k = fetch_k or max(k * 2, 10)
        d_res = self.dense.retrieve(query, k=fetch_k)
        b_res = self.bm25.retrieve(query, k=fetch_k)

        d_map = {r["chunk_id"]: r["score"] for r in d_res}
        b_map = {r["chunk_id"]: r["score"] for r in b_res}
        all_ids = set(d_map) | set(b_map)

        d_min, d_max = self._minmax(list(d_map.values()))
        b_min, b_max = self._minmax(list(b_map.values()))

        by_id = {r["chunk_id"]: r for r in d_res + b_res}

        def norm(score: float, lo: float, hi: float) -> float:
            return (score - lo) / max(hi - lo, 1e-9)

        fused: list[tuple[float, dict]] = []
        for cid in all_ids:
            base = dict(by_id[cid])
            d = norm(d_map.get(cid, d_min), d_min, d_max)
            b = norm(b_map.get(cid, b_min), b_min, b_max)
            s = self.alpha * d + (1.0 - self.alpha) * b
            base["score"] = float(s)
            base["retriever"] = "hybrid"
            fused.append((s, base))

        fused.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in fused[:k]]


@dataclass
class RetrievalBundle:
    dense: DenseRetriever
    bm25: BM25Retriever
    hybrid: HybridRetriever

