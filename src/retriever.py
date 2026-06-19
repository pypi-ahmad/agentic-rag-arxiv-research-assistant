"""
src/retriever.py — Retrieval strategies (shared by notebooks 02 and 03)

Three retrievers, each adding a layer on top of the previous:

  1. DenseRetriever   — FAISS semantic search (built in notebook 01)
  2. BM25Retriever    — keyword-based sparse retrieval (introduced in notebook 02)
  3. HybridRetriever  — alpha-weighted fusion of dense + sparse scores
  4. Reranker         — cross-encoder re-scores a candidate set down to top-k

Why four classes instead of one?
    Each represents a distinct retrieval paradigm. By keeping them separate,
    notebook 02 can swap them in/out and measure the contribution of each step
    independently — which is how the results table is produced.
"""

from __future__ import annotations

import re
import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.ingest import embed_query, EMBED_MODEL_PRIMARY


# ── BM25 tokenisation helpers ─────────────────────────────────────────────────
#
# IMPROVEMENT 3 (see notebook 02 for full explanation and before/after metrics):
#
# OLD approach — whitespace split only:
#   tokens = text.lower().split()
# Problems:
#   • "Transformer" and "transformer" were different tokens (case mismatch)
#   • Stop words ("the", "is", "of") consumed scoring budget — they appear in
#     every document with high tf but near-zero IDF, contributing almost nothing
#   • Punctuation was kept: "attention." ≠ "attention"
#
# NEW approach — lowercase + strip punctuation + remove stop words:

BM25_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "as", "this", "that", "these",
    "those", "it", "its", "we", "our", "they", "their", "which", "who",
    "and", "or", "but", "not", "no", "nor", "so", "yet", "both", "each",
    "about", "than", "more", "also", "such", "into", "its", "over",
})


def _tokenise(text: str) -> list[str]:
    """Lowercase, strip non-alphanumeric chars, remove stop words and single-char tokens."""
    tokens = re.sub(r"[^a-z0-9]", " ", text.lower()).split()
    return [t for t in tokens if t not in BM25_STOP_WORDS and len(t) > 1]


# ── 1. Dense retriever (semantic / embedding-based) ───────────────────────────

class DenseRetriever:
    """
    Retrieves documents by cosine similarity between query and chunk embeddings.

    How it works:
        1. Embed the query with the same model used to build the FAISS index.
        2. Ask FAISS for the k nearest vectors (highest inner product = most similar).
        3. Map the returned integer IDs back to chunk dicts.

    When does dense retrieval win?
        Queries that use *different words* than the documents but mean the same
        thing. Example: query "how do LLMs generate text?" retrieves chunks that
        talk about "autoregressive decoding" even though neither exact phrase appears
        in the other. Embeddings capture *meaning*, not keywords.

    When does dense retrieval struggle?
        Exact-match searches for rare terms, code identifiers, acronyms, or
        proper nouns that weren't well represented in the embedding model's
        training data. This is where BM25 (keyword search) complements it.
    """

    def __init__(
        self,
        index,              # faiss.IndexFlatIP returned by build_faiss_index()
        chunks: list[dict], # parallel list: chunks[i] is the document at index row i
        embed_model: str = EMBED_MODEL_PRIMARY,
    ) -> None:
        self.index = index
        self.chunks = chunks
        self.embed_model = embed_model

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """
        Return the top-k most semantically similar chunks for a query.

        Args:
            query: User question or search string.
            k:     Number of results to return.

        Returns:
            List of chunk dicts (from src/ingest.py) sorted by similarity score,
            each augmented with a "score" key (cosine similarity, 0–1).
        """
        query_vec = embed_query(query, model=self.embed_model)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 when the index has fewer than k vectors
                continue
            chunk = dict(self.chunks[idx])   # copy so we don't mutate the original
            chunk["score"] = float(score)
            chunk["retriever"] = "dense"
            results.append(chunk)

        return results


# ── 2. Sparse retriever (BM25 keyword-based) ─────────────────────────────────

class BM25Retriever:
    """
    Retrieves documents using BM25 (Best Match 25) — a classical keyword-scoring
    algorithm that has been the gold standard in information retrieval since 1994.

    How BM25 scores a document D for query Q:
        For each query term t:
            score += IDF(t) * (tf(t,D) * (k1+1)) / (tf(t,D) + k1*(1 - b + b*|D|/avgdl))

        Where:
            IDF(t)  = log((N - df(t) + 0.5) / (df(t) + 0.5))  ← rare terms score higher
            tf(t,D) = term frequency of t in document D
            |D|     = document length (words)
            avgdl   = average document length across corpus
            k1=1.5, b=0.75 are tuning constants (BM25 "Okapi" variant)

    In plain English:
        A term scores highly if it appears often in the document (tf) but rarely
        in the corpus overall (idf). Long documents are penalised (length norm).

    When does BM25 win over dense retrieval?
        Searches for specific technical terms: model names ("Qwen3"), paper IDs,
        dataset names ("SQuAD"), or any rare tokens that the embedding model
        may have collapsed into a generic semantic region.
    """

    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks

        # OLD tokenisation (whitespace only — kept for reference):
        # tokenised_corpus = [chunk["text"].lower().split() for chunk in chunks]

        # IMPROVED tokenisation (lowercase + strip punctuation + stop words):
        # See _tokenise() above for the full explanation.
        tokenised_corpus = [_tokenise(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenised_corpus)
        logger.info(f"Built BM25 index over {len(chunks)} chunks.")

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """
        Return the top-k highest BM25-scoring chunks for a query.

        Args:
            query: User question or search string.
            k:     Number of results to return.

        Returns:
            List of chunk dicts sorted by BM25 score, each with a "score" key.
            Scores are raw BM25 values (not normalised to 0–1).
        """
        # OLD: query_tokens = query.lower().split()
        # IMPROVED: same tokeniser as the corpus — must match for consistent scoring
        query_tokens = _tokenise(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get indices of top-k scores
        top_k_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_k_indices:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            chunk["retriever"] = "bm25"
            results.append(chunk)

        return results


# ── 3. Hybrid retriever (dense + sparse fusion) ───────────────────────────────

class HybridRetriever:
    """
    Combines dense and BM25 scores using Reciprocal Rank Fusion (RRF) or
    a simple linear alpha blend — both are supported.

    Why hybrid search?
        Neither dense nor sparse retrieval dominates across all query types.
        Dense wins on paraphrase and semantic queries; BM25 wins on exact-match.
        Combining them is consistently better than either alone.

    Two fusion strategies:

    1. Alpha-weighted score fusion (default):
            hybrid_score = alpha * dense_score_norm + (1 - alpha) * bm25_score_norm
       Both scores are min-max normalised before blending so they're on the
       same 0–1 scale. alpha=0.7 means "70% semantic, 30% keyword".

    2. Reciprocal Rank Fusion (RRF):
            rrf_score = Σ_i  1 / (k + rank_i)
       Ignores raw scores entirely — only uses the rank position of each document
       in each retriever's sorted list. Robust to score scale differences.
    """

    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        alpha: float = 0.7,
        fusion: str = "alpha",  # "alpha" or "rrf"
        rrf_k: int = 60,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.alpha = alpha
        self.fusion = fusion
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int = 10) -> list[dict]:
        """
        Fetch candidates from both retrievers and fuse their scores.

        We fetch 2*k from each retriever so the merged candidate pool is
        large enough that the final top-k are truly the best results.

        Args:
            query: User question.
            k:     Final number of results after fusion.

        Returns:
            Top-k chunks sorted by fused score, each with a "score" key.
        """
        fetch_k = k * 2

        dense_results = self.dense.retrieve(query, k=fetch_k)
        bm25_results = self.bm25.retrieve(query, k=fetch_k)

        if self.fusion == "rrf":
            return self._rrf_fusion(dense_results, bm25_results, k)
        else:
            return self._alpha_fusion(dense_results, bm25_results, k)

    def _alpha_fusion(
        self,
        dense_results: list[dict],
        bm25_results: list[dict],
        k: int,
    ) -> list[dict]:
        """Min-max normalise both score sets, then blend with alpha."""
        # Build lookup: chunk_id → score for each retriever
        dense_map = {r["chunk_id"]: r["score"] for r in dense_results}
        bm25_map = {r["chunk_id"]: r["score"] for r in bm25_results}

        # Union of all candidate chunk IDs
        all_ids = set(dense_map) | set(bm25_map)

        # Min-max normalise dense scores
        d_vals = list(dense_map.values())
        d_min, d_max = (min(d_vals), max(d_vals)) if d_vals else (0, 1)

        # Min-max normalise BM25 scores
        b_vals = list(bm25_map.values())
        b_min, b_max = (min(b_vals), max(b_vals)) if b_vals else (0, 1)

        def norm_dense(s):
            return (s - d_min) / max(d_max - d_min, 1e-10)

        def norm_bm25(s):
            return (s - b_min) / max(b_max - b_min, 1e-10)

        # Merge all chunk metadata from both result sets
        all_chunks: dict[str, dict] = {}
        for r in dense_results + bm25_results:
            all_chunks[r["chunk_id"]] = r

        # Compute fused score for each candidate
        fused: list[tuple[float, dict]] = []
        for cid in all_ids:
            chunk = dict(all_chunks[cid])
            d_score = norm_dense(dense_map.get(cid, d_min))
            b_score = norm_bm25(bm25_map.get(cid, b_min))
            chunk["score"] = self.alpha * d_score + (1 - self.alpha) * b_score
            chunk["retriever"] = "hybrid"
            fused.append((chunk["score"], chunk))

        fused.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in fused[:k]]

    def _rrf_fusion(
        self,
        dense_results: list[dict],
        bm25_results: list[dict],
        k: int,
    ) -> list[dict]:
        """Reciprocal Rank Fusion — rank-based, ignores raw scores."""
        scores: dict[str, float] = {}
        all_chunks: dict[str, dict] = {}

        for results in [dense_results, bm25_results]:
            for rank, chunk in enumerate(results):
                cid = chunk["chunk_id"]
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                all_chunks[cid] = chunk

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        result = []
        for cid, score in ranked:
            chunk = dict(all_chunks[cid])
            chunk["score"] = score
            chunk["retriever"] = "hybrid-rrf"
            result.append(chunk)
        return result


# ── 4. Cross-encoder reranker ─────────────────────────────────────────────────

class Reranker:
    """
    Re-scores a candidate set using a cross-encoder (bi-directional attention).

    Why two-stage retrieval (retrieve then rerank)?

        Bi-encoder (what we use for retrieval):
            Encodes query and document INDEPENDENTLY → fast, scales to millions
            of docs, but misses query-document interactions.

            query → encoder → q_vec ─┐
                                      ├── cosine_sim → score
            doc   → encoder → d_vec ─┘

        Cross-encoder (what we use for reranking):
            Encodes query and document TOGETHER → attention flows both ways,
            captures fine-grained interactions, but is O(n) per query per doc
            so it's only practical on a small candidate set (10–50 docs).

            [query] [SEP] [document] → encoder → relevance score

        Strategy: bi-encoder retrieves fast from the full corpus; cross-encoder
        reranks the top-50 candidates to precision.

    Model used: cross-encoder/ms-marco-MiniLM-L-6-v2
        - Trained on MS MARCO (Microsoft Machine Reading Comprehension dataset)
        - 22 MB on disk, runs on CPU
        - Strong zero-shot performance on passage retrieval tasks
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        logger.info(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Re-score and re-rank a candidate list using the cross-encoder.

        Args:
            query:      The original user query.
            candidates: List of chunk dicts (output from any retriever).
            top_k:      Number of results to return after reranking.

        Returns:
            Top-k chunks re-sorted by cross-encoder relevance score,
            each with "score" updated to the cross-encoder logit and
            "retriever" set to "reranked".
        """
        if not candidates:
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        # Zip scores back to chunks and sort descending
        scored = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for score, chunk in scored[:top_k]:
            chunk = dict(chunk)
            chunk["score"] = float(score)
            chunk["retriever"] = "reranked"
            results.append(chunk)

        return results
