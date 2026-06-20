"""Reusable utilities for notebooks 05-09 (new RAG techniques)."""

from .data import load_base_corpus, load_papers_from_chunks
from .retrieval import DenseRetriever, BM25Retriever, HybridRetriever
from .metrics import (
    build_keyword_eval_set,
    compute_retrieval_metrics,
    plot_retrieval_comparison,
)

__all__ = [
    "load_base_corpus",
    "load_papers_from_chunks",
    "DenseRetriever",
    "BM25Retriever",
    "HybridRetriever",
    "build_keyword_eval_set",
    "compute_retrieval_metrics",
    "plot_retrieval_comparison",
]
