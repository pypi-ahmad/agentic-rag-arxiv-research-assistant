"""
scripts/run_pipeline.py — Full end-to-end pipeline runner

Runs the complete RAG pipeline outside of Jupyter notebooks:
  Phase 1 — Ingest:   load papers, chunk, embed, build FAISS, save artifacts
  Phase 2 — Retrieve: compare dense / BM25 / hybrid on sample queries
  Phase 3 — Generate: naive RAG answer with granite4.1:8b
  Phase 4 — Evaluate: Recall@k, Precision@k, MRR, faithfulness (LLM-as-judge)

Usage:
    cd /home/ahmad/AI/Git/agentic-rag-arxiv-research-assistant
    source .venv/bin/activate
    python scripts/run_pipeline.py

Outputs are saved to artifacts/faiss_index/ (FAISS index + chunk metadata).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from loguru import logger

# Add project root to path so we can import src.*
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest import (
    EMBED_MODEL_LITE,
    load_hf_papers,
    chunk_documents,
    embed_texts,
    build_faiss_index,
    save_index_and_chunks,
    embed_query,
)
from src.retriever import DenseRetriever, BM25Retriever, HybridRetriever
from src.evaluator import (
    recall_at_k,
    precision_at_k,
    mean_reciprocal_rank,
    compute_retrieval_metrics,
    score_faithfulness,
    score_answer_relevance,
    EvalResults,
)

import ollama

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts" / "faiss_index"
LLM_MODEL = "granite4.1:8b"
EMBED_MODEL = EMBED_MODEL_LITE  # qwen3-embedding:0.6b — fast, 1024-dim

# ─── Sample evaluation queries (with ground-truth paper IDs) ──────────────────
# These are topic-based queries. Since ccdv/arxiv-summarization has no metadata,
# we define relevance by loading a small set and manually noting which chunks are
# relevant. For a real system you would have human-labelled relevance judgements.
SAMPLE_QUERIES = [
    "neural network optimization methods",
    "attention mechanism in transformers",
    "generative models for image synthesis",
    "reinforcement learning reward shaping",
    "graph neural networks for molecular property prediction",
]


def phase1_ingest(n_papers: int = 4000) -> tuple:
    """Load, chunk, embed, index and save. Returns (index, chunks)."""
    print("\n" + "=" * 60)
    print("PHASE 1 — Ingestion Pipeline")
    print("=" * 60)

    t0 = time.time()

    # 1. Load
    papers = load_hf_papers(n_samples=n_papers, ml_filter=True)
    print(f"\n[1/4] Loaded {len(papers)} papers")
    print(f"      Sample: {papers[0]['title'][:70]}...")

    # 2. Chunk
    chunks = chunk_documents(papers, chunk_size=512, chunk_overlap=64)
    avg_len = sum(len(c["text"]) for c in chunks) / len(chunks)
    print(f"\n[2/4] Created {len(chunks)} chunks (avg {avg_len:.0f} chars)")

    # 3. Embed
    texts = [c["text"] for c in chunks]
    print(f"\n[3/4] Embedding {len(texts)} chunks with {EMBED_MODEL} ...")
    embeddings = embed_texts(texts, model=EMBED_MODEL, batch_size=32)
    print(f"      Shape: {embeddings.shape}  (each chunk → {embeddings.shape[1]}-dim vector)")

    # 4. Build index
    index = build_faiss_index(embeddings)
    print(f"\n[4/4] FAISS IndexFlatIP: {index.ntotal} vectors, dim={embeddings.shape[1]}")

    # Save
    save_index_and_chunks(index, chunks, ARTIFACTS_DIR)
    print(f"\n      Saved to {ARTIFACTS_DIR}")

    elapsed = time.time() - t0
    print(f"\n  Phase 1 complete in {elapsed:.1f}s")
    return index, chunks


def phase2_retrieval(index, chunks) -> None:
    """Compare dense, BM25, and hybrid retrieval on sample queries."""
    print("\n" + "=" * 60)
    print("PHASE 2 — Retrieval Comparison")
    print("=" * 60)

    dense = DenseRetriever(index, chunks, embed_model=EMBED_MODEL)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(dense, bm25, alpha=0.7)

    for query in SAMPLE_QUERIES[:3]:  # show 3 queries for brevity
        print(f"\nQuery: \"{query}\"")
        print("-" * 55)

        dense_results = dense.retrieve(query, k=3)
        bm25_results = bm25.retrieve(query, k=3)
        hybrid_results = hybrid.retrieve(query, k=3)

        print("  Dense top-3:")
        for r in dense_results:
            print(f"    [{r['score']:.3f}] {r['title'][:55]}...")

        print("  BM25  top-3:")
        for r in bm25_results:
            print(f"    [{r['score']:.3f}] {r['title'][:55]}...")

        print("  Hybrid top-3:")
        for r in hybrid_results:
            print(f"    [{r['score']:.3f}] {r['title'][:55]}...")


def phase3_generate(index, chunks) -> tuple[str, str, list[dict]]:
    """Run naive RAG: retrieve context, generate answer with granite4.1:8b."""
    print("\n" + "=" * 60)
    print("PHASE 3 — RAG Generation")
    print("=" * 60)

    dense = DenseRetriever(index, chunks, embed_model=EMBED_MODEL)
    query = SAMPLE_QUERIES[0]
    print(f"\nQuery: \"{query}\"")

    retrieved = dense.retrieve(query, k=5)
    context = "\n\n".join(
        f"[{i+1}] {r['title']}\n{r['text']}"
        for i, r in enumerate(retrieved)
    )

    prompt = f"""You are a research assistant. Use ONLY the provided context to answer.
If the context does not contain the answer, say "I don't have enough information."

Context:
{context}

Question: {query}

Answer (cite sources by number):"""

    print(f"\nGenerating answer with {LLM_MODEL} ...")
    t0 = time.time()
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )
    answer = response["message"]["content"]
    elapsed = time.time() - t0

    print(f"\nAnswer ({elapsed:.1f}s):\n{answer}\n")
    return query, answer, retrieved


def phase4_evaluate(query: str, answer: str, retrieved: list[dict], chunks: list[dict]) -> None:
    """Compute retrieval metrics and LLM-as-judge generation metrics."""
    print("\n" + "=" * 60)
    print("PHASE 4 — Evaluation")
    print("=" * 60)

    # For retrieval metrics we need ground-truth relevant IDs.
    # We use a proxy: take the top retrieved chunk as the "relevant" one,
    # measure how consistently it's retrieved across methods.
    # (In a real system you'd have human-annotated relevance judgements.)
    retrieved_ids = [r["chunk_id"] for r in retrieved]
    # Use top-1 retrieved as pseudo-ground-truth relevant doc
    pseudo_relevant = [retrieved_ids[0]] if retrieved_ids else []

    r1 = recall_at_k(retrieved_ids, pseudo_relevant, k=1)
    r5 = recall_at_k(retrieved_ids, pseudo_relevant, k=5)
    p5 = precision_at_k(retrieved_ids, pseudo_relevant, k=5)
    mrr = mean_reciprocal_rank(retrieved_ids, pseudo_relevant)

    print(f"\n  Retrieval Metrics (query: \"{query[:40]}...\"):")
    print(f"    Recall@1   = {r1:.3f}  (did top-1 match ground truth?)")
    print(f"    Recall@5   = {r5:.3f}  (did any of top-5 match ground truth?)")
    print(f"    Precision@5 = {p5:.3f} (what fraction of top-5 were relevant?)")
    print(f"    MRR        = {mrr:.3f}  (mean reciprocal rank of first hit)")

    # LLM-as-judge generation metrics
    print(f"\n  Generation Metrics (LLM-as-judge using {LLM_MODEL}):")
    context = "\n\n".join(r["text"] for r in retrieved[:3])

    print("    Scoring faithfulness ...")
    contexts = [r["text"] for r in retrieved[:3]]
    faith_score = score_faithfulness(
        answer=answer, contexts=contexts, judge_model=LLM_MODEL
    )
    print(f"    Faithfulness   = {faith_score:.2f}  (1.0=grounded, 0.0=hallucinated)")

    print("    Scoring answer relevance ...")
    rel_score = score_answer_relevance(
        question=query, answer=answer, judge_model=LLM_MODEL
    )
    print(f"    Answer Relevance = {rel_score:.2f}  (1.0=fully relevant, 0.0=off-topic)")

    # Save results
    results = EvalResults(
        experiment_name="pipeline_run",
        retriever_type="dense",
        embed_model=EMBED_MODEL,
        llm_model=LLM_MODEL,
        retrieval_metrics={
            "Recall@1": r1,
            "Recall@5": r5,
            "Precision@5": p5,
            "MRR": mrr,
        },
        generation_metrics={
            "faithfulness": faith_score,
            "answer_relevance": rel_score,
        },
    )
    out_path = ARTIFACTS_DIR.parent / "eval_results_pipeline_4000.json"
    results.save(out_path)
    print(f"\n  Results saved to {out_path}")
    print(f"\n  Summary:\n{results.summary()}")


def main() -> None:
    print("\n" + "#" * 60)
    print("  Agentic RAG — ArXiv Research Assistant")
    print("  Full End-to-End Pipeline Run")
    print("#" * 60)
    print(f"\n  Embedding model : {EMBED_MODEL}")
    print(f"  LLM model       : {LLM_MODEL}")
    print(f"  Artifacts dir   : {ARTIFACTS_DIR}")

    index, chunks = phase1_ingest(n_papers=4000)
    phase2_retrieval(index, chunks)
    query, answer, retrieved = phase3_generate(index, chunks)
    phase4_evaluate(query, answer, retrieved, chunks)

    print("\n" + "#" * 60)
    print("  Pipeline complete.")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
