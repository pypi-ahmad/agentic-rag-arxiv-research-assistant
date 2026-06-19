# Part 2 — Advanced RAG Overview

Part 1 gave us a working RAG pipeline: embed the corpus, search with FAISS, generate with the LLM. It answers questions. But when you measure it honestly against 20 diverse queries, it only achieves an MRR of 0.299. That means the most relevant paper lands at rank 3–4 on average, and sometimes it doesn't appear in the top 5 at all.

Part 2 diagnoses why, then fixes it — without changing the LLM, the corpus, or the chunking strategy. Everything we improve is in the retrieval layer.

---

## What Part 1 gets wrong

Semantic (dense) search is good at paraphrase. Ask "how do LLMs generate text?" and it finds papers about autoregressive decoding even though neither phrase appears verbatim in the other. That's the strength of embeddings.

But ML papers are full of exact terminology: model names (`Qwen3`), method names (`LoRA`, `FlashAttention`), benchmark names (`SQuAD`, `MMLU`). When a user queries "BM25 for document ranking", a semantic model might return papers about neural information retrieval that are semantically close but not what the user meant. The exact string "BM25" is a strong signal — and a cosine similarity search has no way to exploit it.

!!! warning "Retrieval is the biggest lever"
    Swapping the LLM from Llama 3 to GPT-4 on a broken retrieval pipeline doesn't fix bad answers — it generates better-sounding wrong answers. The document that isn't retrieved cannot be cited.

---

## Three layers added in Part 2

Part 2 adds three improvements in sequence, each measured independently:

```
Query
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  Layer 1 — BM25 Retriever                           │
│  Keyword-based exact-match scoring. Improves recall  │
│  for queries with specific technical terms.          │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  Layer 2 — Hybrid Retriever                         │
│  Fuses BM25 + dense scores into one ranked list.    │
│  Two fusion strategies: alpha blend and RRF.         │
└─────────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────────────────┐
│  Layer 3 — Cross-Encoder Reranker                   │
│  Re-scores top-20 candidates with a 22MB model.     │
│  Fixes rank order after fusion.                     │
└─────────────────────────────────────────────────────┘
  │
  ▼
Top-5 results → LLM generation
```

Each layer is a separate class in `src/retriever.py`: `BM25Retriever`, `HybridRetriever`, and `Reranker`. You can swap them in and out individually, which is exactly how the results table is produced.

---

## What the numbers say

Here is the full 20-query evaluation across all strategies:

| Strategy | Recall@5 | Precision@5 | MRR |
|---|---|---|---|
| Dense baseline | 0.250 | 0.120 | 0.299 |
| BM25 improved tokenisation | **0.367** | **0.190** | **0.441** |
| Hybrid alpha=0.7 | 0.267 | 0.130 | 0.308 |
| Hybrid RRF | 0.267 | 0.130 | 0.287 |
| Hybrid RRF + Rerank | 0.333 | 0.170 | 0.363 |

The single most impactful change — improving BM25 tokenisation — raises MRR from 0.299 to 0.441. That's a 47% gain from a 3-line code change.

There's also a surprising result: RRF (Reciprocal Rank Fusion), the theoretically superior rank-based fusion method, does not beat the alpha-blend fusion on this corpus. The evaluation page explains why in detail.

---

## Where this leaves us

After Part 2, retrieval quality is meaningfully better. But the pipeline is still single-hop — it runs one retrieval pass and trusts whatever comes back. If retrieval fails silently (the right paper isn't in the top 5), the LLM generates a plausible-sounding answer with no warning.

Part 3 fixes this by adding a self-correction loop: the system grades its own retrieved documents, falls back to web search when the corpus is insufficient, grades its own answers for hallucinations, and retries if something goes wrong.

Continue reading Part 2:

- [BM25 in Practice](bm25_in_practice.md) — the tokenisation problem and how to fix it
- [Hybrid Retrieval](hybrid_retrieval.md) — alpha fusion and RRF explained side by side
- [Cross-Encoder Reranking](cross_encoder_reranking.md) — two-stage retrieve-then-rerank
- [Evaluation](evaluation.md) — the 20-query benchmark setup and results
- [Results and Limits](results_and_limits.md) — full comparison table and what's next
