# Evaluation

Measuring retrieval quality is harder than it looks. The naive approach — pick 5 queries, eyeball the results, call it good — is almost always misleading. This page explains how the 20-query benchmark is set up, why we replaced an earlier 5-query version, and how the `compute_retrieval_metrics()` function works.

---

## Why evaluation is easy to get wrong

There are two common failure modes in RAG evaluation:

**Too few queries.** With 5 queries, random variation dominates the signal. A system with MRR 0.667 on 5 queries might have MRR 0.350 on 50 queries. The small set happened to contain queries that this system handles well.

**Hand-picked queries.** If you design test queries after looking at the system's outputs (even informally), you've contaminated the benchmark. Queries that "feel like they should work" will work for any reasonable retrieval system.

This project ran into exactly the first problem. The original 5-query benchmark produced results like this:

| Strategy | Recall@5 | MRR |
|---|---|---|
| Dense | 0.533 | 0.417 |
| BM25 (original tokenisation) | 0.533 | 0.667 |
| Hybrid alpha=0.7 | 0.533 | 0.417 |
| Hybrid RRF + Rerank | 0.467 | 0.500 |

These numbers look reasonable, but they're noisy. When the benchmark was expanded to 20 queries — same methodology, same corpus — results dropped significantly across all strategies and the relative ranking of methods changed.

!!! warning "Lower numbers on a harder benchmark are more informative"
    The 5-query BM25 result (MRR 0.667) looked impressive. The 20-query result (MRR 0.441 with improved tokenisation, 0.333 with original) is the honest one. Reporting the inflated number would have suggested the system is better than it is.

---

## The 20-query benchmark

The benchmark is defined as a list of `(query, relevant_chunk_ids)` pairs. Relevant chunk IDs are manually identified by checking which chunks in the corpus actually answer each query.

The 20 queries were designed to cover different retrieval difficulty levels:

- **Exact-term queries** — "BM25 information retrieval scoring" (BM25 should win)
- **Semantic queries** — "how do language models handle long documents" (dense should win)
- **Multi-aspect queries** — "efficient training of large language models" (multiple relevant papers)
- **Rare-term queries** — specific model or dataset names from recent papers

Queries were written before inspecting the retrieval results to avoid contamination. The relevant chunk IDs were then identified by searching the corpus manually.

---

## The metrics

Three metrics are computed per strategy:

**Recall@5** — what fraction of relevant documents appear in the top-5?

```
Recall@5 = |relevant ∩ top_5| / |relevant|
```

If a query has 3 relevant chunks and 2 appear in top-5, Recall@5 = 2/3 ≈ 0.667.

**Precision@5** — what fraction of the top-5 are relevant?

```
Precision@5 = |relevant ∩ top_5| / 5
```

If 2 out of 5 retrieved chunks are relevant, Precision@5 = 0.4.

**MRR (Mean Reciprocal Rank)** — what is the reciprocal rank of the *first* relevant document, averaged over all queries?

```
RR(query) = 1 / rank_of_first_relevant
MRR = mean(RR) over all queries
```

MRR = 1.0 means the top result is always relevant. MRR = 0.5 means the first relevant result is at rank 2 on average. MRR = 0.299 (the dense baseline) means the first relevant result is near rank 3–4.

MRR is the primary metric because it measures what users experience: how far down the list do they need to scroll before seeing something useful?

---

## `compute_retrieval_metrics()`

The function is defined in notebook 02 (not in `src/`) since it's evaluation scaffolding rather than production code:

```python
def compute_retrieval_metrics(
    retriever,
    eval_set: list[tuple[str, list[str]]],
    k: int = 5,
) -> dict:
    recalls, precisions, rr_scores = [], [], []

    for query, relevant_ids in eval_set:
        results = retriever.retrieve(query, k=k)
        retrieved_ids = [r["chunk_id"] for r in results]

        relevant_set = set(relevant_ids)
        hits = [rid for rid in retrieved_ids if rid in relevant_set]

        recall = len(hits) / len(relevant_set) if relevant_set else 0.0
        precision = len(hits) / k

        # MRR: find rank of first hit
        rr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in relevant_set:
                rr = 1.0 / rank
                break

        recalls.append(recall)
        precisions.append(precision)
        rr_scores.append(rr)

    return {
        "recall@k":    round(sum(recalls) / len(recalls), 3),
        "precision@k": round(sum(precisions) / len(precisions), 3),
        "mrr":         round(sum(rr_scores) / len(rr_scores), 3),
    }
```

Running this function against each retriever strategy on the same 20-query set produces a directly comparable table, because the only variable is the retriever.

---

## Running the evaluation

In notebook 02, the evaluation loop looks like this:

```python
strategies = {
    "Dense":               dense_retriever,
    "BM25 improved":       bm25_retriever,
    "Hybrid alpha=0.7":    hybrid_alpha,
    "Hybrid RRF":          hybrid_rrf,
    "Hybrid RRF + Rerank": rerank_pipeline,  # wraps retrieve(k=20) + rerank(top_k=5)
}

results = {}
for name, retriever in strategies.items():
    results[name] = compute_retrieval_metrics(retriever, EVAL_SET, k=5)
    print(f"{name:30s}  {results[name]}")
```

Each strategy gets exactly the same 20 queries and the same relevant-ID ground truth. The output is the table shown in the [Results and Limits](results_and_limits.md) page.
