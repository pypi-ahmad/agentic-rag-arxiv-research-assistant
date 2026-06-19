# Evaluation

Building a RAG pipeline is the easy part. Knowing whether it is any good requires a
structured evaluation. This page covers the 20-query eval set, the ground-truth construction
method, the three metrics, and why each design choice matters.

---

## The 20-query eval set

The evaluation set contains 20 questions that span the range of topics in the corpus:
attention mechanisms, contrastive learning, LLM inference efficiency, diffusion models,
calibration, LoRA, RAG itself, chain-of-thought prompting, and more. Each question has a
`relevant_ids` list — the paper IDs that a retriever should return to be considered correct.

```python
eval_queries = [
    {
        "question": "What is attention head heterogeneity in transformers?",
        "relevant_ids": find_relevant_ids_by_keywords(
            ["attention head", "head heterogeneity", "hydrahead", "head specialization"],
            papers,
        ),
    },
    {
        "question": "How does contrastive learning work?",
        "relevant_ids": find_relevant_ids_by_keywords(
            ["contrastive learning", "contrastive loss", "simclr", "nce loss"],
            papers,
        ),
    },
    # ... 18 more queries
]
```

The `relevant_ids` are derived from the actual corpus at runtime, not hardcoded. This means
the eval set adapts to whatever 600 papers were loaded — a query only has relevant papers if
those papers are actually in the corpus.

---

## `find_relevant_ids_by_keywords()` — multi-keyword OR logic

The ground-truth function accepts a list of keywords and returns paper IDs where any
keyword appears in the title or abstract:

```python
def find_relevant_ids_by_keywords(
    keywords: list[str], papers: list, top_n: int = 3
) -> list[str]:
    matched = []
    for p in papers:
        combined = (p["title"] + " " + p["abstract"]).lower()
        if any(kw.lower() in combined for kw in keywords):
            matched.append(p["id"])
    return matched[:top_n]
```

This is OR logic — a paper is relevant if it contains *any* of the keywords. This design
choice has a significant impact on eval quality. Consider the alternative: a single keyword
like `"attention head"` might match zero papers in a given 600-paper sample. A query with
zero relevant papers cannot contribute meaningfully to Recall or Precision — both would be
zero regardless of what the retriever returns. Over 20 queries, several zero-ground-truth
queries would drag every metric down and tell us nothing about retrieval quality.

!!! note "Why multi-keyword OR captures synonyms"
    The query "How does attention head heterogeneity affect transformers?" could be answered
    by papers that use any of: "attention head," "head heterogeneity," "HydraHead," or
    "head specialization." These are different surface forms for the same concept. The OR
    list ensures that papers using any of these forms are counted as relevant — mirroring
    how a human expert would evaluate retrieval quality.

---

## `compute_retrieval_metrics()` — the three numbers

```python
from src.evaluator import compute_retrieval_metrics

naive_retrieval_metrics = compute_retrieval_metrics(
    queries=eval_queries,
    retriever=dense_retriever,
    k=5,
)
```

For each query, the function calls `retriever.retrieve(query, k=5)`, extracts the
`paper_id` from each result, and computes three metrics:

**Recall@5** — did the right papers appear in the top 5?

```
Recall@5 = |relevant ∩ top-5| / |relevant|
```

If a query has 2 relevant papers and the retriever returns both in the top 5, Recall@5 = 1.0.
If it returns only one, Recall@5 = 0.5. Recall measures completeness — did we find everything
that should have been found?

**Precision@5** — how many of the top 5 were actually relevant?

```
Precision@5 = |relevant ∩ top-5| / 5
```

If 1 of the 5 returned chunks is relevant, Precision@5 = 0.20. Precision measures noise —
how many of the results the LLM has to read are actually useful?

**MRR (Mean Reciprocal Rank)** — how high up was the first relevant result?

```
MRR = 1 / rank_of_first_relevant
```

| First hit at rank | Contribution to MRR |
|---|---|
| 1 | 1.000 |
| 2 | 0.500 |
| 3 | 0.333 |
| 5 | 0.200 |
| Not found | 0.000 |

MRR is averaged across all 20 queries. It captures whether the most relevant document is at
the top — critical for generation quality, because the LLM reads the context in order and
pays more attention to earlier documents.

!!! info "Definition — Why MRR over MAP?"
    Mean Average Precision (MAP) rewards returning all relevant documents early. MRR only
    cares about the first relevant document. For RAG, MRR is more appropriate because the
    LLM typically uses the top-1 or top-2 chunks most heavily when generating an answer.
    Getting one strong relevant chunk at rank 1 matters more than having all relevant
    chunks somewhere in the top 10.

---

## What's next

With the metrics defined and computed, the final page — [Results and Limitations](results_and_limits.md) — presents the actual numbers from the 20-query run, explains why
they are lower than the earlier 5-query results, and identifies the specific failure modes
of naive RAG that Part 2 is designed to fix.
