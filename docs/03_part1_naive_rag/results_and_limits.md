# Results and Limitations

This page presents the real numbers from the Part 1 evaluation, explains honestly why the
eval was redesigned (and why the new numbers look worse), and identifies the specific
failure modes of naive dense retrieval that motivate every change made in Part 2.

---

## The results

Running `compute_retrieval_metrics()` over 20 queries with `k=5` on a 600-paper corpus
produces:

| Metric | Value | What it means |
|---|---|---|
| Recall@5 | **0.250** | In 1 in 4 queries, at least one relevant paper appeared in the top 5 |
| Precision@5 | **0.120** | On average, 0.6 of the 5 retrieved chunks were relevant |
| MRR | **0.299** | The first relevant result appeared, on average, around rank 3–4 |

These are your baseline numbers. Every improvement in Part 2 will be judged against them.

---

## Why the numbers look lower than the earlier 5-query results

You may have seen earlier notebook cells that reported much higher numbers:

| Eval set | Recall@5 | Precision@5 | MRR |
|---|---|---|---|
| Old — 5 queries, 300 papers | 0.533 | 0.240 | 0.417 |
| New — 20 queries, 600 papers | 0.250 | 0.120 | 0.299 |

The drop is almost entirely explained by the eval set redesign, not by a change in retriever
quality. Three specific factors account for the difference:

**1. More queries, more variance exposure**

With 5 queries, a single lucky result moves Recall by 20 percentage points. The 5-query set
happened to contain several queries where the relevant papers were densely represented in the
300-paper corpus — which inflated all three metrics. At 20 queries, the noise averages out and
the true performance of the retriever becomes visible.

**2. Larger corpus — harder retrieval**

300 papers produce roughly 350–400 chunks. 600 papers produce roughly 700–900 chunks. Doubling
the index size doubles the competition for each query: the relevant chunk now has to outscore
twice as many irrelevant candidates. All else being equal, retrieval precision falls as the
corpus grows.

**3. Harder queries in the expanded set**

The original 5 queries were implicitly chosen to match the small corpus well. The 20-query
set includes topics that are less well-represented: web search integration for RAG, LLM
self-correction, speculative decoding, and others. These queries stress the retriever more.

!!! note "The 5-query numbers were not dishonest — they were on a different setup"
    Reporting both sets of numbers and explaining the difference is exactly the right approach.
    A reader who only saw 0.533 and then ran the pipeline on their own 600-paper corpus would
    be confused when they saw 0.250. Honest evaluation means being explicit about what the
    eval set covers and what it does not.

---

## Limitations of naive RAG

The numbers expose three structural weaknesses in the naive pipeline.

**Limitation 1 — Dense retrieval misses exact terms**

`qwen3-embedding:0.6b` is trained to capture semantic similarity. It excels at synonyms and
paraphrases. But a query for "RLHF" or "InstructGPT" requires exact-term matching. If the
embedding model does not place those specific strings close to their conceptual neighbours,
the relevant papers will not rank highly. BM25 — introduced in Part 2 — handles exact-term
queries correctly because it operates on token frequency, not vector geometry.

**Limitation 2 — No hallucination check**

The `naive_rag()` function hands the retrieved context to `granite4.1:8b` and trusts it.
There is no check on whether the answer is actually supported by the context. When the
retrieved chunks are only weakly relevant, the model has a choice: admit ignorance (following
the prompt instruction) or blur the line between retrieved evidence and parametric knowledge.
In practice, it often does the latter. The agentic pipeline in Part 3 adds a faithfulness
scorer to catch this.

**Limitation 3 — No feedback loop**

If the retrieval is poor, there is no mechanism to try again. The pipeline runs once: retrieve,
prompt, generate. A single query with a vague phrasing ("what are some recent methods?")
will retrieve a random-looking set of chunks because the query vector has no sharp semantic
target. Agentic RAG solves this with query rewriting and iterative retrieval.

---

## What these failures motivate

| Failure mode | Fix in Part 2 | Fix in Part 3 |
|---|---|---|
| Dense misses exact terms | BM25 + hybrid search | — |
| Noisy top-5 (irrelevant chunks) | Cross-encoder reranker | — |
| Vague queries | Query rewriting | Agent decides whether to rephrase |
| No grounding check | — | LLM-as-judge faithfulness scorer |
| No fallback when corpus is empty | — | Web search fallback |

The metrics in Part 2 will quantify exactly how much each of these fixes contributes. Part 3
will show how a state machine can combine all of them into a self-correcting loop.

!!! tip "The honest baseline is valuable"
    MRR = 0.299 is not a failure — it is a starting point with a clear measurement. Many
    production RAG systems are deployed without any retrieval evaluation at all. Knowing
    your baseline means you can tell whether a change is actually an improvement.

---

## What's next

You have completed Part 1. The pipeline works, the baseline is measured, and the failure
modes are understood. Head to **Part 2 — Advanced RAG** to see how hybrid retrieval and
reranking move MRR from 0.299 to 0.441.
