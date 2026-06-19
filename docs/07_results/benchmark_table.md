# Full Benchmark Table

This page collects every number produced during development — retrieval metrics from
the 20-query eval and agent-behaviour metrics from the 10-query CRAG eval — in one
place so you can compare strategies side by side.

If you want to understand *why* the old 5-query numbers look better than the new 20-query
numbers, read [Reading the Numbers Honestly](reading_the_numbers.md) first.

---

## Retrieval Benchmark (Dense, BM25, Hybrid)

The table below covers all strategies tested across two evaluation rounds.
The first four rows used the original 5-query eval set on a 300-paper corpus.
The bottom five rows used the expanded 20-query eval set on the full 600-paper corpus.
**Do not compare rows across the horizontal line** — they measure different things.

| Strategy | Eval set | Recall@5 | Precision@5 | MRR |
|---|---|---|---|---|
| Dense (baseline) | 5q / 300 papers | 0.5333 | 0.2400 | 0.4167 |
| BM25 (original tokenisation) | 5q / 300 papers | 0.5333 | 0.2400 | 0.6667 |
| Hybrid α=0.7 | 5q / 300 papers | 0.5333 | 0.2400 | 0.4167 |
| Hybrid RRF + Rerank | 5q / 300 papers | 0.4667 | 0.2000 | 0.5000 |
| **—** | **—** | **—** | **—** | **—** |
| Dense (baseline) | 20q / 600 papers | 0.2500 | 0.1200 | 0.2992 |
| BM25 (improved tokenisation) | 20q / 600 papers | **0.3667** | **0.1900** | **0.4408** |
| Hybrid α=0.7 | 20q / 600 papers | 0.2667 | 0.1300 | 0.3075 |
| Hybrid RRF | 20q / 600 papers | 0.2667 | 0.1300 | 0.2867 |
| Hybrid RRF + Rerank | 20q / 600 papers | 0.3333 | 0.1700 | 0.3625 |

!!! info "Metric definitions"
    **Recall@5** — fraction of relevant documents found in the top 5 results, averaged
    across all queries. A score of 0.37 means roughly 37% of ground-truth relevant
    documents appeared in the top 5.

    **Precision@5** — fraction of the top 5 results that are actually relevant.
    A score of 0.19 means about 1 in 5 retrieved chunks is relevant.

    **MRR (Mean Reciprocal Rank)** — average of 1/rank for the first relevant result.
    MRR = 0.44 means the first relevant document appears at rank 2.3 on average.
    This is the primary metric for RAG systems because it measures how quickly you
    surface *something* useful — the LLM only needs one good document to answer well.

---

## Strategy-by-Strategy Commentary

### Dense baseline (MRR 0.299 on honest eval)

The starting point. `qwen3-embedding` converts every query into a 1024-dimensional
vector and finds the nearest chunks by dot product. It works well on paraphrase-style
queries where the question closely mirrors abstract language. It fails on keyword-heavy
queries ("BM25 Robertson 1994") and topic-spanning queries that require multiple
distinct concepts to all be present.

### BM25 with improved tokenisation (MRR 0.441 — best retriever)

The decisive winner. After fixing the tokeniser to split on hyphens and slashes
(so "self-attention" becomes `["self", "attention"]` instead of a single unseen
token), BM25 recovers strongly. Its exact-match strength dominates on technical
queries — "LoRA", "CRAG", "GraphSAGE" are all low-frequency terms that BM25
weights very highly and dense retrieval often misses.

**MRR improvement over dense baseline: +47%.**

### Hybrid α=0.7 (MRR 0.308)

Blending dense and BM25 scores with `score = 0.7 × dense + 0.3 × bm25` improved
MRR slightly over dense alone, but fell short of pure BM25. On this 600-paper corpus
the BM25 signal is strong enough that diluting it with the weaker dense scores hurts.
On a larger, more semantically diverse corpus alpha-weighting would likely pay off.

### Hybrid RRF (MRR 0.287)

Reciprocal Rank Fusion merges two ranked lists without needing score calibration.
On this corpus it underperformed alpha-weighted hybrid and matched it on Recall@5.
The gap is narrow (0.287 vs 0.308 MRR) and corpus-specific — RRF is generally
preferable in production because it is more stable when score distributions shift.

### Hybrid RRF + Cross-Encoder Reranking (MRR 0.363)

Adding a `cross-encoder/ms-marco-MiniLM-L-6-v2` reranker after RRF retrieval boosts
MRR by 27% over vanilla RRF. The cross-encoder reads the full query and each candidate
chunk together, catching relevance signals that both retrieval stages missed. The cost
is latency: a synchronous cross-encoder call per query. In a production system this
would be made async or cached.

**Best single-stage: BM25 improved. Best two-stage: RRF + Rerank.**

---

## CRAG Agent Behaviour

These numbers come from a separate 10-query evaluation that measures the agent's
*behaviour* rather than retrieval rank. Each query has a known-relevant paper in the
corpus and a hand-written reference answer for faithfulness grading.

| Metric | Before (5q / 300 papers / 3-tier) | After (10q / 600 papers / 2-tier) |
|---|---|---|
| Queries with relevant doc in corpus | 4 / 5 (80%) | 10 / 10 (100%) |
| Web search triggered | 1 / 5 (20%) | 0 / 10 (0%) |
| Faithful answers | 3 / 5 (60%) | 7 / 10 (70%) |

### Commentary

**Corpus docs relevant (80% → 100%)** is the largest single driver of improvement.
When the corpus jumps from 300 to 600 papers, every one of the 10 test queries now
has at least one relevant abstract present. The agent is not forced to fall back to
web search because the knowledge it needs actually exists locally. This is the most
important structural finding: *corpus coverage determines ceiling performance*.

**Web search triggered (20% → 0%)** is a direct consequence of better corpus
coverage. Fewer gaps means fewer fallbacks. Whether zero web fallbacks is a goal
depends on the deployment — in some settings web fallback is desirable for recent
papers not yet in the index. Here it confirms the corpus is sufficient for the
test queries.

**Faithful answers (60% → 70%)** improved despite the harder eval set (10 queries
vs 5) and stricter threshold tuning. The 2-tier threshold (good / bad, no
"ambiguous" middle tier) forces the agent to make cleaner decisions and reduces
the case where an ambiguous document passes grading and contaminates the context.

!!! warning "Small sample caveat"
    10 queries is not a statistically robust eval. A ±10% fluctuation (±1 query)
    in either direction would not be meaningful. These numbers are directional
    indicators, not production-grade estimates. A rigorous eval would require
    100+ queries with multiple annotators per query.

---

## Clear Winners by Category

| Category | Winner | Key metric |
|---|---|---|
| Best single-stage retriever | BM25 (improved tokenisation) | MRR 0.441 |
| Best two-stage pipeline | Hybrid RRF + Cross-Encoder Rerank | MRR 0.363 |
| Best agent behaviour | CRAG 2-tier, 600-paper corpus | 70% faithful, 0% web fallback |
| Largest improvement lever | BM25 tokenisation fix | +47% MRR over dense baseline |
| Best corpus coverage | 600-paper corpus | 100% queries have relevant doc |
