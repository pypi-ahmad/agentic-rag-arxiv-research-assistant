# Reading the Numbers Honestly

The retrieval metrics in this project tell two different stories depending on which
row of the table you are reading. The old 5-query numbers look impressive. The new
20-query numbers look disappointing. Understanding *why* they differ is more valuable
than either set of numbers in isolation — it is the core lesson of applied evaluation.

---

## Why the Old 5-Query Numbers Are Misleading

The initial evaluation produced these results:

| Strategy | Recall@5 | Precision@5 | MRR |
|---|---|---|---|
| Dense | 0.5333 | 0.2400 | 0.4167 |
| BM25 | 0.5333 | 0.2400 | 0.6667 |
| Hybrid α=0.7 | 0.5333 | 0.2400 | 0.4167 |
| Hybrid RRF + Rerank | 0.4667 | 0.2000 | 0.5000 |

There are three things wrong with this picture.

### Problem 1 — Five queries cannot distinguish strategies

With only 5 queries, the difference between a strategy that ranks the relevant doc at
position 1 versus position 3 amounts to a single query. That single query swings MRR
by 0.13. The strategies are not actually performing similarly — we simply cannot
measure the difference with this sample size.

Notice that Dense, BM25, and Hybrid α=0.7 all score *identically* on Recall@5 and
Precision@5 (0.5333 / 0.2400). This is statistically implausible if the strategies
are genuinely different. The real explanation: with 5 queries, all three strategies
happen to retrieve the same set of relevant documents (or the same number of them),
and the Recall/Precision metrics cannot separate them.

### Problem 2 — Few distractors inflate precision

The 300-paper corpus is small. When 300 chunks are indexed, many queries have
semantically close documents in the top 5 even by chance. On a 600-paper corpus the
same query has twice as many potential distractors competing for the top-5 slots.
Precision inevitably drops as corpus size grows — this is expected and correct
behaviour, not a regression.

### Problem 3 — Trivial ground truth

The original ground truth used exact single-keyword matching: a document is relevant
if its title contains a specific phrase like "BM25 ranking". This misses:

- Synonyms ("Okapi BM25", "best match 25")
- Related papers that use the concept without naming it in the title
- Abstracts where the relevant content is in the body, not the title

Single-keyword ground truth systematically understimates recall because it excludes
relevant documents that use different vocabulary. It also means the 5 queries were
cherry-picked around topics with obvious, easy-to-match ground truth — not a random
sample of the queries real users would ask.

---

## Why the New 20-Query Numbers Are Honest

The expanded evaluation shows:

| Strategy | Recall@5 | Precision@5 | MRR |
|---|---|---|---|
| Dense | 0.2500 | 0.1200 | 0.2992 |
| BM25 (improved) | **0.3667** | **0.1900** | **0.4408** |
| Hybrid α=0.7 | 0.2667 | 0.1300 | 0.3075 |
| Hybrid RRF + Rerank | 0.3333 | 0.1700 | 0.3625 |

These numbers look lower. They are lower. They are also more accurate.

### What changed

**600-paper corpus** — twice as many distractors. Every relevant document now competes
against twice as many irrelevant ones for the top-5 slots. Precision@5 of 0.19 for
BM25 means roughly 1 in 5 retrieved chunks is relevant — which is actually excellent
given the corpus density.

**20 diverse queries** — the queries span a broader range of topics, including ones
where dense retrieval is weak (keyword-heavy technical queries) and ones where BM25
is weak (conceptual paraphrase queries). A fair eval set is not designed to make
any particular strategy look good.

**Multi-keyword OR ground truth** — a document is now relevant if its text contains
any of several synonymous phrases. This is harder to satisfy precisely because more
documents qualify, but it is far more representative of real relevance than a single
keyword match.

**No cherry-picking** — the 20 queries were selected to cover the major topics in the
tutorial, not to maximise any particular metric.

---

## The Lesson: Eval Set Design Determines What You Are Measuring

This is the most important takeaway from the entire project:

!!! danger "The fundamental evaluation trap"
    A small, easy eval set does not measure whether your system is good.
    It measures whether your system is overfit to that eval set.

    If all three of your retrieval strategies score identically on Recall@5,
    you have not found a universal truth about them. You have found that your
    eval set cannot distinguish them.

The pattern is common in ML systems:

1. Build a prototype.
2. Write a quick eval with 5 easy test cases.
3. Iterate until the 5 easy test cases all pass.
4. Ship, and discover the system fails on real-world queries.

The 5-query eval produced MRR numbers in the 0.42–0.67 range. These numbers would
suggest the system is quite capable. The 20-query eval produced MRR numbers in the
0.30–0.44 range, which more honestly reflects a system that is useful but not yet
production-grade.

The difference is not that the system got worse between evaluations. The difference
is that the *measurement* became honest.

---

## When MRR Matters More Than Recall@k for RAG

Both Recall@5 and MRR are reasonable metrics for retrieval evaluation, but they
measure different things and one is usually more important for RAG.

### Recall@5 measures coverage

Recall@5 answers: *does the relevant document appear somewhere in the top 5?*

This is the right metric when:

- You are passing all top-k results to the LLM (the LLM can find the relevant one)
- False negatives are expensive (missing the document is worse than some noise)
- Your LLM is good at ignoring irrelevant context

### MRR measures rank quality

MRR answers: *how early does the first relevant document appear?*

This is the right metric when:

- Your LLM is sensitive to position in the context window
- You are truncating the context to save tokens (only passing top-2 or top-3)
- The LLM tends to anchor on the first piece of evidence it encounters

For RAG systems in practice, **MRR is the primary metric**. Here is why:

1. **Context window budget** — passing 10 chunks to a language model is expensive.
   Most deployments use top-3 to top-5 chunks. If the relevant document is at rank
   6, it is never seen.

2. **LLM attention bias** — language models have a well-documented recency and primacy
   bias. A document at rank 1 is more likely to be used than a document at rank 5
   even if both are present in the context.

3. **One good document is often enough** — for factoid and summary queries, the LLM
   needs one high-quality, relevant chunk. Getting it at rank 1 (MRR contribution:
   1.0) versus rank 5 (MRR contribution: 0.2) is a large practical difference.

Recall@k is still worth tracking: a system with MRR 0.44 but Recall@5 of 0.20 has
a different failure mode (it finds the right document quickly when it finds it at all,
but misses it completely on many queries) than a system with MRR 0.35 and Recall@5
of 0.45 (finds something relevant most of the time but not always first).

---

## Summary

| Claim | 5-query eval verdict | 20-query eval verdict |
|---|---|---|
| Dense and BM25 perform identically | "True" (artefact of sample size) | False — BM25 +47% MRR |
| MRR ≈ 0.50–0.67 is achievable | "True" (overfit to easy eval) | No — honest MRR is 0.30–0.44 |
| Hybrid always beats single-stage | "Supported" | Partially — only with reranking |
| System is production-ready | Appears yes | Directionally yes, needs more eval |

A mature ML practitioner should feel *better* about a system after the honest numbers
come in, not worse — because the honest numbers are what you will actually see in
production.
