# Results and Limits

This page collects the full results from both evaluation rounds, interprets the key findings, explains what the numbers mean for practical use, and identifies what Part 2 still cannot fix — setting up the motivation for Part 3.

---

## Full results table

### 5-query evaluation (original benchmark — for comparison only)

| Strategy | Recall@5 | MRR |
|---|---|---|
| Dense baseline | 0.533 | 0.417 |
| BM25 (original tokenisation) | 0.533 | 0.667 |
| Hybrid alpha=0.7 | 0.533 | 0.417 |
| Hybrid RRF + Rerank | 0.467 | 0.500 |

### 20-query evaluation (current benchmark — the honest numbers)

| Strategy | Recall@5 | Precision@5 | MRR |
|---|---|---|---|
| Dense baseline | 0.250 | 0.120 | 0.299 |
| BM25 improved tokenisation | **0.367** | **0.190** | **0.441** |
| Hybrid alpha=0.7 | 0.267 | 0.130 | 0.308 |
| Hybrid RRF | 0.267 | 0.130 | 0.287 |
| Hybrid RRF + Rerank | 0.333 | 0.170 | 0.363 |

The 5-query numbers are preserved for transparency, not because they're trustworthy. Every number below compares against the 20-query benchmark.

---

## Key findings

### Finding 1: BM25 tokenisation was the biggest lever

The single most impactful change was switching from `text.lower().split()` to the `_tokenise()` function — lowercase, strip punctuation, remove stop words. This lifted BM25 MRR from approximately 0.333 (with the original whitespace split) to 0.441, a gain of roughly +0.108.

Why did this matter so much? ML paper abstracts are dense with exact technical terms. `"LoRA"`, `"RLHF"`, and `"attention mechanism"` are the signals that differentiate relevant from irrelevant. Under the original tokeniser, `"attention."` (end of sentence) and `"attention"` (mid-sentence) were different tokens, causing a term-frequency mismatch that quietly deflated BM25 scores across the board. Fixing this freed BM25 to do what it was designed for.

### Finding 2: Hybrid search did not beat BM25 alone

This is the most surprising result in Part 2. Every intuition about hybrid retrieval says it should outperform either component on its own — and in the literature, it usually does. But on this 20-query benchmark:

| Strategy | MRR |
|---|---|
| BM25 improved | **0.441** |
| Hybrid alpha=0.7 | 0.308 |
| Hybrid RRF | 0.287 |

Adding dense scores to the mix *hurt* performance.

The most likely explanation: with improved BM25 tokenisation, keyword search became very good on this corpus. The queries in the benchmark skew toward exact technical terms (model names, method names, benchmark names) — the kind of queries where BM25 dominates. Adding dense scores at alpha=0.3 introduced noise for these queries without compensating on the paraphrase queries where dense would help.

This is a corpus-composition effect, not a general finding about hybrid retrieval. A benchmark with more semantic / paraphrase queries would likely reverse this result.

!!! note "What this means for your own RAG system"
    Don't assume hybrid retrieval beats its components. Always measure on your actual query distribution. If your users ask exact-term questions (e.g., product names, SKU codes, error messages), a well-tuned BM25 may be better than any hybrid.

### Finding 3: Reranking adds consistent gain over any upstream retriever

The cross-encoder reranker improved MRR by +0.075 over Hybrid RRF (0.287 → 0.363). This is a consistent pattern: the reranker doesn't care which retriever produced the candidates, it just re-scores the top-20 with better query-document interaction.

This makes reranking a low-risk addition to any pipeline. You're not replacing retrieval — you're adding a second pass that can only improve rank order within the candidate set you already have.

### Finding 4: The 5-query numbers were misleading

The original BM25 MRR of 0.667 on 5 queries dropped to approximately 0.333 on 20 queries (before tokenisation improvements). That's not a system getting worse — it's noise in a small benchmark getting resolved. The 5-query evaluation happened to contain queries that BM25 handled unusually well. The 20-query benchmark has harder queries and a more realistic distribution.

---

## What Part 2 does not fix

### Single-hop retrieval

Every retrieval strategy in Part 2 runs exactly one search and returns a fixed top-k. If the right answer requires synthesising information from two different papers — one about LoRA architecture and another about quantisation efficiency — a single retrieval pass may not surface both.

Multi-hop retrieval requires iterative querying: retrieve once, read the results, formulate a follow-up query based on what's missing, retrieve again. Part 3 implements this as part of the LangGraph agent.

### Silent retrieval failure

When retrieval fails (the right paper is not in the top-5), the system proceeds to generation anyway. The LLM generates a plausible-sounding answer from whatever was retrieved, with no indication that the retrieved context was poor. There is no mechanism to detect "I found nothing useful" vs "I found exactly what was asked for".

### No hallucination detection

The generation stage in Part 2 is identical to Part 1: inject the top-5 chunks into a prompt, generate, return. The system does not verify whether the LLM's answer is actually supported by the retrieved chunks. An LLM that "knows" the answer from training data may generate a confident response that has nothing to do with the retrieved context.

---

## What Part 3 adds

Part 3 wraps the Advanced RAG retrieval pipeline in a LangGraph state machine that addresses both remaining failure modes:

1. **Document grading** — after retrieval, the LLM grades each chunk: "Is this document relevant to the query?" Chunks that fail grading are filtered out.
2. **Web search fallback** — if too few chunks pass grading, the agent triggers a web search (Tavily) to supplement the corpus.
3. **Answer faithfulness grading** — after generation, the LLM checks whether each claim in the answer is supported by the retrieved context. If not, the answer is flagged as a hallucination.
4. **Retry loop** — on a hallucination flag, the agent regenerates (up to twice) before returning.

This is the CRAG (Corrective RAG) architecture. Legacy small-scope runs showed high
faithfulness with little fallback, while current 4,000-baseline runs expose harder
query conditions (see `artifacts/eval_results/03_agentic_rag_4000.json` for current
agentic baseline behavior).

Continue to [Part 3 — Agentic RAG](../05_part3_agentic_rag/overview.md).
