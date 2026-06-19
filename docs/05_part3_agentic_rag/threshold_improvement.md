# Grading Threshold Improvement

This page documents Improvement 5: the change from a 3-tier relevance threshold to a
2-tier threshold. It is the smallest code change in the entire project — one line in a
routing function — and it produced the largest improvement in agent behaviour.

---

## The original 3-tier approach

The initial implementation classified retrieval quality into three buckets:

```python
# OLD — 3-tier threshold
def grade_documents(state: GraphState) -> dict:
    # ... (grading loop unchanged) ...

    if n_relevant >= 2:
        retrieval_grade = "relevant"
    elif n_relevant == 1:
        retrieval_grade = "ambiguous"   # single match → uncertain
    else:
        retrieval_grade = "irrelevant"
```

And the routing function handled three cases:

```python
# OLD — 3-tier routing
graph_builder.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "relevant":   "generate_answer",
        "ambiguous":  "web_search",      # single match treated as insufficient
        "irrelevant": "web_search",
    }
)
```

The logic was intuitive: one matching document feels uncertain, so supplement with web
results. But this reasoning had a flaw.

---

## Why the 3-tier logic backfired

Consider a highly specific query like *"What is the approach proposed in the paper on
sparse mixture-of-experts routing?"*. The corpus contains exactly one paper that
directly addresses this. The LLM judge correctly identifies that one document as
relevant. Under the 3-tier scheme, `n_relevant == 1` triggers `ambiguous`, which sends
the agent to `web_search`.

The web results return general blog posts about mixture-of-experts. These are noisier
and less precise than the single highly-relevant corpus document that was already found.
The agent then generates an answer from web noise instead of from the exact paper it
retrieved.

The 3-tier scheme optimised for quantity of evidence over quality of evidence. A single
high-relevance corpus document is often better than three web results.

The numbers confirmed it. On a 5-query pilot:

| Metric | 3-tier scheme |
|---|---|
| Web search triggered | 1/5 (20%) |
| Faithful answers | 3/5 (60%) |

One of those web-search triggers was a query that had a perfectly good corpus document
rated `n_relevant == 1`. The web fallback made it worse.

---

## The fix: 2-tier threshold

The change is a single conditional:

```python
# NEW — 2-tier threshold (active in notebook 03)
if n_relevant >= 1:
    retrieval_grade = "relevant"
else:
    retrieval_grade = "irrelevant"
```

And the routing simplifies to two branches:

```python
# NEW — 2-tier routing
graph_builder.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "relevant":   "generate_answer",
        "irrelevant": "web_search",
    }
)
```

One relevant document is enough. Only a complete failure — zero relevant documents —
triggers the web fallback.

---

## Side-by-side comparison

=== "Old (3-tier)"

    ```python
    # In grade_documents node
    if n_relevant >= 2:
        retrieval_grade = "relevant"
    elif n_relevant == 1:
        retrieval_grade = "ambiguous"
    else:
        retrieval_grade = "irrelevant"
    ```

    ```python
    # Routing map
    {
        "relevant":   "generate_answer",
        "ambiguous":  "web_search",
        "irrelevant": "web_search",
    }
    ```

=== "New (2-tier)"

    ```python
    # In grade_documents node
    if n_relevant >= 1:
        retrieval_grade = "relevant"
    else:
        retrieval_grade = "irrelevant"
    ```

    ```python
    # Routing map
    {
        "relevant":   "generate_answer",
        "irrelevant": "web_search",
    }
    ```

---

## Measured improvement

Expanding the evaluation from 5 queries to 10 and switching to the 2-tier threshold:

| Metric | 3-tier (5 queries) | 2-tier (10 queries) |
|---|---|---|
| Web search triggered | 1/5 (20%) | 0/10 (0%) |
| Faithful answers | 3/5 (60%) | 7/10 (70%) |
| Relevant docs found | — | 10/10 (100%) |

The web search rate dropped to zero. Every query found at least one relevant corpus
document, and the agent used it instead of supplementing with web noise.

The faithfulness improvement (60% → 70%) is partly attributable to higher-quality
context (corpus docs vs web results) and partly to the larger sample size stabilising
the estimate.

!!! note "Why this matters architecturally"
    The improvement illustrates a general principle in agentic systems: **routing
    decisions should be calibrated to the reliability of the underlying signal**.
    The LLM judge is good at finding relevant documents; it reliably identifies at
    least one relevant doc when one exists in the top-10. Routing away from it based
    on quantity (requiring two matches instead of one) discarded good signal and
    replaced it with noisy web results.

---

## When the 3-tier approach would be appropriate

The 3-tier logic makes sense in a scenario where the corpus is genuinely sparse and
the web is genuinely better. If the query set includes many questions that the local
corpus cannot answer well, having a lower threshold for web fallback improves overall
answer quality.

For our corpus of 600 ML/AI papers and evaluation queries focused on ML topics, the
corpus is almost always sufficient. The 2-tier threshold is right for this deployment.

---

## What's next

[Evaluation](evaluation.md) covers the full 10-query evaluation setup — the
`run_agent()` function, the `agent_results` structure, and why the metrics in Part 3
measure agent behaviour rather than retrieval rank.
