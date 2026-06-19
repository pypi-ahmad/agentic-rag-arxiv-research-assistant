# Results and Limitations

This page presents the full Part 3 results, compares agent behaviour before and after
Improvement 5, and gives an honest account of what CRAG adds over Part 2 — and where
the approach still falls short.

---

## Final results: 10-query evaluation

| Metric | Value |
|---|---|
| Queries evaluated | 10 |
| Relevant corpus docs found | 10 / 10 (100%) |
| Web search triggered | 0 / 10 (0%) |
| Faithful answers | 7 / 10 (70%) |
| Average latency | ~15 s / query |

The 100% relevant retrieval rate reflects a well-matched corpus: 600 ML/AI papers
covering the topics in the evaluation queries. The 0% web search rate is a consequence
of both corpus quality and the 2-tier threshold — the agent trusts a single relevant
document rather than demanding two.

The 70% faithfulness rate is the headline quality metric. Three queries out of ten
produced answers that the hallucination judge flagged as containing unsupported claims.
In two of those cases the retry brought the faithfulness grade to `"faithful"` on the
second attempt; in one case both attempts were flagged and the second attempt's answer
was returned as-is (the 2-attempt cap).

---

## Before vs after Improvement 5

| Metric | Before (3-tier, 5 queries) | After (2-tier, 10 queries) |
|---|---|---|
| Web search triggered | 1/5 (20%) | 0/10 (0%) |
| Faithful answers | 3/5 (60%) | 7/10 (70%) |
| Sample size | 5 | 10 |

The improvement in faithfulness (60% → 70%) is partly real and partly an artefact of
the larger sample. What is clearly real is the elimination of unnecessary web search
calls. The 3-tier scheme sent one query to the web despite the corpus having a relevant
document for it; the 2-tier scheme does not.

---

## What CRAG adds over Part 2

Part 2 retrieves and ranks better. Part 3 acts differently when retrieval is uncertain.
These are not competing improvements — they operate at different levels of the stack.

| Capability | Part 2 (Advanced RAG) | Part 3 (CRAG Agent) |
|---|---|---|
| Retrieval quality | Hybrid BM25 + dense + reranking | Dense only (top-10 from DenseRetriever) |
| Retrieval failure handling | None — bad results go to LLM | Grade and gate: bad results trigger fallback |
| Answer quality signal | None | Faithfulness check with retry |
| Web knowledge access | None | DuckDuckGo fallback when corpus fails |
| Execution traceability | Pipeline steps only | Full execution_trace with reasoning |
| Latency | ~200 ms | ~15 s |

The most important addition is the feedback loop. Part 2 is a straight line from query
to answer. Part 3 is a cycle that can catch and correct its own mistakes. This is
qualitatively different, not just quantitatively better.

A Part 2 system that retrieves the wrong documents produces a confidently wrong answer
with no indication that anything went wrong. A Part 3 system detects the retrieval
failure (via the judge), either supplements with web results or routes to the
available evidence, and then checks its own output before returning it.

---

## Remaining limitations

**Latency.** Fifteen seconds per query is not suitable for interactive use cases. The
bottleneck is the per-document grading loop: 10 LLM calls, each taking ~1 second,
running sequentially. The obvious fix is parallelisation — run all 10 judge calls
concurrently with `asyncio.gather`. This is not implemented in the tutorial to keep
the code synchronous and easy to follow.

**Single-hop reasoning only.** The current architecture handles one retrieval step.
A question like *"How does the method in the LoRA paper compare to the approach used
in the Chinchilla scaling paper?"* requires retrieving from two different papers and
synthesising across them. The current graph retrieves once, grades once, and generates
once. Multi-hop reasoning requires a different graph architecture — typically a loop
that issues multiple retrieval queries with different reformulations.

**Judge quality ceiling.** Both the relevance and faithfulness grades come from the
same granite4.1:8b model that generates the answers. This creates a potential
consistency bias: the judge may be more lenient on its own generation style. A better
setup uses a separate, independently trained judge model. The 70% faithfulness rate
is a lower bound on actual answer quality (the judge may be missing some hallucinations)
and also an upper bound (the judge may be flagging answers that are technically
faithful but worded differently from the source).

**Corpus coverage gaps.** The 600-paper corpus covers broad ML/AI topics well but will
fail on specialised subfields, very recent papers, and non-ML computer science topics.
The web fallback compensates for some of this but introduces content quality risks.

**No memory across queries.** Each query is independent. If a user asks a follow-up
question, the agent has no access to the previous answer or retrieved documents. A
production system would maintain a conversation state and allow multi-turn refinement.

---

## What comes next: Part 4 — GraphRAG

The most significant structural limitation is single-hop reasoning. GraphRAG addresses
this by treating the corpus as a knowledge graph rather than a bag of chunks. Entities
(papers, authors, concepts) are nodes; relationships (cites, proposes, evaluates on)
are edges. A query that requires multi-hop reasoning can traverse the graph: find the
LoRA paper node, follow the "proposes" edge to the LoRA method node, follow the
"compared-to" edge to related method nodes, and retrieve context from all of them in
a single structured traversal.

This is the architecture that handles *"How have fine-tuning efficiency methods evolved
since 2020?"* — a question that requires understanding relationships across dozens of
papers, not just retrieving the most similar chunk.

!!! tip "Key takeaway from Part 3"
    The CRAG agent does not retrieve better than Part 2 — it retrieves the same
    documents using the same dense retriever. What it adds is the ability to detect
    when retrieval has failed and take a different action before generating. The
    self-correction loop catches hallucinations before they reach the user. These two
    properties — failure detection and self-correction — are what distinguish an agent
    from a pipeline, and they are the foundation everything in Part 4 builds on.
