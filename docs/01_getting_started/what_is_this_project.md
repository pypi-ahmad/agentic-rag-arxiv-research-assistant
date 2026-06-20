# What Is This Project?

Before running a single cell, it's worth understanding *why* this project exists and what problem it actually solves. This page gives you the full picture: the problem with LLMs, why RAG is the standard fix, why this particular corpus was chosen, and what makes this tutorial different from the dozens of others out there.

---

## The problem: LLMs are confident, but frozen and fallible

Large language models are impressive. Ask `granite4.1:8b` about transformer attention mechanisms, and it will give you a fluent, detailed answer. But that answer has two fundamental problems.

**Problem 1 — The knowledge cutoff.** Every LLM was trained on data up to a certain date. Anything published after that date — a new paper, a new model, a new technique — simply doesn't exist in the model's world. For a field like machine learning, where the landscape changes every few months, a frozen knowledge base is a serious limitation.

**Problem 2 — Hallucination.** LLMs generate the most *statistically probable* next token given their training data. When a question touches something they don't know well, they don't say "I don't know" — they generate a plausible-sounding answer anyway. This is called *hallucination*, and it's not a bug that will be patched — it's a structural feature of how autoregressive language models work.

!!! warning "Why hallucination matters for research"
    If you're using an LLM to answer questions about recent ML papers, the model will sometimes fabricate paper titles, invent author names, or confidently describe results that don't exist. This is dangerous precisely *because* the answer sounds authoritative.

---

## The solution: RAG grounds the answer in real documents

**Retrieval-Augmented Generation (RAG)** is the standard architectural pattern for fixing both problems at once. The core idea is simple:

> Before the LLM generates an answer, *retrieve* the relevant documents from a knowledge base and *inject* them into the prompt. Instruct the model to answer only from those documents.

When you do this, the model's answer is *grounded* — every claim it makes is constrained to the content you gave it. If the answer tries to go beyond the documents, you can detect that (which Part 3 of this tutorial does automatically).

A RAG pipeline has two phases:

1. **Indexing (offline):** Load your documents, split them into chunks, embed each chunk as a vector, store the vectors in a searchable index.
2. **Querying (online):** When a question arrives, embed the question, search the index for the most similar chunks, inject those chunks into the LLM prompt, generate an answer.

This tutorial builds both phases, from scratch, step by step.

---

## Why ArXiv ML/AI papers?

The current baseline corpus is 4,000 ArXiv ML/AI paper abstracts from HuggingFace
(`ccdv/arxiv-summarization`, ML-filtered). Historical 600-paper runs are preserved as
legacy comparison artifacts.

This is an unusual choice for a tutorial — most use Wikipedia articles or random web pages. Here's why ArXiv ML papers are better for learning RAG:

**Self-referential.** We're building a RAG system *about* RAG systems, transformers, and fine-tuning. The system can answer questions about itself. This makes it easy to spot when retrieval is working (you'll recognise the papers it finds) and when it's failing (you'll notice when it retrieves something irrelevant).

**Verifiable.** ArXiv papers have titles, abstracts, and well-defined claims. A good answer is clearly good; a bad answer is clearly bad. This makes evaluation honest rather than subjective.

**Naturally multi-topic.** A question like "how does LoRA relate to quantization?" genuinely requires finding information across multiple papers. This exposes the real challenges of retrieval — not just "find the paper about X" but "find all the papers that together answer the question."

**Terminology-dense.** ML papers are full of exact technical terms — model names, method names, benchmark names. This is the ideal stress test for the hybrid retrieval system in Part 2, where keyword-based BM25 search handles exact terms that semantic search sometimes misses.

---

## The progression: Naive → Advanced → Agentic → GraphRAG → New Techniques

This tutorial is structured as five progressively more capable tracks.
Each one fixes the failures of the previous one.

### Part 1 — Naive RAG: The baseline

The simplest possible RAG pipeline: embed everything with one model, search with FAISS, generate with the LLM. It works, but the retrieval quality is poor. When measured honestly against 20 diverse queries, it achieves an MRR of 0.299 — meaning the most relevant paper appears at rank 3–4 on average.

You'll see exactly *why* it fails: semantic search alone misses queries that use exact technical terminology, and there's no way to recover once the wrong documents are retrieved.

### Part 2 — Advanced RAG: Better retrieval

Part 2 adds three improvements to the retrieval layer:

- **BM25 keyword search** — catches exact technical terms that semantic search misses
- **Hybrid retrieval** — combines BM25 and dense scores into a single ranked list
- **Cross-encoder reranking** — re-scores the top candidates with a more accurate model

The result: MRR jumps to 0.441 — a 47% improvement over the naive baseline, with the same LLM and the same corpus. The lesson is that *retrieval quality is the biggest lever in a RAG system*, not the LLM.

### Part 3 — Agentic RAG: Self-correction

Part 3 adds intelligence on top of the Advanced RAG retriever. Instead of blindly generating from whatever was retrieved, the system:

1. **Grades the retrieved documents** — asks the LLM "are these relevant?"
2. **Falls back to web search** — if no relevant documents are found in the corpus
3. **Grades its own answer** — checks whether the answer contains only claims from the retrieved documents
4. **Retries** — if a hallucination is detected, regenerates the answer (up to twice)

This is the CRAG architecture (Corrective RAG, Yan et al. 2024), implemented as a LangGraph state machine with conditional routing. Legacy 10-query runs showed strong local-corpus coverage, while current 4,000-baseline artifacts also expose the tradeoff between faithfulness and fallback behavior under harder queries.

### Part 4 — GraphRAG: Structured retrieval at scale

Part 4 adds entity extraction, knowledge-graph construction, community detection, and
dual backend execution (ChromaDB + Pinecone), then wraps GraphRAG in an agentic LangGraph
loop. This introduces multi-hop and community-level retrieval strategies that are harder
to express in a pure chunk-similarity pipeline.

### Part 5 — New techniques: focused standalone implementations

Part 5 adds five additional implementations as isolated, production-style modules and notebooks:

- Hybrid RAG
- GraphRAG
- Agentic RAG
- Corrective RAG (CRAG)
- Multimodal RAG (including OCR and vision branches)

These runs are tracked under `artifacts/rag_v2/` and documented in
`docs/09_part5_new_techniques/`.

---

## What makes this tutorial different?

Most RAG tutorials show you *how to build* the pipeline but not *how well it works*. They use 5 hand-picked test questions and don't measure anything. They don't explain why certain design choices were made, or what happens when something fails.

This tutorial does things differently:

- **Every step is measured.** Retrieval quality is evaluated with benchmarked metrics and
  persisted JSON artifacts at both legacy 600 and current 4,000 scale.
- **The eval is honest.** Legacy 600 numbers are kept visible, and current 4,000 numbers
  are reported side-by-side so readers can compare scaling effects directly.
- **Failures are explained.** When hybrid search with alpha-blending doesn't beat BM25 alone, the tutorial explains why rather than hiding the result.
- **Local-first with optional cloud backend.** The core path runs locally with Ollama. Part 4 also includes an optional Pinecone backend that requires valid credentials.

---

Ready to check whether your machine is set up? Continue to [Prerequisites](prerequisites.md).
