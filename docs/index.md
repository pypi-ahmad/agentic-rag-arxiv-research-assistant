# Agentic RAG — Zero to Hero

**Build a fully local, production-grade RAG research assistant over 600 ArXiv ML/AI papers — from a blank Python file to a self-correcting LangGraph agent.**

No cloud APIs. No API keys. No paid services. Everything runs on your machine.

---

## What is this project?

This tutorial teaches you how to build a **Retrieval-Augmented Generation (RAG)** system — the pattern behind every serious LLM application deployed in production today. You will build three progressively more powerful systems, measure each one, understand why it fails, and fix it.

The corpus is 600 recent ArXiv machine learning and AI paper abstracts. Ask it questions like:

- *"What is retrieval-augmented generation and how does it work?"*
- *"How does LoRA reduce memory consumption during fine-tuning?"*
- *"What are the key differences between BM25 and dense retrieval?"*

By the end, you will have a working Q&A system backed by a LangGraph state machine that grades its own retrieved documents, detects hallucinations in its own answers, and falls back to web search when the corpus can't help.

---

## The learning journey

```mermaid
graph LR
    A[Part 1<br/>Naive RAG] -->|measure the gap| B[Part 2<br/>Advanced RAG]
    B -->|add intelligence| C[Part 3<br/>Agentic RAG]

    A:::done
    B:::done
    C:::done

    classDef done fill:#4051b5,stroke:#303f9f,color:#fff
```

| Part | Notebook | What you build | Key metric |
|------|----------|----------------|------------|
| **1 — Naive RAG** | `01_naive_rag.ipynb` | FAISS semantic search + granite4.1:8b generation | MRR = 0.299 (baseline) |
| **2 — Advanced RAG** | `02_advanced_rag.ipynb` | BM25 + hybrid retrieval + cross-encoder reranking | MRR = 0.441 (+47% over baseline) |
| **3 — Agentic RAG** | `03_agentic_rag_langgraph.ipynb` | LangGraph CRAG state machine with self-correction | 7/10 faithful answers, 0/10 web fallbacks |

Each part builds on the previous one. The FAISS index you build in Part 1 is loaded by Parts 2 and 3. The hybrid retriever from Part 2 is the retrieval backbone for the LangGraph agent.

---

## Quick start

If you want to jump straight in, here are the three commands to go from a fresh clone to a running notebook environment:

```bash
# 1 — Clone and enter the repo
git clone https://github.com/pypi-ahmad/agentic-rag-arxiv-research-assistant.git
cd agentic-rag-arxiv-research-assistant

# 2 — Set up the Python environment (requires uv)
uv venv --python 3.13 && source .venv/bin/activate && uv pip install -r requirements.txt

# 3 — Register the Jupyter kernel and open notebooks
.venv/bin/python -m ipykernel install --user --name agentic-rag-env --display-name "Agentic RAG"
jupyter notebook
```

!!! warning "Pull the models first"
    The notebooks call Ollama for embeddings and LLM generation. Before running any notebook, make sure you have pulled the models:

    ```bash
    ollama pull qwen3-embedding:0.6b
    ollama pull granite4.1:8b
    ```

    If Ollama is not installed yet, see the [Prerequisites](01_getting_started/prerequisites.md) page.

---

## What each part builds

### Part 1 — Naive RAG (Notebook 01)

You download 600 ArXiv papers, split them into overlapping chunks, embed them with `qwen3-embedding:0.6b`, store the vectors in a FAISS index, and wire up a simple RAG chain using `granite4.1:8b` for generation. Then you measure it honestly with a 20-query evaluation set.

**New concepts:** embeddings, cosine similarity, FAISS, chunking, RAG prompt design, Recall@k, Precision@k, MRR.

### Part 2 — Advanced RAG (Notebook 02)

You add a BM25 keyword retriever alongside the dense FAISS retriever, combine their scores (hybrid search), and add a cross-encoder reranker as a second-pass filter. You measure each component in isolation to understand which one actually moved the needle.

**New concepts:** BM25 scoring, hybrid retrieval, alpha-weighted score fusion, RRF, bi-encoder vs. cross-encoder, two-stage retrieve-then-rerank.

### Part 3 — Agentic RAG (Notebook 03)

You implement the CRAG architecture (Corrective RAG, Yan et al. 2024) as a LangGraph state machine. The agent grades retrieved documents for relevance, falls back to DuckDuckGo web search when the corpus can't help, generates an answer, then grades its own answer for hallucinations — retrying up to twice if it detects a problem.

**New concepts:** state machines, LangGraph TypedDict state, conditional edges, LLM-as-judge, CRAG architecture, web search tool.

---

## How the docs relate to the notebooks

!!! info "Docs explain the *why*. Notebooks are the runnable code."
    Every concept in these docs has a corresponding cell in a notebook. The docs give you the intuition and the mental model; the notebooks let you run the code and see the numbers yourself.

    If something in a notebook is unclear, come back to the docs for the deeper explanation. If a docs page feels abstract, open the corresponding notebook cell and run it.

---

## Who this is for

This tutorial is for you if:

- You are a Python developer or ML practitioner who has heard about RAG but never built one from scratch
- You understand what a neural network does at a high level, but don't need a PhD-level treatment
- You want to see real, measured improvements — not just "here's some code that might work"
- You want everything to run locally, without paying for API calls or setting up cloud infrastructure

You do **not** need prior experience with RAG, LangChain, LangGraph, FAISS, or BM25. Everything is explained from scratch.

---

## Ready to begin?

Start with [What Is This Project](01_getting_started/what_is_this_project.md) for the full context and motivation, or jump straight to [Prerequisites](01_getting_started/prerequisites.md) if you're already sold on the idea.
