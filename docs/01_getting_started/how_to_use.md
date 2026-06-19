# How to Use This Tutorial

You've installed everything. Now let's orient you: what order to run things in, why that order matters, and how the documentation pages connect to the notebook cells.

---

## Run the notebooks in order

This is the single most important rule in the project:

```
01_naive_rag.ipynb  →  02_advanced_rag.ipynb  →  03_agentic_rag_langgraph.ipynb
```

The reason is a concrete dependency: notebook 01 builds and saves the FAISS index to `artifacts/faiss_index/index.bin`. Notebooks 02 and 03 *load* that index at startup. If you open notebook 02 or 03 before running notebook 01, the very first cell that tries to load the index will raise:

```
FileNotFoundError: FAISS index not found at artifacts/faiss_index/index.bin
```

!!! warning "Always run notebook 01 first"
    If you see a `FileNotFoundError` about the FAISS index, the fix is always the same: open `01_naive_rag.ipynb`, select **Kernel → Restart & Run All**, wait for it to finish, then return to the notebook you were trying to run.

The dependency chain in full:

| Notebook | Depends on | Builds |
|----------|-----------|--------|
| `01_naive_rag.ipynb` | Nothing (downloads data from ArXiv API) | `artifacts/faiss_index/index.bin`, `artifacts/faiss_index/chunks.pkl`, `artifacts/eval_results/01_naive_rag.json` |
| `02_advanced_rag.ipynb` | `index.bin` and `chunks.pkl` from notebook 01 | `artifacts/eval_results/02_advanced_rag.json`, comparison plots |
| `03_agentic_rag_langgraph.ipynb` | `index.bin` and `chunks.pkl` from notebook 01 | `artifacts/agent_traces/eval_traces.json` |

Notebooks 02 and 03 are independent of each other — you can run them in either order once notebook 01 has completed.

---

## How each notebook is structured

Each notebook follows the same pattern:

1. **Setup and imports** — dependencies, constants, src module imports
2. **Load / build the component** — either loads the FAISS index (02, 03) or builds it from scratch (01)
3. **Demonstrate the core concept** — a few focused cells showing the thing working
4. **Evaluate** — runs the 20-query benchmark and prints the metrics table
5. **Improvements section** — old code is left in commented cells alongside the improved version, with an explanation of what changed and why

!!! tip "Read the improvement comments"
    The most educational parts of the notebooks are the improvement cells — the ones where old code sits commented out next to the new code. These explain the *decision* behind each change, not just the change itself. Don't skip them.

---

## How the docs relate to the notebooks

Think of it this way:

> **The docs explain the *why*. The notebooks are the runnable *what*.**

Every major concept in these docs maps to a specific cell or section in a notebook. The docs give you the intuition and the mental model *before* you encounter the code. The notebooks let you run the code and see the numbers yourself.

A practical reading pattern:

```
Read the docs page for a concept
    → Open the corresponding notebook section
        → Run the cells
            → Come back to the docs if something is unclear
```

For example: before running the BM25 cells in notebook 02, read [BM25 — Keyword Retrieval](../02_core_concepts/bm25_sparse_retrieval.md). After running the hybrid retrieval cells and seeing the metrics, read [Hybrid Search](../02_core_concepts/hybrid_search.md) to understand why alpha-blending sometimes doesn't beat BM25 alone.

You don't have to read every docs page before opening a notebook — but the docs are there when a concept feels thin or the code doesn't click.

---

## Navigating the docs site

The docs are organised in parallel with the notebooks:

| Nav section | What it covers |
|-------------|---------------|
| **Getting Started** (this section) | Setup, installation, orientation |
| **Core Concepts** | One page per concept: embeddings, BM25, FAISS, hybrid search, reranking, evaluation metrics, LangGraph |
| **Part 1 — Naive RAG** | Mirrors notebook 01 section-by-section |
| **Part 2 — Advanced RAG** | Mirrors notebook 02 section-by-section |
| **Part 3 — Agentic RAG** | Mirrors notebook 03 section-by-section |
| **Results & Improvements** | Full benchmark table, all 5 improvements explained, honest analysis |
| **Reference** | Glossary, API docs for `src/` modules, troubleshooting, further reading |

The **Core Concepts** section is standalone — you can read it at any point without the notebooks open. It's useful if you want to understand a concept deeply before encountering it in code, or if you want to revisit it after running a cell that confused you.

The **Reference** section is the place to go when something breaks. The [Troubleshooting](../08_reference/troubleshooting.md) page covers the most common errors with exact fixes.

---

## Practical tips for working through the tutorial

**Run cells one at a time on first pass.** Don't hit "Run All" until you've read through the notebook once. The cells that take a long time (indexing ~2100 chunks, running the 20-query evaluation) are clearly marked — you'll want to know when to expect a wait.

**The indexing step in notebook 01 takes about 2 minutes.** Embedding ~2100 chunks with `qwen3-embedding:0.6b` on an RTX 4060 takes roughly 2 minutes. On CPU, expect 15–20 minutes. This only runs once — after the first run, notebook 01 loads the saved index on subsequent runs.

**Ollama models load in ~3–5 seconds.** The first call to any Ollama model in a fresh session will be slower than subsequent calls because the model is loading from disk into VRAM. Don't be alarmed by a slow first generation — the second call will be much faster.

**Keep a terminal open alongside the notebook.** If Ollama produces errors or if you need to check which models are loaded (`ollama ps`), a terminal is faster than switching windows.

!!! info "Expected notebook run times"
    These are approximate times on an RTX 4060 Laptop with models already pulled:

    | Notebook | First run (builds index) | Subsequent runs (loads index) |
    |----------|------------------------|-------------------------------|
    | 01 | ~10–12 minutes | ~3–4 minutes |
    | 02 | ~5–6 minutes | ~5–6 minutes |
    | 03 | ~8–10 minutes (10-query eval) | ~8–10 minutes |

---

## Where to go from here

You're oriented. Now pick your path:

- **Read the concepts first** → Start with [What Is RAG](../02_core_concepts/what_is_rag.md) in the Core Concepts section
- **Jump straight into the code** → Open `notebooks/01_naive_rag.ipynb` in Jupyter with the "Agentic RAG" kernel selected
- **Something already broke** → Check [Troubleshooting](../08_reference/troubleshooting.md)
