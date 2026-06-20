# How to Use This Tutorial

This tutorial is notebook-first and dependency-aware.

## Rule 1: Follow the execution order

Run notebooks in this order:

```text
01_naive_rag.ipynb
→ 02_advanced_rag.ipynb
→ 03_agentic_rag_langgraph.ipynb
→ 04_graph_rag.ipynb
→ 05_hybrid_rag.ipynb
→ 06_graphrag.ipynb
→ 07_agentic_rag.ipynb
→ 08_crag.ipynb
→ 09_multimodal_rag.ipynb
```

## Rule 2: Understand hard dependencies

- Notebook 01 builds shared FAISS/chunk artifacts:
  - `artifacts/faiss_index/index.bin`
  - `artifacts/faiss_index/chunks.pkl`
- Notebooks 02 and 03 load these files at startup.
- Part 5 notebooks (`05`-`09`) are isolated technique implementations under `src/rag_v2/` and `artifacts/rag_v2/`.

If notebook 02 or 03 fails with FAISS file-not-found, rerun notebook 01 fully.

## Notebook map

| Notebook | What it does | Main outputs |
|---|---|---|
| `01_naive_rag.ipynb` | Build baseline dense RAG index and eval | `artifacts/faiss_index/*`, `artifacts/eval_results/01_naive_rag_4000.json` |
| `02_advanced_rag.ipynb` | BM25/hybrid/reranking evaluation | `artifacts/eval_results/02_advanced_rag_4000.json` |
| `03_agentic_rag_langgraph.ipynb` | LangGraph CRAG evaluation | `artifacts/eval_results/03_agentic_rag_4000.json`, traces |
| `04_graph_rag.ipynb` | GraphRAG variant evaluations | runtime-generated `04_*_graphrag_4000.json` |
| `05_hybrid_rag.ipynb` | Part 5 Hybrid RAG implementation | `artifacts/rag_v2/hybrid/05_hybrid_metrics.json` |
| `06_graphrag.ipynb` | Part 5 GraphRAG implementation | `artifacts/rag_v2/graphrag/06_graphrag_metrics.json` |
| `07_agentic_rag.ipynb` | Part 5 Agentic RAG implementation | `artifacts/rag_v2/agentic/07_agentic_metrics.json` |
| `08_crag.ipynb` | Part 5 Corrective RAG implementation | `artifacts/rag_v2/crag/08_crag_metrics.json` |
| `09_multimodal_rag.ipynb` | Part 5 Multimodal RAG implementation | `artifacts/rag_v2/multimodal/09_multimodal_metrics.json` |

## How docs and notebooks connect

- Docs explain design choices, failure modes, and interpretation.
- Notebooks are runnable execution artifacts.
- Use docs pages to understand why a metric or behavior changed, then verify in notebook outputs.

For full navigation across all tracks, see [Tutorial Index](tutorial_index.md).
