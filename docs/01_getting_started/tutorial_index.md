# Tutorial Index

This is the canonical index for the end-to-end tutorial track.

Use this page to decide notebook order, dependencies, and where outputs are written.

For guided routes, see [Learning Tracks](learning_tracks.md).
For offline study, see [Full Tutorial PDF](handbook_pdf.md).

## Recommended Execution Order

1. `notebooks/01_naive_rag.ipynb`
2. `notebooks/02_advanced_rag.ipynb`
3. `notebooks/03_agentic_rag_langgraph.ipynb`
4. `notebooks/04_graph_rag.ipynb`
5. `notebooks/05_hybrid_rag.ipynb`
6. `notebooks/06_graphrag.ipynb`
7. `notebooks/07_agentic_rag.ipynb`
8. `notebooks/08_crag.ipynb`
9. `notebooks/09_multimodal_rag.ipynb`

## Notebook Matrix

| Notebook | Technique | Depends on | Primary outputs | Docs chapter |
|---|---|---|---|---|
| `01_naive_rag.ipynb` | Naive dense RAG | none | `artifacts/faiss_index/index.bin`, `artifacts/faiss_index/chunks.pkl`, `artifacts/eval_results/01_naive_rag_4000.json` | `docs/03_part1_naive_rag/` |
| `02_advanced_rag.ipynb` | BM25 + hybrid + reranking | FAISS/chunks from NB01 | `artifacts/eval_results/02_advanced_rag_4000.json`, retrieval plots in `artifacts/eval_results/` | `docs/04_part2_advanced_rag/` |
| `03_agentic_rag_langgraph.ipynb` | LangGraph CRAG (legacy track) | FAISS/chunks from NB01 | `artifacts/eval_results/03_agentic_rag_4000.json`, traces in `artifacts/agent_traces/` | `docs/05_part3_agentic_rag/` |
| `04_graph_rag.ipynb` | GraphRAG (ChromaDB, Pinecone, agentic graph) | corpus and runtime backends | runtime-generated `04_*_graphrag_4000.json` files under `artifacts/eval_results/` | `docs/06_part4_graph_rag/` |
| `05_hybrid_rag.ipynb` | Part 5 Hybrid RAG | prior project corpus/index assets | `artifacts/rag_v2/hybrid/05_hybrid_metrics.json` | `docs/09_part5_new_techniques/hybrid_rag.md` |
| `06_graphrag.ipynb` | Part 5 GraphRAG | prior project corpus/index assets | `artifacts/rag_v2/graphrag/06_graphrag_metrics.json` | `docs/09_part5_new_techniques/graphrag.md` |
| `07_agentic_rag.ipynb` | Part 5 Agentic RAG | prior project corpus/index assets | `artifacts/rag_v2/agentic/07_agentic_metrics.json` | `docs/09_part5_new_techniques/agentic_rag.md` |
| `08_crag.ipynb` | Part 5 Corrective RAG | prior project corpus/index assets | `artifacts/rag_v2/crag/08_crag_metrics.json` | `docs/09_part5_new_techniques/crag.md` |
| `09_multimodal_rag.ipynb` | Part 5 Multimodal RAG | local PDF/image/model runtime | `artifacts/rag_v2/multimodal/09_multimodal_metrics.json` plus PDF/image assets | `docs/09_part5_new_techniques/multimodal_rag.md` |

## Model Requirements by Path

### Core models

```bash
ollama pull qwen3-embedding:0.6b
ollama pull granite4.1:8b
```

### Optional/extended models

```bash
ollama pull qwen3-embedding:4b
ollama pull qwen3.5:4b
ollama pull glm-ocr
```

## Dual-Baseline Guidance

- Treat 4,000-paper outputs as the current baseline.
- Keep 600-paper outputs as historical reference for scale comparison.
- Do not compare metrics across differently scoped eval settings without noting the scope.

## Validation Commands

```bash
uv run python scripts/check_docs_integrity.py
uv run python scripts/check_docs_facts.py
uv run mkdocs build --strict
```
