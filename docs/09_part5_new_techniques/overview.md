# Part 5 — New RAG Techniques (05-09)

Part 5 adds five new standalone implementations on top of the existing completed project:

1. Hybrid RAG
2. GraphRAG
3. Agentic RAG
4. Corrective RAG (CRAG)
5. Multimodal RAG

All implementations are isolated to new assets:

- `notebooks/05_hybrid_rag.ipynb` to `notebooks/09_multimodal_rag.ipynb`
- `src/rag_v2/*`
- `artifacts/rag_v2/*`
- `docs/09_part5_new_techniques/*`

The existing completed tutorial material remains unchanged.

---

## Execution Status (Real Runs)

All five notebooks were executed end-to-end and saved as executed notebooks:

- `notebooks/05_hybrid_rag.executed.ipynb`
- `notebooks/06_graphrag.executed.ipynb`
- `notebooks/07_agentic_rag.executed.ipynb`
- `notebooks/08_crag.executed.ipynb`
- `notebooks/09_multimodal_rag.executed.ipynb`

Run date for this final pass: **June 20, 2026**.

Judge model used in new notebooks: **`granite4.1:8b`**.

Multimodal paths used in Notebook 09:

- OCR path: **`ollama run glm-ocr`**
- Vision model path: **`qwen3.5:4b`**

---

## Real Result Snapshot

| Technique | Main Output Artifact | Key Real Metrics |
|---|---|---|
| Hybrid RAG | `artifacts/rag_v2/hybrid/05_hybrid_metrics.json` | Hybrid Recall@5 = `0.0000`, Hybrid latency P50/P95 = `166.96 / 175.30 ms`, generation faithfulness mean on sampled rows = `0.3333` |
| GraphRAG | `artifacts/rag_v2/graphrag/06_graphrag_metrics.json` | GraphRAG Recall@5 = `0.0000`, GraphRAG latency P50/P95 = `166.74 / 174.00 ms`, graph size = `4000 papers, 78 entities, 2644 edges` |
| Agentic RAG | `artifacts/rag_v2/agentic/07_agentic_metrics.json` | Routes: `{'hybrid': 6}`, faithfulness mean on evaluated rows = `1.0`, latency P50/P95 = `167.89 / 20352.27 ms` |
| CRAG | `artifacts/rag_v2/crag/08_crag_metrics.json` | Rewrite rate = `0.8333`, rewrite improvement rate = `0.8333`, faithfulness mean on evaluated rows = `0.0`, latency P50/P95 = `353.28 / 28985.80 ms` |
| Multimodal RAG | `artifacts/rag_v2/multimodal/09_multimodal_metrics.json` | PDFs processed = `2`, OCR avg chars = `2776.0`, vision avg chars = `0.0`, faithfulness mean on evaluated rows = `1.0`, latency P50/P95 = `22476.63 / 23847.47 ms` |

---

## How To Read These Numbers Correctly

Part 5 demonstrates architecture patterns and implementation quality, but the run also shows important evaluation caveats:

1. Retrieval metrics are flat (`0.0`) across Hybrid/GraphRAG on the weak keyword-supervision task.
2. Agentic and CRAG are intentionally run in a practical runtime-light mode for most rows, with sampled full LLM judging rows to keep runs tractable on local resources.
3. In this environment, the `qwen3.5:4b` multimodal path returned empty vision content, so Notebook 09 outcomes are OCR-dominant in measured output.

This is documented explicitly in each technique page and in notebook post-run analysis cells.

---

## Artifact Map

- Hybrid: `artifacts/rag_v2/hybrid/05_hybrid_metrics.json`
- GraphRAG: `artifacts/rag_v2/graphrag/06_graphrag_metrics.json`
- Agentic: `artifacts/rag_v2/agentic/07_agentic_metrics.json`
- CRAG: `artifacts/rag_v2/crag/08_crag_metrics.json`
- Multimodal: `artifacts/rag_v2/multimodal/09_multimodal_metrics.json`
- Multimodal PDFs: `artifacts/rag_v2/multimodal/pdfs/*`
- Multimodal rendered images: `artifacts/rag_v2/multimodal/images/*`

---

## Final Part 5 Summary

Part 5 is complete as a standalone extension: five new techniques, full notebooks, runnable code, executed outputs, and result-aware tutorial documentation. The strongest engineering outcome is architectural completeness and reproducibility; the strongest scientific takeaway is that evaluation design and runtime environment strongly shape observed RAG quality signals.
