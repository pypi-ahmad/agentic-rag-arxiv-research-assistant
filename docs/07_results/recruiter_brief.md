# Recruiter Brief

This page is a fast technical summary of what this repository demonstrates.

---

## Project Scope

- Domain: Retrieval-Augmented Generation over ArXiv ML/AI literature
- Baselines preserved:
  - Current baseline: 4,000-paper corpus
  - Historical baseline: 600-paper corpus
- Implementations: 9 end-to-end notebooks covering naive, advanced, agentic, graph, CRAG, and multimodal RAG
- Runtime: local-first via Ollama, optional Pinecone backend for one GraphRAG variant

---

## What This Proves

| Capability | Evidence |
|---|---|
| Build and evaluate multiple RAG architectures | `notebooks/01_*` to `notebooks/09_*` + `artifacts/` metrics |
| Keep results reproducible and auditable | committed JSON metrics under `artifacts/eval_results/` and `artifacts/rag_v2/` |
| Compare strategies with explicit tradeoffs | `docs/07_results/benchmark_table.md` and `docs/07_results/reading_the_numbers.md` |
| Apply agentic control loops (judge, reroute, rewrite) | `notebooks/03_agentic_rag_langgraph.ipynb`, `notebooks/08_crag.ipynb` |
| Extend to multimodal evidence (PDF + OCR + vision) | `notebooks/09_multimodal_rag.ipynb` + multimodal artifacts |

---

## Key Design Decisions

1. Keep dual baselines instead of overwriting old numbers.
2. Prioritize artifact traceability over polished demo-only claims.
3. Use local LLM/embedding runtime for reproducibility and cost control.
4. Separate architecture variants into standalone notebooks for educational clarity.
5. Treat evaluation quality as part of system design, not an afterthought.

---

## Measured Highlights (Artifact-backed)

| Claim | Artifact |
|---|---|
| NB01 (4,000 baseline): Recall@5 = 0.05, Precision@5 = 0.01, MRR = 0.0167 | `artifacts/eval_results/01_naive_rag_4000.json` |
| Legacy 600 baseline: BM25 improved MRR = 0.4408 vs dense MRR = 0.2992 | `artifacts/eval_results/legacy_600/02_advanced_rag_legacy_600.json`, `artifacts/eval_results/legacy_600/01_naive_rag_legacy_600.json` |
| GraphRAG build (Part 5): 4,000 paper nodes, 78 entity nodes, 2,644 edges | `artifacts/rag_v2/graphrag/06_graphrag_metrics.json` |
| Multimodal run: 2 PDFs, OCR avg chars = 2776.0, faithfulness mean = 1.0 (evaluated rows) | `artifacts/rag_v2/multimodal/09_multimodal_metrics.json` |

For full claim-to-artifact mapping, see [Claims Traceability](../08_reference/claims_traceability.md).

---

## Practical Tradeoffs

| Choice | Benefit | Tradeoff |
|---|---|---|
| Local Ollama stack | low external dependency, reproducible demos | slower generation latency on commodity hardware |
| Hybrid retrieval + rerank | better rank quality on difficult queries | extra model calls and latency |
| Agentic control loops | better failure handling and transparency | additional orchestration complexity |
| Graph expansion | better relational retrieval opportunities | graph build and maintenance overhead |
| Multimodal OCR + vision | can use PDF/image evidence | higher pipeline complexity and latency |

---

## Reviewer Checklist

1. Open [Tutorial Index](../01_getting_started/tutorial_index.md) for the end-to-end path.
2. Inspect [Benchmark Table](benchmark_table.md) and [Reading the Numbers Honestly](reading_the_numbers.md).
3. Validate factual claims with [Claims Traceability](../08_reference/claims_traceability.md).
4. Optionally review the [Full Tutorial PDF](../01_getting_started/handbook_pdf.md).
