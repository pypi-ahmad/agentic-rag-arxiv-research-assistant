# Claims Traceability

Use this page to verify that documented performance claims map to committed artifacts.

All paths below are relative to repository root.

---

## Claims Registry

| Claim ID | Claim | Artifact | Check values |
|---|---|---|---|
| C001 | NB01 (4,000) dense baseline metrics | `artifacts/eval_results/01_naive_rag_4000.json` | `retrieval_metrics.recall@5=0.05; retrieval_metrics.precision@5=0.01; retrieval_metrics.mrr=0.0167` |
| C002 | NB02 (4,000) advanced baseline metrics | `artifacts/eval_results/02_advanced_rag_4000.json` | `retrieval_metrics.recall@5=0.05; retrieval_metrics.precision@5=0.01; retrieval_metrics.mrr=0.0167` |
| C003 | NB03 (4,000) CRAG judge rates | `artifacts/eval_results/03_agentic_rag_4000.json` | `retrieval_metrics.recall@5=0.0; retrieval_metrics.mrr=0.0; generation_metrics.faithfulness_rate=0.8; generation_metrics.web_search_rate=0.8` |
| C004 | NB05 hybrid retrieval latency snapshot | `artifacts/rag_v2/hybrid/05_hybrid_metrics.json` | `retrieval.Hybrid.latency_p50_ms=166.96; retrieval.Hybrid.latency_p95_ms=175.3` |
| C005 | NB06 graph construction summary | `artifacts/rag_v2/graphrag/06_graphrag_metrics.json` | `graph.paper_nodes=4000; graph.entity_nodes=78; graph.edges=2644` |
| C006 | NB07 agentic quality summary | `artifacts/rag_v2/agentic/07_agentic_metrics.json` | `quality.faithfulness_mean=1.0; quality.llm_evaluated_rows=1` |
| C007 | NB08 CRAG rewrite summary | `artifacts/rag_v2/crag/08_crag_metrics.json` | `summary.rewrite_rate=0.8333; summary.rewrite_improvement_rate=0.8333; summary.n_questions=6` |
| C008 | NB09 multimodal summary | `artifacts/rag_v2/multimodal/09_multimodal_metrics.json` | `summary.n_pdfs=2; summary.ocr_avg_chars=2776.0; summary.faithfulness_mean=1.0` |
| C009 | Legacy 600 dense baseline metrics | `artifacts/eval_results/legacy_600/01_naive_rag_legacy_600.json` | `retrieval_metrics.recall@5=0.25; retrieval_metrics.precision@5=0.12; retrieval_metrics.mrr=0.2992` |
| C010 | Legacy 600 BM25 improved metrics | `artifacts/eval_results/legacy_600/02_advanced_rag_legacy_600.json` | `retrieval_metrics.recall@5=0.3667; retrieval_metrics.precision@5=0.19; retrieval_metrics.mrr=0.4408` |
| C011 | Judge model requirement for docs/tutorial path | `README.md` | `contains=granite4.1:8b` |
| C012 | Multimodal OCR model requirement for docs/tutorial path | `README.md` | `contains=glm-ocr` |

---

## How To Validate

Run:

```bash
uv run python scripts/check_docs_facts.py
```

The script verifies:

1. Artifact files exist.
2. Numeric checks match actual JSON values (with a small tolerance).
3. Deprecated model references are not present in docs.
4. Secret-like strings are not accidentally present in documentation.
