# Agentic RAG ArXiv Research Assistant

Production-style RAG tutorial and reference implementation that evolves from a naive baseline to advanced retrieval, agentic correction, graph expansion, and multimodal evidence.

The repository keeps **dual baselines**:
- **Current baseline:** 4,000-paper ML/AI corpus (default reference path)
- **Legacy baseline:** 600-paper results (historical comparison)

## What You Build

1. Naive RAG (`notebooks/01_naive_rag.ipynb`)
2. Advanced RAG (`notebooks/02_advanced_rag.ipynb`)
3. Agentic RAG with LangGraph (`notebooks/03_agentic_rag_langgraph.ipynb`)
4. GraphRAG variants (`notebooks/04_graph_rag.ipynb`)
5. Hybrid RAG (`notebooks/05_hybrid_rag.ipynb`)
6. GraphRAG (Part 5 implementation) (`notebooks/06_graphrag.ipynb`)
7. Agentic RAG (Part 5 implementation) (`notebooks/07_agentic_rag.ipynb`)
8. Corrective RAG / CRAG (`notebooks/08_crag.ipynb`)
9. Multimodal RAG (`notebooks/09_multimodal_rag.ipynb`)

Executed notebooks are available as `notebooks/*.executed.ipynb`.

## Quick Start

```bash
git clone https://github.com/pypi-ahmad/agentic-rag-arxiv-research-assistant.git
cd agentic-rag-arxiv-research-assistant

uv python install 3.13.13
uv venv --python 3.13.13
source .venv/bin/activate
uv pip install -r requirements.txt

# Core models
ollama pull qwen3-embedding:0.6b
ollama pull granite4.1:8b

# Optional models used in extended notebooks
ollama pull qwen3-embedding:4b
ollama pull qwen3.5:4b
ollama pull glm-ocr
```

## Tutorial Index

See the full tutorial map, dependencies, and output artifacts here:
- [`docs/01_getting_started/tutorial_index.md`](docs/01_getting_started/tutorial_index.md)

## Benchmarks (Current Snapshot)

### Current 4,000 baseline (committed artifacts)

| System | Artifact | Recall@5 | Precision@5 | MRR | Notes |
|---|---|---:|---:|---:|---|
| Naive Dense (NB01) | `artifacts/eval_results/01_naive_rag_4000.json` | 0.0500 | 0.0100 | 0.0167 | Dense FAISS baseline |
| Advanced (NB02) | `artifacts/eval_results/02_advanced_rag_4000.json` | 0.0500 | 0.0100 | 0.0167 | Hybrid + reranker configuration |
| Agentic CRAG (NB03) | `artifacts/eval_results/03_agentic_rag_4000.json` | 0.0000 | n/a | 0.0000 | Faithfulness rate 0.8, web-search rate 0.8 |
| Hybrid RAG (NB05) | `artifacts/rag_v2/hybrid/05_hybrid_metrics.json` | 0.0000 | 0.0000 | 0.0000 | P50/P95 retrieval latency: 166.96 / 175.30 ms |
| GraphRAG (NB06) | `artifacts/rag_v2/graphrag/06_graphrag_metrics.json` | 0.0000 | 0.0000 | 0.0000 | Graph: 4,000 papers, 78 entities, 2,644 edges |
| Agentic RAG (NB07) | `artifacts/rag_v2/agentic/07_agentic_metrics.json` | n/a | n/a | n/a | Route stats + faithfulness traces |
| CRAG (NB08) | `artifacts/rag_v2/crag/08_crag_metrics.json` | n/a | n/a | n/a | Rewrite/correction metrics |
| Multimodal RAG (NB09) | `artifacts/rag_v2/multimodal/09_multimodal_metrics.json` | n/a | n/a | n/a | OCR + vision pipeline metrics |

### Legacy 600 baseline (historical reference)

| System | Artifact | Recall@5 | Precision@5 | MRR |
|---|---|---:|---:|---:|
| Naive Dense | `artifacts/eval_results/legacy_600/01_naive_rag_legacy_600.json` | 0.2500 | 0.1200 | 0.2992 |
| BM25 Improved | `artifacts/eval_results/legacy_600/02_advanced_rag_legacy_600.json` | 0.3667 | 0.1900 | 0.4408 |

## Artifact Map

- Baseline eval JSONs: `artifacts/eval_results/`
- Legacy eval JSONs: `artifacts/eval_results/legacy_600/`
- Part 5 technique metrics: `artifacts/rag_v2/`
- FAISS index and chunks: `artifacts/faiss_index/`
- Graph build artifacts: `artifacts/graph/`

Note on Part 4 GraphRAG result files:
- `notebooks/04_graph_rag.ipynb` writes `04_*_graphrag_4000.json` during execution.
- Those specific `04_*` JSON files are runtime-generated and may not be committed in every snapshot.

## Runtime and Credentials

- Fully local path works for Parts 1-3 and Part 5 notebooks with Ollama.
- Pinecone-backed GraphRAG in Part 4 is optional and requires valid Pinecone credentials.

## Documentation

- Docs home: [`docs/index.md`](docs/index.md)
- MkDocs config: [`mkdocs.yml`](mkdocs.yml)
- New techniques chapter: [`docs/09_part5_new_techniques/overview.md`](docs/09_part5_new_techniques/overview.md)

### Documentation QA

```bash
python scripts/check_docs_integrity.py
.venv/bin/mkdocs build --strict
```
