# 📘 MASTER LEARNING HANDBOOK: Agentic RAG ArXiv Research Assistant

**Edition**: Static-analysis edition (no project code execution)

**Evidence baseline used**
- Source modules: `src/*.py`, `src/rag_v2/*.py`
- Operational scripts: `scripts/*.py`
- Tutorial notebooks: `notebooks/01_naive_rag.ipynb` through `notebooks/09_multimodal_rag.ipynb` (not executed in this handbook)
- Documentation + config: `docs/**/*.md`, `mkdocs.yml`, `README.md`, `requirements.txt`
- Artifacts reviewed for real output schemas: `artifacts/**/*.json`, plus generated/static outputs under `site/`

**Scope lock (from planning decisions)**
- Included in mapping: all git-tracked files + generated `artifacts/` + generated `site/` files/directories.
- Excluded from mapping: environment internals (`.venv/`, `.uv-cache/`).

---

## 🌐 Module 1: Theoretical Foundations & Architecture

### 1.1 CS Theory & Definitions (Before Implementation)

**Object-Oriented Programming (OOP)**
- Definition: A paradigm where code is organized around objects/classes that encapsulate state and behavior.
- In this repository:
  - Retriever/store abstractions are class-based: `DenseRetriever`, `BM25Retriever`, `HybridRetriever`, `Reranker` (`src/retriever.py`), plus `VectorStore`/`ChromaVectorStore`/`PineconeVectorStore` (`src/vectorstore.py`).
  - Data contracts are modeled with dataclasses: `EvalResults` (`src/evaluator.py`), `RetrievalBundle` (`src/rag_v2/retrieval.py`), `GenerationEvalRow` (`src/rag_v2/metrics.py`), `AgenticResult` (`src/rag_v2/agentic.py`).

**Functional Programming (Pure-ish Transform Pipelines)**
- Definition: A paradigm emphasizing composable functions and data transformations, minimizing hidden mutable state.
- In this repository:
  - Ingestion and retrieval are composed as function pipelines: `load_* -> chunk_documents -> embed_texts -> build_faiss_index` (`src/ingest.py`).
  - Metrics modules expose mostly stateless functions (`recall_at_k`, `precision_at_k`, `mrr`, `ndcg_at_k`) over explicit inputs (`src/evaluator.py`, `src/rag_v2/metrics.py`).

**Event-Driven / Conditional-Route Orchestration**
- Definition: Control flow advances via events/conditions (for example, quality grades) rather than fixed linear execution.
- In this repository:
  - NB03 LangGraph routing functions `route_after_grading` and `route_after_hallucination_check` branch execution after grading events.
  - NB04 agentic GraphRAG uses route conditions over `GraphRAGState` (`grade` -> `global_retrieve`/`web_search`/`generate`).

**Retrieval-Augmented Generation (RAG)**
- Definition: A two-stage architecture where an external knowledge retriever fetches context documents and a generator LLM produces an answer conditioned on that context.
- In this repository: Ingestion/indexing is implemented in `src/ingest.py` and retrieval/generation orchestration appears across notebooks `01`–`09` and `src/rag_v2/*`.

**Dense Retrieval (Embedding-based IR)**
- Definition: Represent query/documents as dense vectors and rank by vector similarity.
- In this repository: `src.retriever.DenseRetriever` and `src.rag_v2.retrieval.DenseRetriever`; FAISS `IndexFlatIP` is used with L2-normalized vectors (`src/ingest.py`).

**Sparse Retrieval (BM25)**
- Definition: Lexical ranking based on term frequency, inverse document frequency, and length normalization.
- In this repository: `src.retriever.BM25Retriever` and `src.rag_v2.retrieval.BM25Retriever` built on `rank_bm25.BM25Okapi`.

**Hybrid Retrieval**
- Definition: Combine dense and sparse signals to improve robustness across paraphrase-heavy and exact-term queries.
- In this repository:
  - Alpha fusion: `hybrid_score = alpha * dense_norm + (1-alpha) * bm25_norm`
  - RRF fusion: reciprocal rank aggregation with `rrf_k=60` (`src/retriever.py`).

**Cross-Encoder Reranking**
- Definition: A second-stage ranking model that scores query-document pairs jointly.
- In this repository: `src.retriever.Reranker`, default model `cross-encoder/ms-marco-MiniLM-L-6-v2`.

**Agentic Workflow / State Machine**
- Definition: A graph of decision states where control flow depends on intermediate quality checks.
- In this repository:
  - NB03 LangGraph CRAG flow with `GraphState` and conditional routes.
  - NB04 agentic GraphRAG with `GraphRAGState` and nodes (`retrieve`, `grade`, `global_retrieve`, `web_search`, `generate`, `grade_hallu`).

**CRAG (Corrective RAG)**
- Definition: Retrieval quality grading + corrective actions (rewrite/retrieve/search) before final answer.
- In this repository: NB03 and NB08 (`rewrite_query`, retrieval grading/faithfulness scoring in `src/rag_v2/agentic.py`).

**Knowledge Graph + Community Detection**
- Definition: Build graph nodes/edges from entities and use modularity-based clustering for topic-level retrieval/synthesis.
- In this repository: `src/graph_builder.py` (entity extraction, graph build, community summarization) and lightweight graph utilities in `src/rag_v2/graph.py`.

**Multimodal RAG**
- Definition: Retrieval/generation pipeline augmented with non-text modalities (PDF page image + OCR + vision summary).
- In this repository: `src/rag_v2/multimodal.py`, NB09.

### 1.2 System Topology (ASCII)

```text
[ArXiv/HF papers]
   |
   v
load_hf_papers / load_arxiv_papers (src/ingest.py)
   |
   v
chunk_documents(chunk_size=512, chunk_overlap=64)
   |
   v
embed_texts(model=qwen3-embedding:0.6b or 4b, batch_size=32)
   |
   +--> FAISS IndexFlatIP + chunks.pkl (artifacts/faiss_index/)
   |         |
   |         +--> DenseRetriever / BM25Retriever / HybridRetriever / Reranker
   |
   +--> ChromaVectorStore or PineconeVectorStore (NB04)
   |         |
   |         +--> local_search / global_search
   |
   +--> Graph pipeline (src/graph_builder.py)
             |-- extract_all_entities(model=granite4.1:8b)
             |-- build_knowledge_graph(NetworkX)
             |-- detect_communities(greedy modularity)
             +-- summarise_all_communities(model=granite4.1:8b)

[Agentic layer]
   NB03 (LangGraph CRAG): retrieve -> grade_documents -> (web_search?) -> generate_answer -> grade_hallucination
   NB07/NB08 (rag_v2): route_query + relevance grading + (optional rewrite_query) + faithfulness scoring

[Multimodal branch NB09]
   arxiv PDF -> pdftoppm first-page PNG -> glm-ocr CLI text + qwen3.5:4b vision -> merged context -> answer

[Evaluation outputs]
   artifacts/eval_results/*.json
   artifacts/rag_v2/*/*_metrics.json
   artifacts/agent_traces/eval_traces.json
```

### 1.3 Technology Stack With Exact Roles and Parameters

**Core models (Ollama)**
- Embedding:
  - `qwen3-embedding:0.6b` (default light path; mapped as 1024-dim in `src/rag_v2/retrieval.py`)
  - `qwen3-embedding:4b` canonical contract: 2560-dim vectors.
  - Practical implication for this codebase: index/store dimension must match the runtime embedding output.
- Generation/Judging: `granite4.1:8b`
- Optional/extended:
  - `qwen3.5:4b` (NB09 answer/vision usage)
  - `glm-ocr` (NB09 OCR via `ollama run glm-ocr`)
  - `granite4.1:3b` (NB04 default judge model; fallback to `granite4.1:8b`)

**Vector search and stores**
- FAISS (`faiss-cpu==1.11.0`)
  - Index: `faiss.IndexFlatIP`
  - Metric: cosine-equivalent through L2 normalization (`FAISS_METRIC = faiss.METRIC_INNER_PRODUCT`).
- ChromaDB (`chromadb==1.5.9`)
  - Collection metadata sets `{"hnsw:space": "cosine"}`.
  - Upsert batching: 500.
- Pinecone (`pinecone==9.1.0`)
  - Defaults in `PineconeVectorStore`: `index_name="agentic-rag-arxiv"`, `dimension=2560`, `cloud="aws"`, `region="us-east-1"`.
  - Upsert batching: 100.

**Retrieval/scoring parameters**
- Default chunking: `DEFAULT_CHUNK_SIZE = 512`, `DEFAULT_CHUNK_OVERLAP = 64`.
- Hybrid defaults:
  - `alpha = 0.7`.
  - `fusion = "alpha"` by default in `src/retriever.py`, optional `"rrf"` with `rrf_k = 60`.
- Reranker:
  - default cross-encoder model `cross-encoder/ms-marco-MiniLM-L-6-v2`.

**LLM options explicitly coded**
- `src/rag_v2/agentic.py`:
  - relevance grading: `temperature=0`, `num_ctx=3072`, `num_predict=96`, `num_gpu=0`
  - answer generation: `temperature=0.2` (parameterized), `num_ctx=3072`, `num_predict=220`
  - faithfulness: `temperature=0`, `num_ctx=3072`, `num_predict=110`
  - rewrite: `temperature=0`, `num_ctx=2048`, `num_predict=48`
- `src/rag_v2/multimodal.py`:
  - qwen fallback text mode: `temperature=0.0`, `num_ctx=4096`, `num_predict=100`
  - qwen image mode: `temperature=0.0`, `num_ctx=8192`, `num_predict=80`

**Agent/graph cache persistence**
- Entity cache path default: `artifacts/graph/entities_cache.json`
- Community summary cache path default: `artifacts/graph/community_summaries.json`
- `CACHE_SAVE_INTERVAL = 10` in `src/graph_builder.py`.

**Documentation stack**
- MkDocs Material with plugins:
  - `search`
  - `git-revision-date-localized`
  - `print-site`
- Custom assets:
  - `docs/stylesheets/extra.css`
  - `docs/javascripts/mathjax.js`


## 📂 Module 2: Strict Repository Tour & Mapping

Repository scope counts used for this module:
- Total files mapped: **258**
- Total directories mapped: **116**

### Directory Mapping
| File/Directory Path | Primary Responsibility | Key Classes/Functions Exported | Actual Configurations/Variables Defined |
|---|---|---|---|
| artifacts | Committed and generated runtime artifacts (indexes, metrics, traces, media) | N/A | file_count_in_scope=39 |
| artifacts/agent_traces | Artifact subdirectory for specific experiment/backend output set | N/A | file_count_in_scope=1 |
| artifacts/cache | Artifact subdirectory for specific experiment/backend output set | N/A | file_count_in_scope=1 |
| artifacts/chromadb | Artifact subdirectory for specific experiment/backend output set | N/A | file_count_in_scope=6 |
| artifacts/chromadb/ef3a1799-9aab-41e6-b9b9-3f14dd06bdd1 | Artifact subdirectory for specific experiment/backend output set | N/A | file_count_in_scope=5 |
| artifacts/eval_results | Artifact subdirectory for specific experiment/backend output set | N/A | file_count_in_scope=16 |
| artifacts/eval_results/legacy_600 | Artifact subdirectory for specific experiment/backend output set | N/A | file_count_in_scope=7 |
| artifacts/faiss_index | Artifact subdirectory for specific experiment/backend output set | N/A | file_count_in_scope=2 |
| artifacts/graph | Artifact subdirectory for specific experiment/backend output set | N/A | file_count_in_scope=3 |
| artifacts/rag_v2 | Artifact subdirectory for specific experiment/backend output set | N/A | file_count_in_scope=9 |
| artifacts/rag_v2/agentic | Artifact subdirectory for specific experiment/backend output set | N/A | file_count_in_scope=1 |
| artifacts/rag_v2/crag | Artifact subdirectory for specific experiment/backend output set | N/A | file_count_in_scope=1 |
| artifacts/rag_v2/graphrag | Artifact subdirectory for specific experiment/backend output set | N/A | file_count_in_scope=1 |
| artifacts/rag_v2/hybrid | Artifact subdirectory for specific experiment/backend output set | N/A | file_count_in_scope=1 |
| artifacts/rag_v2/multimodal | Artifact subdirectory for specific experiment/backend output set | N/A | file_count_in_scope=5 |
| artifacts/rag_v2/multimodal/images | Artifact subdirectory for specific experiment/backend output set | N/A | file_count_in_scope=2 |
| artifacts/rag_v2/multimodal/pdfs | Artifact subdirectory for specific experiment/backend output set | N/A | file_count_in_scope=2 |
| docs | MkDocs source documentation root (tutorial chapters, reference pages, assets) | N/A | file_count_in_scope=63 |
| docs/01_getting_started | Documentation subsection directory | N/A | file_count_in_scope=7 |
| docs/02_core_concepts | Documentation subsection directory | N/A | file_count_in_scope=8 |
| docs/03_part1_naive_rag | Documentation subsection directory | N/A | file_count_in_scope=7 |
| docs/04_part2_advanced_rag | Documentation subsection directory | N/A | file_count_in_scope=6 |
| docs/05_part3_agentic_rag | Documentation subsection directory | N/A | file_count_in_scope=8 |
| docs/06_part4_graph_rag | Documentation subsection directory | N/A | file_count_in_scope=6 |
| docs/07_results | Documentation subsection directory | N/A | file_count_in_scope=4 |
| docs/08_reference | Documentation subsection directory | N/A | file_count_in_scope=7 |
| docs/09_part5_new_techniques | Documentation subsection directory | N/A | file_count_in_scope=6 |
| docs/assets | Documentation subsection directory | N/A | file_count_in_scope=1 |
| docs/javascripts | Documentation subsection directory | N/A | file_count_in_scope=1 |
| docs/stylesheets | Documentation subsection directory | N/A | file_count_in_scope=1 |
| notebooks | Primary tutorial notebooks (`01`-`09`) plus executed notebook snapshots | N/A | file_count_in_scope=17 |
| scripts | Operational scripts for pipeline run, docs QA, and PDF generation | N/A | file_count_in_scope=4 |
| site | Generated MkDocs static site output root | N/A | file_count_in_scope=117 |
| site/01_getting_started | Generated static site subdirectory | N/A | file_count_in_scope=7 |
| site/01_getting_started/handbook_pdf | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/01_getting_started/how_to_use | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/01_getting_started/installation | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/01_getting_started/learning_tracks | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/01_getting_started/prerequisites | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/01_getting_started/tutorial_index | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/01_getting_started/what_is_this_project | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/02_core_concepts | Generated static site subdirectory | N/A | file_count_in_scope=8 |
| site/02_core_concepts/agentic_ai_langgraph | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/02_core_concepts/bm25_sparse_retrieval | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/02_core_concepts/embeddings | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/02_core_concepts/evaluation_metrics | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/02_core_concepts/hybrid_search | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/02_core_concepts/reranking | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/02_core_concepts/vector_databases | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/02_core_concepts/what_is_rag | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/03_part1_naive_rag | Generated static site subdirectory | N/A | file_count_in_scope=7 |
| site/03_part1_naive_rag/chunking | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/03_part1_naive_rag/embedding_indexing | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/03_part1_naive_rag/evaluation | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/03_part1_naive_rag/loading_data | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/03_part1_naive_rag/overview | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/03_part1_naive_rag/results_and_limits | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/03_part1_naive_rag/retrieval_generation | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/04_part2_advanced_rag | Generated static site subdirectory | N/A | file_count_in_scope=6 |
| site/04_part2_advanced_rag/bm25_in_practice | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/04_part2_advanced_rag/cross_encoder_reranking | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/04_part2_advanced_rag/evaluation | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/04_part2_advanced_rag/hybrid_retrieval | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/04_part2_advanced_rag/overview | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/04_part2_advanced_rag/results_and_limits | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/05_part3_agentic_rag | Generated static site subdirectory | N/A | file_count_in_scope=8 |
| site/05_part3_agentic_rag/crag_architecture | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/05_part3_agentic_rag/evaluation | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/05_part3_agentic_rag/langgraph_basics | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/05_part3_agentic_rag/llm_as_judge | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/05_part3_agentic_rag/overview | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/05_part3_agentic_rag/results_and_limits | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/05_part3_agentic_rag/threshold_improvement | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/05_part3_agentic_rag/web_search_fallback | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/06_part4_graph_rag | Generated static site subdirectory | N/A | file_count_in_scope=6 |
| site/06_part4_graph_rag/agentic_graphrag | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/06_part4_graph_rag/chromadb_graphrag | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/06_part4_graph_rag/evaluation | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/06_part4_graph_rag/overview | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/06_part4_graph_rag/pinecone_graphrag | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/06_part4_graph_rag/results_and_limits | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/07_results | Generated static site subdirectory | N/A | file_count_in_scope=4 |
| site/07_results/benchmark_table | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/07_results/improvements_explained | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/07_results/reading_the_numbers | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/07_results/recruiter_brief | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/08_reference | Generated static site subdirectory | N/A | file_count_in_scope=7 |
| site/08_reference/claims_traceability | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/08_reference/further_reading | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/08_reference/glossary | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/08_reference/src_evaluator | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/08_reference/src_ingest | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/08_reference/src_retriever | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/08_reference/troubleshooting | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/09_part5_new_techniques | Generated static site subdirectory | N/A | file_count_in_scope=6 |
| site/09_part5_new_techniques/agentic_rag | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/09_part5_new_techniques/crag | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/09_part5_new_techniques/graphrag | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/09_part5_new_techniques/hybrid_rag | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/09_part5_new_techniques/multimodal_rag | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/09_part5_new_techniques/overview | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/assets | Generated static site subdirectory | N/A | file_count_in_scope=44 |
| site/assets/images | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/assets/javascripts | Generated static site subdirectory | N/A | file_count_in_scope=38 |
| site/assets/javascripts/lunr | Generated static site subdirectory | N/A | file_count_in_scope=34 |
| site/assets/javascripts/lunr/min | Generated static site subdirectory | N/A | file_count_in_scope=32 |
| site/assets/javascripts/workers | Generated static site subdirectory | N/A | file_count_in_scope=2 |
| site/assets/stylesheets | Generated static site subdirectory | N/A | file_count_in_scope=4 |
| site/css | Generated static site subdirectory | N/A | file_count_in_scope=3 |
| site/javascripts | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/js | Generated static site subdirectory | N/A | file_count_in_scope=3 |
| site/print_page | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/search | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| site/stylesheets | Generated static site subdirectory | N/A | file_count_in_scope=1 |
| src | Core Python implementation modules for ingestion/retrieval/eval/vectorstore/graph | N/A | file_count_in_scope=13 |
| src/rag_v2 | Part-5 reusable module package (hybrid/graph/agentic/crag/multimodal helpers) | N/A | file_count_in_scope=7 |

### File Mapping
| File/Directory Path | Primary Responsibility | Key Classes/Functions Exported | Actual Configurations/Variables Defined |
|---|---|---|---|
| .gitattributes | Git attributes / LFS tracking rules for large binary artifacts | N/A | rules: artifacts/chromadb/**/data_level0.bin, artifacts/chromadb/chroma.sqlite3, artifacts/graph/chunk_embeddings.npy |
| .gitignore | Git ignore patterns for env/cache/generated outputs | N/A | patterns include .venv/, data/, .env, site/, artifacts/faiss_index/*.bin/*.pkl |
| README.md | Repository overview, architecture, baseline metrics, quick-start commands | N/A | models: qwen3-embedding:0.6b/4b, granite4.1:8b, qwen3.5:4b, glm-ocr |
| artifacts/agent_traces/eval_traces.json | Agent trace log for LangGraph CRAG evaluation runs | list item keys: question, retrieval_grade, faithfulness_grade, generation_attempts, web_search_triggered, answer, relevant_ids, retrieved_ids | length=10 |
| artifacts/cache/papers_4000.pkl | Cached 4,000-paper corpus snapshot for ingestion reuse | N/A | pickle list[dict] |
| artifacts/chromadb/chroma.sqlite3 | ChromaDB SQLite metadata store | N/A | persist_dir=artifacts/chromadb |
| artifacts/chromadb/ef3a1799-9aab-41e6-b9b9-3f14dd06bdd1/data_level0.bin | ChromaDB internal HNSW binary segment | N/A | managed by chromadb.PersistentClient |
| artifacts/chromadb/ef3a1799-9aab-41e6-b9b9-3f14dd06bdd1/header.bin | ChromaDB internal HNSW binary segment | N/A | managed by chromadb.PersistentClient |
| artifacts/chromadb/ef3a1799-9aab-41e6-b9b9-3f14dd06bdd1/index_metadata.pickle | ChromaDB internal metadata pickle | N/A | N/A |
| artifacts/chromadb/ef3a1799-9aab-41e6-b9b9-3f14dd06bdd1/length.bin | ChromaDB internal HNSW binary segment | N/A | managed by chromadb.PersistentClient |
| artifacts/chromadb/ef3a1799-9aab-41e6-b9b9-3f14dd06bdd1/link_lists.bin | ChromaDB internal HNSW binary segment | N/A | managed by chromadb.PersistentClient |
| artifacts/eval_results.json | Evaluation metrics artifact (retrieval/generation metadata) | top-level keys: experiment_name, retriever_type, embed_model, llm_model, retrieval_metrics, generation_metrics, notes | N/A |
| artifacts/eval_results/01_naive_rag.json | Evaluation metrics artifact (retrieval/generation metadata) | top-level keys: experiment_name, retriever_type, embed_model, llm_model, retrieval_metrics, generation_metrics, notes | N/A |
| artifacts/eval_results/01_naive_rag_4000.json | Evaluation metrics artifact (retrieval/generation metadata) | top-level keys: experiment_name, retriever_type, embed_model, llm_model, retrieval_metrics, generation_metrics, notes | N/A |
| artifacts/eval_results/02_advanced_rag.json | Evaluation metrics artifact (retrieval/generation metadata) | top-level keys: experiment_name, retriever_type, embed_model, llm_model, retrieval_metrics, generation_metrics, notes | N/A |
| artifacts/eval_results/02_advanced_rag_4000.json | Evaluation metrics artifact (retrieval/generation metadata) | top-level keys: experiment_name, retriever_type, embed_model, llm_model, retrieval_metrics, generation_metrics, notes | N/A |
| artifacts/eval_results/03_agentic_rag_4000.json | Evaluation metrics artifact (retrieval/generation metadata) | top-level keys: experiment_name, retriever_type, embed_model, llm_model, retrieval_metrics, generation_metrics, notes | N/A |
| artifacts/eval_results/alpha_ablation.png | Plot/image artifact (evaluation chart or multimodal page render) | N/A | N/A |
| artifacts/eval_results/corpus_stats.png | Plot/image artifact (evaluation chart or multimodal page render) | N/A | N/A |
| artifacts/eval_results/legacy_600/01_naive_rag_legacy_600.json | Evaluation metrics artifact (retrieval/generation metadata) | top-level keys: experiment_name, retriever_type, embed_model, llm_model, retrieval_metrics, generation_metrics, notes | N/A |
| artifacts/eval_results/legacy_600/02_advanced_rag_legacy_600.json | Evaluation metrics artifact (retrieval/generation metadata) | top-level keys: experiment_name, retriever_type, embed_model, llm_model, retrieval_metrics, generation_metrics, notes | N/A |
| artifacts/eval_results/legacy_600/alpha_ablation_legacy_600.png | Plot/image artifact (evaluation chart or multimodal page render) | N/A | N/A |
| artifacts/eval_results/legacy_600/corpus_stats_legacy_600.png | Plot/image artifact (evaluation chart or multimodal page render) | N/A | N/A |
| artifacts/eval_results/legacy_600/eval_results_pipeline_legacy_600.json | Evaluation metrics artifact (retrieval/generation metadata) | top-level keys: experiment_name, retriever_type, embed_model, llm_model, retrieval_metrics, generation_metrics, notes | N/A |
| artifacts/eval_results/legacy_600/retrieval_comparison_improved_legacy_600.png | Plot/image artifact (evaluation chart or multimodal page render) | N/A | N/A |
| artifacts/eval_results/legacy_600/retrieval_comparison_legacy_600.png | Plot/image artifact (evaluation chart or multimodal page render) | N/A | N/A |
| artifacts/eval_results/retrieval_comparison.png | Plot/image artifact (evaluation chart or multimodal page render) | N/A | N/A |
| artifacts/eval_results/retrieval_comparison_improved.png | Plot/image artifact (evaluation chart or multimodal page render) | N/A | N/A |
| artifacts/faiss_index/chunks.pkl | Serialized chunk metadata aligned with FAISS index rows | N/A | pickle list[dict] with chunk_id/paper_id/title/category/text/chunk_index |
| artifacts/faiss_index/index.bin | FAISS binary index persisted by ingestion pipeline | N/A | metric=IndexFlatIP with L2-normalized embeddings |
| artifacts/graph/chunk_embeddings.npy | Numpy embedding matrix artifact for graph/multimodal workflows | N/A | shape≈(n_chunks, embed_dim) |
| artifacts/graph/chunks.pkl | Graph pipeline pickle artifact | N/A | N/A |
| artifacts/graph/entities_cache.json | Entity extraction cache keyed by paper_id | top-level keys: arxiv_train_05005, arxiv_train_05019, arxiv_train_05037, arxiv_train_05043, arxiv_train_05073, arxiv_train_05077, arxiv_train_05079, arxiv_train_05095, arxiv_train_05098, arxiv_train_05103 | N/A |
| artifacts/rag_v2/agentic/07_agentic_metrics.json | Part-5 technique metrics artifact | top-level keys: route_stats, quality, runs | N/A |
| artifacts/rag_v2/crag/08_crag_metrics.json | Part-5 technique metrics artifact | top-level keys: summary, runs | N/A |
| artifacts/rag_v2/graphrag/06_graphrag_metrics.json | Part-5 technique metrics artifact | top-level keys: retrieval, graph | N/A |
| artifacts/rag_v2/hybrid/05_hybrid_metrics.json | Part-5 technique metrics artifact | top-level keys: retrieval, generation | N/A |
| artifacts/rag_v2/multimodal/09_multimodal_metrics.json | Part-5 technique metrics artifact | top-level keys: summary, records, eval | N/A |
| artifacts/rag_v2/multimodal/images/1706.03762_p1.png | Plot/image artifact (evaluation chart or multimodal page render) | N/A | N/A |
| artifacts/rag_v2/multimodal/images/2005.14165_p1.png | Plot/image artifact (evaluation chart or multimodal page render) | N/A | N/A |
| artifacts/rag_v2/multimodal/pdfs/1706.03762.pdf | Downloaded ArXiv PDF artifact for multimodal notebook | N/A | N/A |
| artifacts/rag_v2/multimodal/pdfs/2005.14165.pdf | Downloaded ArXiv PDF artifact for multimodal notebook | N/A | N/A |
| docs/01_getting_started/handbook_pdf.md | Markdown documentation: Full Tutorial PDF | N/A | N/A |
| docs/01_getting_started/how_to_use.md | Markdown documentation: How to Use This Tutorial | N/A | N/A |
| docs/01_getting_started/installation.md | Markdown documentation: Installation | N/A | N/A |
| docs/01_getting_started/learning_tracks.md | Markdown documentation: Learning Tracks | N/A | N/A |
| docs/01_getting_started/prerequisites.md | Markdown documentation: Prerequisites | N/A | N/A |
| docs/01_getting_started/tutorial_index.md | Markdown documentation: Tutorial Index | N/A | N/A |
| docs/01_getting_started/what_is_this_project.md | Markdown documentation: What Is This Project? | N/A | N/A |
| docs/02_core_concepts/agentic_ai_langgraph.md | Markdown documentation: Agentic AI and LangGraph | N/A | N/A |
| docs/02_core_concepts/bm25_sparse_retrieval.md | Markdown documentation: BM25 and Sparse Retrieval | N/A | N/A |
| docs/02_core_concepts/embeddings.md | Markdown documentation: Embeddings — Text as Vectors | N/A | N/A |
| docs/02_core_concepts/evaluation_metrics.md | Markdown documentation: Evaluation Metrics | N/A | N/A |
| docs/02_core_concepts/hybrid_search.md | Markdown documentation: Hybrid Search | N/A | N/A |
| docs/02_core_concepts/reranking.md | Markdown documentation: Reranking | N/A | N/A |
| docs/02_core_concepts/vector_databases.md | Markdown documentation: Vector Databases | N/A | N/A |
| docs/02_core_concepts/what_is_rag.md | Markdown documentation: What Is RAG? | N/A | N/A |
| docs/03_part1_naive_rag/chunking.md | Markdown documentation: Chunking | N/A | N/A |
| docs/03_part1_naive_rag/embedding_indexing.md | Markdown documentation: Embedding and Indexing | N/A | N/A |
| docs/03_part1_naive_rag/evaluation.md | Markdown documentation: Evaluation | N/A | N/A |
| docs/03_part1_naive_rag/loading_data.md | Markdown documentation: Loading Data | N/A | N/A |
| docs/03_part1_naive_rag/overview.md | Markdown documentation: Part 1 — Naive RAG: Overview | N/A | N/A |
| docs/03_part1_naive_rag/results_and_limits.md | Markdown documentation: Results and Limitations | N/A | N/A |
| docs/03_part1_naive_rag/retrieval_generation.md | Markdown documentation: Retrieval and Generation | N/A | N/A |
| docs/04_part2_advanced_rag/bm25_in_practice.md | Markdown documentation: BM25 in Practice | N/A | N/A |
| docs/04_part2_advanced_rag/cross_encoder_reranking.md | Markdown documentation: Cross-Encoder Reranking | N/A | N/A |
| docs/04_part2_advanced_rag/evaluation.md | Markdown documentation: Evaluation | N/A | N/A |
| docs/04_part2_advanced_rag/hybrid_retrieval.md | Markdown documentation: Hybrid Retrieval | N/A | N/A |
| docs/04_part2_advanced_rag/overview.md | Markdown documentation: Part 2 — Advanced RAG Overview | N/A | N/A |
| docs/04_part2_advanced_rag/results_and_limits.md | Markdown documentation: Results and Limits | N/A | N/A |
| docs/05_part3_agentic_rag/crag_architecture.md | Markdown documentation: CRAG Architecture | N/A | N/A |
| docs/05_part3_agentic_rag/evaluation.md | Markdown documentation: Evaluation | N/A | N/A |
| docs/05_part3_agentic_rag/langgraph_basics.md | Markdown documentation: LangGraph Basics | N/A | N/A |
| docs/05_part3_agentic_rag/llm_as_judge.md | Markdown documentation: LLM as Judge | N/A | N/A |
| docs/05_part3_agentic_rag/overview.md | Markdown documentation: Part 3 — Agentic RAG: Overview | N/A | N/A |
| docs/05_part3_agentic_rag/results_and_limits.md | Markdown documentation: Results and Limitations | N/A | N/A |
| docs/05_part3_agentic_rag/threshold_improvement.md | Markdown documentation: Grading Threshold Improvement | N/A | N/A |
| docs/05_part3_agentic_rag/web_search_fallback.md | Markdown documentation: Web Search Fallback | N/A | N/A |
| docs/06_part4_graph_rag/agentic_graphrag.md | Markdown documentation: Agentic GraphRAG (LangGraph) | N/A | N/A |
| docs/06_part4_graph_rag/chromadb_graphrag.md | Markdown documentation: GraphRAG on ChromaDB | N/A | N/A |
| docs/06_part4_graph_rag/evaluation.md | Markdown documentation: Evaluation Design | N/A | N/A |
| docs/06_part4_graph_rag/overview.md | Markdown documentation: Overview | N/A | N/A |
| docs/06_part4_graph_rag/pinecone_graphrag.md | Markdown documentation: GraphRAG on Pinecone | N/A | N/A |
| docs/06_part4_graph_rag/results_and_limits.md | Markdown documentation: Results and Limits | N/A | N/A |
| docs/07_results/benchmark_table.md | Markdown documentation: Full Benchmark Table | N/A | N/A |
| docs/07_results/improvements_explained.md | Markdown documentation: All 5 Improvements Explained | N/A | N/A |
| docs/07_results/reading_the_numbers.md | Markdown documentation: Reading the Numbers Honestly | N/A | N/A |
| docs/07_results/recruiter_brief.md | Markdown documentation: Recruiter Brief | N/A | N/A |
| docs/08_reference/claims_traceability.md | Markdown documentation: Claims Traceability | N/A | N/A |
| docs/08_reference/further_reading.md | Markdown documentation: Further Reading | N/A | N/A |
| docs/08_reference/glossary.md | Markdown documentation: Glossary | N/A | N/A |
| docs/08_reference/src_evaluator.md | Markdown documentation: `src/evaluator.py` — Evaluation Metrics | N/A | N/A |
| docs/08_reference/src_ingest.md | Markdown documentation: `src/ingest.py` — Data Loading and FAISS Indexing | N/A | N/A |
| docs/08_reference/src_retriever.md | Markdown documentation: `src/retriever.py` — Retrieval Strategies | N/A | N/A |
| docs/08_reference/troubleshooting.md | Markdown documentation: Troubleshooting | N/A | N/A |
| docs/09_part5_new_techniques/agentic_rag.md | Markdown documentation: Agentic RAG | N/A | N/A |
| docs/09_part5_new_techniques/crag.md | Markdown documentation: Corrective RAG (CRAG) | N/A | N/A |
| docs/09_part5_new_techniques/graphrag.md | Markdown documentation: GraphRAG | N/A | N/A |
| docs/09_part5_new_techniques/hybrid_rag.md | Markdown documentation: Hybrid RAG | N/A | N/A |
| docs/09_part5_new_techniques/multimodal_rag.md | Markdown documentation: Multimodal RAG | N/A | N/A |
| docs/09_part5_new_techniques/overview.md | Markdown documentation: Part 5 — New RAG Techniques (05-09) | N/A | N/A |
| docs/assets/agentic-rag-full-tutorial.pdf | PDF asset | N/A | N/A |
| docs/index.md | Markdown documentation: Agentic RAG — Zero to Hero | N/A | N/A |
| docs/javascripts/mathjax.js | MathJax client configuration | window.MathJax | options: inlineMath, displayMath, processEscapes, ignoreHtmlClass, processHtmlClass |
| docs/stylesheets/extra.css | Custom MkDocs CSS overrides | N/A | selectors: .md-grid, .md-typeset table, .highlight .filename |
| mkdocs.yml | MkDocs site configuration and navigation for full tutorial docs | N/A | top-level keys: site_name, site_description, site_author, site_url, repo_name, repo_url, edit_uri, theme, nav, markdown_extensions, plugins, extra |
| notebooks/01_naive_rag.executed.ipynb | Part 1 — Naive RAG: Building a Semantic Search + Generation Pipeline from Scratch | defs: naive_rag, find_relevant_ids_by_keywords; imports: import sys; from pathlib import Path; import numpy as np; import pandas as pd | constants: CWD, PROJECT_ROOT, ARTIFACTS_DIR, INDEX_DIR, EVAL_DIR, EMBED_MODEL, RUN_EMBED_MODEL_COMPARISON, RUN_GENERATION_DEMO, PAPER_CACHE, RAG_PROMPT |
| notebooks/01_naive_rag.ipynb | Part 1 — Naive RAG: Building a Semantic Search + Generation Pipeline from Scratch | defs: naive_rag, find_relevant_ids_by_keywords; imports: import sys; from pathlib import Path; import numpy as np; import pandas as pd | constants: CWD, PROJECT_ROOT, ARTIFACTS_DIR, INDEX_DIR, EVAL_DIR, EMBED_MODEL, RUN_EMBED_MODEL_COMPARISON, RUN_GENERATION_DEMO, PAPER_CACHE, RAG_PROMPT |
| notebooks/02_advanced_rag.executed.ipynb | Part 2 — Advanced RAG: Hybrid Search + Cross-Encoder Reranking | defs: find_relevant_ids_by_keywords; imports: import sys; from pathlib import Path; import numpy as np; import pandas as pd | constants: CWD, PROJECT_ROOT, ARTIFACTS_DIR, INDEX_DIR, EVAL_DIR, EMBED_MODEL, OLD_RESULTS |
| notebooks/02_advanced_rag.ipynb | Part 2 — Advanced RAG: Hybrid Search + Cross-Encoder Reranking | defs: find_relevant_ids_by_keywords; imports: import sys; from pathlib import Path; import numpy as np; import pandas as pd | constants: CWD, PROJECT_ROOT, ARTIFACTS_DIR, INDEX_DIR, EVAL_DIR, EMBED_MODEL, OLD_RESULTS |
| notebooks/03_agentic_rag_langgraph.executed.ipynb | Part 3 — Agentic RAG with LangGraph: CRAG State Machine | defs: retrieve, grade_documents, web_search, generate_answer, grade_hallucination, route_after_grading, route_after_hallucination_check, run_agent, find_relevant_ids_by_keywords; imports: import sys; from pathlib import Path; import json; import time | constants: CWD, PROJECT_ROOT, ARTIFACTS_DIR, INDEX_DIR, EVAL_DIR, TRACE_DIR, EMBED_MODEL, LLM_MODEL, GRADING_PROMPT, GENERATION_PROMPT, HALLUCINATION_PROMPT, MAX_REGENERATIONS |
| notebooks/03_agentic_rag_langgraph.ipynb | Part 3 — Agentic RAG with LangGraph: CRAG State Machine | defs: retrieve, grade_documents, web_search, generate_answer, grade_hallucination, route_after_grading, route_after_hallucination_check, run_agent, find_relevant_ids_by_keywords; imports: import sys; from pathlib import Path; import json; import time | constants: CWD, PROJECT_ROOT, ARTIFACTS_DIR, INDEX_DIR, EVAL_DIR, TRACE_DIR, EMBED_MODEL, LLM_MODEL, GRADING_PROMPT, GENERATION_PROMPT, HALLUCINATION_PROMPT, MAX_REGENERATIONS |
| notebooks/04_graph_rag.ipynb | Part 4 — Graph RAG: ChromaDB · Pinecone · Agentic LangGraph | defs: local_search, global_search, _find_ids, ndcg_at_k, f1_at_k, mrr, compute_retrieval_metrics, _normalise, exact_match, bleu; imports: import subprocess, sys; import nltk; import json, os, pickle, sys, re; from pathlib import Path | constants: CWD, PROJECT_ROOT, EMBED_MODEL, LLM_MODEL, JUDGE_MODEL, JUDGE_FALLBACK_MODEL, EMBED_DIM, ARTIFACTS, GRAPH_DIR, CHROMA_DIR, EVAL_DIR, NPY_PATH, CHUNKS_PATH, GRAPH_PATH, G |
| notebooks/05_hybrid_rag.executed.ipynb | Part 5A — Hybrid RAG (Notebook 05) | imports: from __future__ import annotations; import sys; import time; from pathlib import Path | constants: PROJECT_ROOT, ART |
| notebooks/05_hybrid_rag.ipynb | Part 5A — Hybrid RAG (Notebook 05) | imports: from __future__ import annotations; import sys; import time; from pathlib import Path | constants: PROJECT_ROOT, ART |
| notebooks/06_graphrag.executed.ipynb | Part 5B — GraphRAG (Notebook 06) | defs: __init__, retrieve; imports: from __future__ import annotations; import sys; import time; from pathlib import Path | constants: PROJECT_ROOT, ART |
| notebooks/06_graphrag.ipynb | Part 5B — GraphRAG (Notebook 06) | defs: __init__, retrieve; imports: from __future__ import annotations; import sys; import time; from pathlib import Path | constants: PROJECT_ROOT, ART |
| notebooks/07_agentic_rag.executed.ipynb | Part 5C — Agentic RAG (Notebook 07) | defs: graph_retrieve, cheap_relevance_grade, run_agent, clean_nan; imports: from __future__ import annotations; import sys; import time; from pathlib import Path | constants: PROJECT_ROOT, ART |
| notebooks/07_agentic_rag.ipynb | Part 5C — Agentic RAG (Notebook 07) | defs: graph_retrieve, cheap_relevance_grade, run_agent, clean_nan; imports: from __future__ import annotations; import sys; import time; from pathlib import Path | constants: PROJECT_ROOT, ART |
| notebooks/08_crag.executed.ipynb | Part 5D — Corrective RAG (CRAG) (Notebook 08) | defs: cheap_relevance_grade, deterministic_rewrite, crag_answer, clean_nan; imports: from __future__ import annotations; import sys; import time; from pathlib import Path | constants: PROJECT_ROOT, ART |
| notebooks/08_crag.ipynb | Part 5D — Corrective RAG (CRAG) (Notebook 08) | defs: cheap_relevance_grade, deterministic_rewrite, crag_answer, clean_nan; imports: from __future__ import annotations; import sys; import time; from pathlib import Path | constants: PROJECT_ROOT, ART |
| notebooks/09_multimodal_rag.executed.ipynb | Part 5E — Multimodal RAG (Notebook 09) | defs: mm_retrieve, clean_nan; imports: from __future__ import annotations; import sys; import time; from pathlib import Path | constants: PROJECT_ROOT, ART, MM_DIR, PDF_DIR, IMG_DIR |
| notebooks/09_multimodal_rag.ipynb | Part 5E — Multimodal RAG (Notebook 09) | defs: mm_retrieve, clean_nan; imports: from __future__ import annotations; import sys; import time; from pathlib import Path | constants: PROJECT_ROOT, ART, MM_DIR, PDF_DIR, IMG_DIR |
| requirements.txt | Pinned Python dependency manifest for tutorial/runtime stack | N/A | packages: ollama, langchain, langchain-community, langchain-ollama, langgraph, faiss-cpu, rank-bm25, sentence-transformers, datasets, huggingface-hub, arxiv, numpy, pandas, matplotlib, tqdm, ipykernel, notebook, duckduckgo-search |
| scripts/build_tutorial_pdf.py | Build the full tutorial PDF from MkDocs print output. | functions: run_mkdocs_build, find_print_page, render_pdf, parse_args, main | constants: ROOT, SITE_DIR, DEFAULT_OUTPUT |
| scripts/check_docs_facts.py | Validate docs claims against artifacts and scan for unsafe text patterns. | classes: ClaimRow; functions: markdown_files, strip_ticks, parse_claim_rows, try_float, lookup_path, validate_claim_rows, scan_docs_text, main | constants: ROOT, DOCS_ROOT, CLAIMS_FILE, DEPRECATED_PATTERNS, SECRET_PATTERNS, TABLE_HEADER |
| scripts/check_docs_integrity.py | Docs integrity checks for README + MkDocs markdown pages. | functions: markdown_files, is_path_like, normalize_token, iter_path_tokens, resolve_path, path_is_optional_missing, main | constants: ROOT, DOC_ROOT, STALE_PATTERNS, OPTIONAL_PATH_PREFIXES, CODE_PAT, LINK_PAT, SAFE_PATH_TOKEN, ROOT_PATH_PREFIXES, ROOT_FILES |
| scripts/run_pipeline.py | scripts/run_pipeline.py — Full end-to-end pipeline runner | functions: phase1_ingest, phase2_retrieval, phase3_generate, phase4_evaluate, main | constants: ARTIFACTS_DIR, LLM_MODEL, EMBED_MODEL, SAMPLE_QUERIES |
| site/01_getting_started/handbook_pdf/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/01_getting_started/how_to_use/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/01_getting_started/installation/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/01_getting_started/learning_tracks/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/01_getting_started/prerequisites/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/01_getting_started/tutorial_index/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/01_getting_started/what_is_this_project/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/02_core_concepts/agentic_ai_langgraph/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/02_core_concepts/bm25_sparse_retrieval/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/02_core_concepts/embeddings/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/02_core_concepts/evaluation_metrics/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/02_core_concepts/hybrid_search/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/02_core_concepts/reranking/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/02_core_concepts/vector_databases/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/02_core_concepts/what_is_rag/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/03_part1_naive_rag/chunking/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/03_part1_naive_rag/embedding_indexing/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/03_part1_naive_rag/evaluation/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/03_part1_naive_rag/loading_data/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/03_part1_naive_rag/overview/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/03_part1_naive_rag/results_and_limits/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/03_part1_naive_rag/retrieval_generation/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/04_part2_advanced_rag/bm25_in_practice/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/04_part2_advanced_rag/cross_encoder_reranking/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/04_part2_advanced_rag/evaluation/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/04_part2_advanced_rag/hybrid_retrieval/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/04_part2_advanced_rag/overview/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/04_part2_advanced_rag/results_and_limits/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/05_part3_agentic_rag/crag_architecture/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/05_part3_agentic_rag/evaluation/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/05_part3_agentic_rag/langgraph_basics/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/05_part3_agentic_rag/llm_as_judge/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/05_part3_agentic_rag/overview/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/05_part3_agentic_rag/results_and_limits/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/05_part3_agentic_rag/threshold_improvement/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/05_part3_agentic_rag/web_search_fallback/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/06_part4_graph_rag/agentic_graphrag/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/06_part4_graph_rag/chromadb_graphrag/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/06_part4_graph_rag/evaluation/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/06_part4_graph_rag/overview/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/06_part4_graph_rag/pinecone_graphrag/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/06_part4_graph_rag/results_and_limits/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/07_results/benchmark_table/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/07_results/improvements_explained/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/07_results/reading_the_numbers/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/07_results/recruiter_brief/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/08_reference/claims_traceability/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/08_reference/further_reading/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/08_reference/glossary/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/08_reference/src_evaluator/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/08_reference/src_ingest/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/08_reference/src_retriever/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/08_reference/troubleshooting/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/09_part5_new_techniques/agentic_rag/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/09_part5_new_techniques/crag/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/09_part5_new_techniques/graphrag/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/09_part5_new_techniques/hybrid_rag/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/09_part5_new_techniques/multimodal_rag/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/09_part5_new_techniques/overview/index.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/404.html | Generated MkDocs page artifact | N/A | rendered from corresponding docs/*.md source |
| site/assets/agentic-rag-full-tutorial.pdf | Generated/packaged tutorial PDF distributed in site assets | N/A | N/A |
| site/assets/images/favicon.png | Generated static site artifact | N/A | N/A |
| site/assets/javascripts/bundle.13a4f30d.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/bundle.13a4f30d.min.js.map | Generated source-map for minified site asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.ar.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.da.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.de.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.du.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.el.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.es.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.fi.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.fr.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.he.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.hi.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.hu.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.hy.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.it.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.ja.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.jp.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.kn.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.ko.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.multi.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.nl.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.no.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.pt.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.ro.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.ru.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.sa.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.stemmer.support.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.sv.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.ta.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.te.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.th.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.tr.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.vi.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/min/lunr.zh.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/tinyseg.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/lunr/wordcut.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/workers/search.d50fe291.min.js | Generated site JavaScript asset | N/A | N/A |
| site/assets/javascripts/workers/search.d50fe291.min.js.map | Generated source-map for minified site asset | N/A | N/A |
| site/assets/stylesheets/main.342714a4.min.css | Generated site CSS asset | N/A | N/A |
| site/assets/stylesheets/main.342714a4.min.css.map | Generated source-map for minified site asset | N/A | N/A |
| site/assets/stylesheets/palette.06af60db.min.css | Generated site CSS asset | N/A | N/A |
| site/assets/stylesheets/palette.06af60db.min.css.map | Generated source-map for minified site asset | N/A | N/A |
| site/css/print-site-material.css | Generated site CSS asset | N/A | N/A |
| site/css/print-site.css | Generated site CSS asset | N/A | N/A |
| site/css/timeago.css | Generated site CSS asset | N/A | N/A |
| site/index.html | Generated MkDocs home page | N/A | rendered from docs/index.md via mkdocs-material |
| site/javascripts/mathjax.js | MathJax client configuration | window.MathJax | options: inlineMath, displayMath, processEscapes, ignoreHtmlClass, processHtmlClass |
| site/js/print-site.js | Generated site JavaScript asset | N/A | N/A |
| site/js/timeago.min.js | Generated site JavaScript asset | N/A | N/A |
| site/js/timeago_mkdocs_material.js | Generated site JavaScript asset | N/A | N/A |
| site/print_page/index.html | Generated print-site HTML bundle used for handbook PDF rendering | N/A | source plugin: mkdocs-print-site-plugin |
| site/search/search_index.json | Generated static site artifact | N/A | N/A |
| site/sitemap.xml | Generated static site artifact | N/A | N/A |
| site/sitemap.xml.gz | Generated compressed sitemap artifact | N/A | N/A |
| site/stylesheets/extra.css | Generated site CSS asset | N/A | N/A |
| src/__init__.py | Python module `__init__.py` | N/A | N/A |
| src/evaluator.py | src/evaluator.py — RAG evaluation metrics (shared by all three notebooks) | classes: EvalResults; functions: recall_at_k, precision_at_k, mean_reciprocal_rank, compute_retrieval_metrics, score_faithfulness, score_answer_relevance | N/A |
| src/graph_builder.py | src/graph_builder.py — Knowledge graph construction for Graph RAG | functions: _extract_entities_single, extract_all_entities, build_knowledge_graph, detect_communities, _summarise_community_single, summarise_all_communities, get_papers_for_entities, get_entity_ids_for_papers | constants: ENTITY_EXTRACTION_PROMPT, COMMUNITY_SUMMARY_PROMPT, CACHE_SAVE_INTERVAL |
| src/ingest.py | src/ingest.py — Document ingestion pipeline (shared by all three notebooks) | functions: load_arxiv_papers, load_hf_papers, chunk_documents, embed_texts, build_faiss_index, save_index_and_chunks, load_index_and_chunks, embed_query | constants: OLLAMA_CLIENT, EMBED_MODEL_PRIMARY, EMBED_MODEL_LITE, TARGET_CATEGORIES, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, FAISS_METRIC |
| src/rag_v2/__init__.py | Reusable utilities for notebooks 05-09 (new RAG techniques). | N/A | N/A |
| src/rag_v2/agentic.py | Agentic and CRAG-style orchestration helpers for notebooks 07-08. | classes: AgenticResult; functions: _safe_json, llm_relevance_grade, llm_answer, llm_faithfulness, rewrite_query, route_query | N/A |
| src/rag_v2/data.py | Data loading utilities for new notebooks (05-09). | functions: load_base_corpus, load_papers_from_chunks | constants: PROJECT_ROOT, DEFAULT_INDEX_PATH, DEFAULT_CHUNKS_PATH |
| src/rag_v2/graph.py | GraphRAG utilities with lightweight entity extraction. | functions: _extract_entities, build_paper_entity_graph, detect_entity_communities, build_community_summaries, expand_with_graph, build_entity_to_papers | constants: DEFAULT_TERMS |
| src/rag_v2/metrics.py | Evaluation helpers for new RAG notebooks. | classes: GenerationEvalRow; functions: recall_at_k, precision_at_k, mrr, ndcg_at_k, f1_at_k, build_keyword_eval_set, run_retrieval_eval, compute_retrieval_metrics, save_json, plot_retrieval_comparison | N/A |
| src/rag_v2/multimodal.py | Multimodal helpers for notebook 09. | functions: download_arxiv_pdf, pdf_first_page_to_png, _clean_cli_output, run_glm_ocr_cli, run_qwen_vision | constants: ANSI_RE |
| src/rag_v2/retrieval.py | Retrieval primitives for Hybrid / Agentic / CRAG notebooks. | classes: DenseRetriever, BM25Retriever, HybridRetriever, RetrievalBundle; functions: _tokenise | constants: EMBED_MODEL_DIM_TO_NAME, STOP_WORDS |
| src/retriever.py | src/retriever.py — Retrieval strategies (shared by notebooks 02 and 03) | classes: DenseRetriever, BM25Retriever, HybridRetriever, Reranker; functions: _tokenise | constants: BM25_STOP_WORDS |
| src/vectorstore.py | src/vectorstore.py — Swappable vector store layer (ChromaDB and Pinecone) | classes: VectorStore, ChromaVectorStore, PineconeVectorStore | N/A |

## 🔍 Module 3: Line-by-Line Code & Output Breakdown

### 3.1 Core Flow A — Ingestion and Index Build (`src/ingest.py`)

1. Model/timeouts/constants are fixed at module scope:
- `OLLAMA_CLIENT = ollama.Client(timeout=300.0)`
- `EMBED_MODEL_PRIMARY = "qwen3-embedding:4b"`
- `EMBED_MODEL_LITE = "qwen3-embedding:0.6b"`
- `TARGET_CATEGORIES = ["cs.CL", "cs.AI", "cs.LG"]`
- `DEFAULT_CHUNK_SIZE = 512`, `DEFAULT_CHUNK_OVERLAP = 64`

2. `load_arxiv_papers(...)` returns `list[dict]` rows with exact keys:
- `id`, `title`, `abstract`, `category`, `url`.
- Fallback path invokes `load_hf_papers(n_samples=n_samples, ml_filter=True)`.

3. `chunk_documents(...)` transforms paper rows into chunk rows with exact keys:
- `chunk_id`, `paper_id`, `title`, `category`, `text`, `chunk_index`.
- `chunk_id` format: `"<paper_id>_chunk_<n>"`.

4. `embed_texts(...)`
- Batches with `batch_size` (default `32`), calls `OLLAMA_CLIENT.embed(model=model, input=batch)`.
- Collects `response["embeddings"]`.
- Converts to `np.float32` matrix and applies L2 normalization:
  - `norms = np.linalg.norm(embeddings, axis=1, keepdims=True)`
  - `embeddings = embeddings / np.maximum(norms, 1e-10)`.

5. `build_faiss_index(...)`
- Builds `faiss.IndexFlatIP(dim)` and `index.add(embeddings)`.

6. Persistence contract:
- `save_index_and_chunks(...)` writes:
  - `<base_path>/index.bin`
  - `<base_path>/chunks.pkl`
- `load_index_and_chunks(...)` reads same pair and raises `FileNotFoundError` if missing.

### 3.2 Core Flow B — Dense/BM25/Hybrid/Rerank (`src/retriever.py`)

1. Dense retrieve path (`DenseRetriever.retrieve`)
- Query embedding from `embed_query(query, model=self.embed_model)`.
- FAISS call: `scores, indices = self.index.search(query_vec, k)`.
- Result rows are copied chunk dicts plus:
  - `score: float`
  - `retriever: "dense"`.

2. BM25 path (`BM25Retriever`)
- Corpus tokenization via `_tokenise` using regex `[^a-z0-9]` cleanup and stop-word removal.
- Query tokenization uses the same `_tokenise`.
- Results include:
  - `score: float` (raw BM25)
  - `retriever: "bm25"`.

3. Hybrid path (`HybridRetriever.retrieve`)
- Fetches `2*k` from each retriever.
- Alpha fusion (`_alpha_fusion`): min-max normalize each score family, combine with `self.alpha`.
- RRF fusion (`_rrf_fusion`): score accumulation `1.0 / (self.rrf_k + rank + 1)`.
- Output rows set `retriever` to `"hybrid"` or `"hybrid-rrf"`.

4. Rerank path (`Reranker.rerank`)
- Builds `(query, c["text"])` pairs.
- Calls `CrossEncoder.predict(pairs)`.
- Overwrites each row with cross-encoder `score` and `retriever: "reranked"`.

### 3.3 Core Flow C — Metrics Contracts (`src/evaluator.py` + artifacts)

1. Retrieval metrics API:
- `compute_retrieval_metrics(...)` expects eval rows:
  - `{ "question": str, "relevant_ids": list[str] }`
- Returns dict keys:
  - `recall@<k>`, `precision@<k>`, `mrr`, `k`, `n_queries`.

2. Generation judge APIs:
- `score_faithfulness(answer, contexts, judge_model="granite4.1:8b") -> float`
- `score_answer_relevance(question, answer, judge_model="granite4.1:8b") -> float`

3. `EvalResults` schema (serialized JSON):
- `experiment_name`, `retriever_type`, `embed_model`, `llm_model`,
- `retrieval_metrics` (dict), `generation_metrics` (dict), `notes`.

4. Real output examples (from artifacts):
- `artifacts/eval_results/01_naive_rag_4000.json`
  - `retrieval_metrics = {"recall@5":0.05,"precision@5":0.01,"mrr":0.0167,"k":5,"n_queries":20}`
- `artifacts/eval_results/03_agentic_rag_4000.json`
  - `generation_metrics` includes `faithfulness_rate` and `web_search_rate`.

### 3.4 Core Flow D — Graph Construction and Community Summaries (`src/graph_builder.py`)

1. Entity extraction:
- `_extract_entities_single(...)` calls `ollama.chat(..., format="json", options={"temperature":0})`.
- Valid entity row fields expected: `name`, `type`, optional `description`.

2. Cache format:
- `extract_all_entities(...)` returns `dict[str, list[dict]]` where key is `paper_id`.
- Cache path default: `artifacts/graph/entities_cache.json`.

3. Graph schema (`build_knowledge_graph`)
- Paper node attrs: `node_type="paper"`, `title`, `category`.
- Entity node attrs: `node_type="entity"`, `entity_type`, `name`, `description`, `paper_count`.
- Edge relations:
  - paper↔entity: `relation="contains"`, `weight=1`
  - entity↔entity: `relation="co_occurs"`, `weight=<count>`.

4. Community summaries:
- `summarise_all_communities(...)` cache object maps `community_index` to:
  - `summary`, `size`, `entity_names`, `paper_ids`.

### 3.5 Core Flow E — Part-5 `rag_v2` Pipelines (NB05–NB09)

1. Shared corpus loaders (`src/rag_v2/data.py`)
- `load_base_corpus()` reads:
  - `artifacts/faiss_index/index.bin`
  - `artifacts/faiss_index/chunks.pkl`
- `load_papers_from_chunks(...)` reconstructs paper rows with keys:
  - `id`, `title`, `category`, `text`, `n_chunks`.

2. Retrieval metrics (`src/rag_v2/metrics.py`)
- Per-run keys include:
  - `recall@5`, `precision@5`, `mrr`, `f1@5`, `ndcg@5`,
  - `latency_p50_ms`, `latency_p95_ms`, `n_queries`.

3. Agentic helpers (`src/rag_v2/agentic.py`)
- `route_query(question)` outputs one of: `"graph" | "bm25" | "hybrid"`.
- `AgenticResult` fields:
  - `question`, `answer`, `route`, `retrieval_grade`, `retrieval_confidence`,
  - `faithfulness`, `faithfulness_reason`, `latency_ms`, `attempts`, `trace`.

4. Multimodal helpers (`src/rag_v2/multimodal.py`)
- `download_arxiv_pdf(arxiv_id, output_path)` -> PDF file.
- `pdf_first_page_to_png(pdf_path, output_png)` uses `pdftoppm` command.
- `run_glm_ocr_cli(image_path)` calls `ollama run glm-ocr`.
- `run_qwen_vision(...)` calls `ollama.chat(model="qwen3.5:4b", ...)` with image or fallback OCR text.

5. Real multimodal artifact schema (`artifacts/rag_v2/multimodal/09_multimodal_metrics.json`)
- top-level keys: `summary`, `records`, `eval`.
- `records` rows contain `arxiv_id`, `pdf_path`, `image_path`.
- `eval` rows contain `question`, `sources`, `faithfulness`, `latency_ms`, `reason`, `llm_evaluated`.

### 3.6 Core Flow F — CLI Pipeline (`scripts/run_pipeline.py`)

- Global constants:
  - `ARTIFACTS_DIR = artifacts/faiss_index`
  - `LLM_MODEL = "granite4.1:8b"`
  - `EMBED_MODEL = EMBED_MODEL_LITE`
- Phases:
  1. `phase1_ingest(n_papers=4000)`
  2. `phase2_retrieval(...)`
  3. `phase3_generate(...)`
  4. `phase4_evaluate(...)`
- Saved output:
  - `artifacts/eval_results_pipeline_4000.json`.


## 🛠️ Module 4: Step-by-Step Setup & Development Guide

### 4.1 Environment + Tooling (Exact)

1. Clone + enter repo
```bash
git clone https://github.com/pypi-ahmad/agentic-rag-arxiv-research-assistant.git
cd agentic-rag-arxiv-research-assistant
```

2. Pull required Ollama models (core)
```bash
ollama pull qwen3-embedding:0.6b
ollama pull granite4.1:8b
```

3. Optional models used by extended notebooks
```bash
ollama pull qwen3-embedding:4b
ollama pull qwen3.5:4b
ollama pull glm-ocr
```

4. Python environment with `uv` (as documented)
```bash
uv python install 3.13.13
uv venv --python 3.13.13
source .venv/bin/activate
uv pip install -r requirements.txt
```

5. Optional Jupyter kernel registration
```bash
.venv/bin/python -m ipykernel install --user --name agentic-rag-env --display-name "Agentic RAG"
```

### 4.2 Required Runtime Configuration Variables

- `PINECONE_API_KEY`
  - Used by `src.vectorstore.PineconeVectorStore` and NB04 Section 2.
  - If missing, NB04 notebook logic falls back to Chroma simulation path.

No mandatory `.env` file is defined in this repository source; secrets are expected from environment variables.

### 4.3 Conceptual Boot Sequence (Cold Machine -> Working Tutorial)

1. Toolchain readiness
- `ollama`, `uv`, Python 3.13 available.

2. Model availability
- `ollama list` contains at least `qwen3-embedding:0.6b`, `granite4.1:8b`.

3. Notebook dependency order (hard dependency chain)
```text
01_naive_rag.ipynb
-> 02_advanced_rag.ipynb
-> 03_agentic_rag_langgraph.ipynb
-> 04_graph_rag.ipynb
-> 05_hybrid_rag.ipynb
-> 06_graphrag.ipynb
-> 07_agentic_rag.ipynb
-> 08_crag.ipynb
-> 09_multimodal_rag.ipynb
```

4. Artifact dependency gate
- NB02 and NB03 require NB01-generated:
  - `artifacts/faiss_index/index.bin`
  - `artifacts/faiss_index/chunks.pkl`

5. Optional docs/site pipeline
- Docs QA scripts:
```bash
uv run python scripts/check_docs_integrity.py
uv run python scripts/check_docs_facts.py
uv run mkdocs build --strict
```
- Handbook PDF builder:
```bash
uv run python scripts/build_tutorial_pdf.py
```
- Output target: `docs/assets/agentic-rag-full-tutorial.pdf`.

### 4.4 Development Workflow (Repository-native)

- Library code lives under `src/` and `src/rag_v2/`.
- Notebook experiments write evidence into `artifacts/`.
- Documentation source is under `docs/`; generated static site under `site/`.
- Claims should be traceable via `docs/08_reference/claims_traceability.md` and corresponding artifact JSON keys.


## 💼 Module 5: Tech Interview & Hiring Preparation

### 5.1 Five Core Technical Interview Questions

1. Why does this repository keep both dense retrieval and BM25 instead of choosing one retriever globally?
2. In `src/retriever.py::HybridRetriever`, what tradeoff is being made between alpha fusion and RRF, and why does `rrf_k=60` matter?
3. Explain how state/memory is managed in the agentic paths (NB03 LangGraph and `src/rag_v2/agentic.py`) and where determinism is intentionally reduced.
4. Why does `src/graph_builder.py` run community detection on the entity subgraph rather than the full paper+entity graph?
5. What are the deployment implications of using `ChromaVectorStore` vs `PineconeVectorStore` in this codebase, based on actual constructor and batch settings?

### 5.2 Three Hard Engineering Scenarios

1. Scenario A: Retrieval latency P95 in `artifacts/rag_v2/hybrid/05_hybrid_metrics.json` spikes, while P50 stays moderate. How would you isolate the bottleneck in this architecture?
2. Scenario B: `extract_all_entities` takes too long on larger corpora and cache files become huge. How would you redesign without changing the downstream graph API contract?
3. Scenario C: Multimodal NB09 gets empty vision outputs (`vision_avg_chars = 0.0`) but OCR is non-empty. How would you harden answer quality while preserving current function boundaries?

### 5.3 Detailed Model Answers

**Q1 Answer**
- The repository explicitly targets mixed query types.
- Dense path (`DenseRetriever`) captures semantic similarity when wording diverges.
- BM25 path (`BM25Retriever`) captures exact lexical signals for acronyms/technical tokens.
- Evidence: both are constructed and compared in NB02, NB05, NB06, NB07, NB08.
- Tradeoff: extra compute/complexity vs robustness. This codebase chooses robustness and keeps fusion optional.

**Q2 Answer**
- Alpha fusion uses normalized score magnitudes and parameter `alpha` (default `0.7`) to weight dense vs sparse.
- RRF ignores score magnitudes; it uses ranking positions and smoothing `rrf_k=60`.
- `rrf_k` prevents top-rank dominance and stabilizes contribution across lists.
- In this repo, both are available to expose tradeoff experimentally (`fusion="alpha"` or `"rrf"`).

**Q3 Answer**
- NB03 LangGraph encodes state in `GraphState` fields (`question`, `documents`, `retrieval_grade`, `answer`, `faithfulness_grade`, etc.).
- Transition functions use graded decisions; non-determinism enters via LLM outputs for grading and generation (`ollama.chat`).
- `src/rag_v2/agentic.py` uses structured JSON prompts and clamps confidence scores, adding stability but not strict determinism.
- State persistence for auditability appears in artifact traces (`artifacts/agent_traces/eval_traces.json`) and per-run metric JSONs.

**Q4 Answer**
- Full graph includes dense paper-to-entity edges that can dominate clustering due to high-degree paper nodes.
- Entity-only clustering focuses on concept co-occurrence structure (`detect_communities` creates subgraph of nodes where `node_type=="entity"`).
- This improves topic-community interpretation and later global-search summary quality.

**Q5 Answer**
- Chroma path:
  - Local persistent client (`chromadb.PersistentClient`), metadata `{"hnsw:space":"cosine"}`.
  - Upsert batches of 500.
  - Zero cloud credential requirement.
- Pinecone path:
  - Requires `PINECONE_API_KEY`.
  - Creates/uses serverless index with explicit `dimension`, `cloud`, `region`.
  - Upsert batches of 100.
- Architectural implication: same `VectorStore` interface allows backend swap with minimal retrieval code change; operational profile differs (local simplicity vs cloud scalability).

**Scenario A Answer**
- Start with per-stage timing decomposition:
  - query embedding (`embed_query`),
  - dense search,
  - BM25 scoring,
  - fusion/rerank (if enabled),
  - generation/judge calls (if mixed into same benchmark path).
- In this repo, long-tail often comes from sampled LLM-evaluated rows (seen in NB07/NB08 comments and P95 gaps).
- Mitigation:
  - separate retrieval-only benchmarks from LLM-augmented rows,
  - cap LLM eval sampling frequency,
  - enforce dedicated latency metrics by stage in saved JSON schema.

**Scenario B Answer**
- Keep output contract: `dict[paper_id -> list[entity dict]]`.
- Introduce batched/parallel extraction with durable shard files and merge step.
- Keep `CACHE_SAVE_INTERVAL` semantics but store compressed partitioned caches (for example by paper id prefix) to avoid monolithic write amplification.
- Add typed schema validation during merge to protect downstream `build_knowledge_graph` assumptions.

**Scenario C Answer**
- Preserve current boundaries: `run_glm_ocr_cli`, `run_qwen_vision`, `llm_faithfulness`.
- Harden by policy:
  - If vision output is empty, force fallback text mode with OCR (already partially implemented),
  - increase OCR context truncation control and citation-style answer formatting,
  - mark source provenance (`sources`) explicitly in eval rows, and gate final answer confidence on available modalities.
- This keeps pipeline operational even when image branch is unstable while making quality degradation explicit.
