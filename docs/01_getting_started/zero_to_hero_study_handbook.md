# Zero to Hero Study Handbook: Agentic RAG ArXiv Research Assistant

## Scope and Method
- This handbook is based on static analysis of repository files only.
- All names, paths, classes, functions, constants, and JSON keys below come from this codebase.
- No code execution, compilation, or runtime verification is used in this document.

## Module 1: Foundations & Architecture

### 1.1 What this project does
This repository is a production-style tutorial implementation of Retrieval-Augmented Generation (RAG) over an ArXiv-focused ML/AI corpus. It starts with naive dense retrieval and progressively adds BM25, hybrid retrieval, reranking, agentic routing (LangGraph), graph-based retrieval, and multimodal OCR/vision retrieval.

Primary use cases implemented in this repo:
1. Build a local RAG stack from raw paper abstracts to FAISS index.
2. Compare dense vs sparse vs hybrid retrieval quality.
3. Add corrective/agentic control loops before generation.
4. Add graph expansion and community-level global retrieval.
5. Evaluate outputs and store reproducible JSON artifacts under `artifacts/`.

### 1.2 Core paradigms and patterns used here

**Object-Oriented Programming (OOP)**
- Definition: organize behavior and state into classes/objects.
- In this repo:
1. `DenseRetriever`, `BM25Retriever`, `HybridRetriever`, `Reranker` in `src/retriever.py`.
2. `VectorStore`, `ChromaVectorStore`, `PineconeVectorStore` in `src/vectorstore.py`.
3. Dataclasses: `EvalResults` (`src/evaluator.py`), `GenerationEvalRow` (`src/rag_v2/metrics.py`), `AgenticResult` (`src/rag_v2/agentic.py`).

**Functional pipeline style**
- Definition: compose transformations as explicit functions over data.
- In this repo:
1. `load_* -> chunk_documents -> embed_texts -> build_faiss_index -> save_index_and_chunks` in `src/ingest.py`.
2. Metrics functions (`recall_at_k`, `precision_at_k`, `mrr`, `ndcg_at_k`) in `src/evaluator.py` and `src/rag_v2/metrics.py`.

**State-machine / event-driven orchestration**
- Definition: execution advances through nodes and conditional transitions based on state.
- In this repo:
1. Notebook `notebooks/03_agentic_rag_langgraph.ipynb` defines `GraphState` and routes via `route_after_grading` and `route_after_hallucination_check`.
2. Notebook `notebooks/04_graph_rag.ipynb` defines `GraphRAGState`, node functions, and conditional routing (`relevant`, `try_global`, `web_search`).

**Hybrid IR pattern (dense + sparse + rerank)**
- Definition: combine semantic and lexical retrieval, optionally rerank with a stronger cross-encoder.
- In this repo:
1. `HybridRetriever` supports alpha fusion and RRF in `src/retriever.py`.
2. Cross-encoder reranking via `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")` in `src/retriever.py`.

**Graph-based retrieval pattern**
- Definition: use entities and graph connectivity to expand beyond first-hop vector hits.
- In this repo:
1. `extract_all_entities`, `build_knowledge_graph`, `detect_communities`, `summarise_all_communities` in `src/graph_builder.py`.
2. Lightweight graph utilities in `src/rag_v2/graph.py`.

**LLM-as-judge evaluation pattern**
- Definition: use an LLM to score faithfulness/relevance where deterministic labels are unavailable.
- In this repo:
1. `score_faithfulness` and `score_answer_relevance` in `src/evaluator.py`.
2. `llm_relevance_grade` and `llm_faithfulness` in `src/rag_v2/agentic.py`.

### 1.3 Architecture and component interaction

Main architecture layers:
1. Data ingestion/indexing (`src/ingest.py`) creates chunked text + FAISS index.
2. Retrieval layer (`src/retriever.py`, `src/rag_v2/retrieval.py`) provides dense/BM25/hybrid.
3. Agentic/graph layer (`notebooks/03`, `notebooks/04`, `src/rag_v2/agentic.py`, `src/graph_builder.py`).
4. Vector store abstraction for GraphRAG (`src/vectorstore.py` with Chroma/Pinecone backends).
5. Evaluation and artifact logging (`src/evaluator.py`, `src/rag_v2/metrics.py`, `artifacts/**/*.json`).
6. Documentation/PDF export (`mkdocs.yml`, `scripts/build_tutorial_pdf.py`).

ASCII main flow:

```text
[ArXiv / HF data]
   |
   v
src/ingest.py
  load_arxiv_papers / load_hf_papers
   -> chunk_documents
   -> embed_texts
   -> build_faiss_index
   -> save_index_and_chunks
   |
   +--> artifacts/faiss_index/index.bin + chunks.pkl

[Retrieval layer]
   |
   +--> src/retriever.py
   |      DenseRetriever
   |      BM25Retriever
   |      HybridRetriever
   |      Reranker
   |
   +--> notebooks/03_agentic_rag_langgraph.ipynb
   |      GraphState nodes + routing
   |
   +--> notebooks/04_graph_rag.ipynb
   |      GraphRAGState + local/global/web routes
   |      src/vectorstore.py (Chroma/Pinecone)
   |
   +--> src/rag_v2/* (NB05-NB09 reusable modules)

[Evaluation + outputs]
   |
   +--> src/evaluator.py, src/rag_v2/metrics.py
   +--> artifacts/eval_results/*.json
   +--> artifacts/rag_v2/*/*.json
   +--> artifacts/agent_traces/eval_traces.json
```

Important implementation note (dimension consistency):
1. `src/ingest.py` comment says `EMBED_MODEL_PRIMARY = "qwen3-embedding:4b"` produces "4096-dim" vectors.
2. `src/rag_v2/retrieval.py` maps `2560` to `qwen3-embedding:4b` in `EMBED_MODEL_DIM_TO_NAME`.
3. `src/vectorstore.py` defaults Pinecone `dimension=2560`.

A new contributor should treat embedding dimension as a runtime contract that must match the index/store used in that workflow.

## Module 2: Repository Map

The table below focuses on the first files a contributor should master.

| File/Directory Path | Primary Responsibility | Key Classes/Functions | Important Configs/Variables |
|---|---|---|---|
| `README.md` | Project orientation, technique ladder, quick-start commands, artifact-backed metrics | N/A | 4,000 baseline and legacy 600 references |
| `requirements.txt` | Exact dependency lock for runtime, notebooks, docs | N/A | `ollama==0.4.8`, `langgraph==0.4.8`, `faiss-cpu==1.11.0`, `chromadb==1.5.9`, `pinecone==9.1.0` |
| `mkdocs.yml` | Documentation site configuration and navigation | N/A | `plugins: search, git-revision-date-localized, print-site` |
| `src/ingest.py` | Corpus loading, chunking, embedding, FAISS build, index persistence | `load_arxiv_papers`, `load_hf_papers`, `chunk_documents`, `embed_texts`, `build_faiss_index`, `save_index_and_chunks`, `load_index_and_chunks`, `embed_query` | `EMBED_MODEL_PRIMARY`, `EMBED_MODEL_LITE`, `DEFAULT_CHUNK_SIZE=512`, `DEFAULT_CHUNK_OVERLAP=64`, `TARGET_CATEGORIES` |
| `src/retriever.py` | Dense/BM25/hybrid retrieval + cross-encoder reranking | `DenseRetriever`, `BM25Retriever`, `HybridRetriever`, `Reranker`, `_tokenise` | `BM25_STOP_WORDS`, `alpha=0.7`, `fusion`, `rrf_k=60` |
| `src/evaluator.py` | Retrieval and generation evaluation + result serialization | `recall_at_k`, `precision_at_k`, `mean_reciprocal_rank`, `compute_retrieval_metrics`, `score_faithfulness`, `score_answer_relevance`, `EvalResults` | Default judge model `granite4.1:8b` |
| `src/graph_builder.py` | Entity extraction, graph creation, community detection, community summarization | `_extract_entities_single`, `extract_all_entities`, `build_knowledge_graph`, `detect_communities`, `summarise_all_communities` | `ENTITY_EXTRACTION_PROMPT`, `COMMUNITY_SUMMARY_PROMPT`, `CACHE_SAVE_INTERVAL=10` |
| `src/vectorstore.py` | Unified vector-store interface with local and cloud backends | `VectorStore`, `ChromaVectorStore`, `PineconeVectorStore` | Chroma `collection_name="arxiv_graph_rag"`, Pinecone `index_name="agentic-rag-arxiv"`, `dimension=2560`, `cloud="aws"`, `region="us-east-1"` |
| `src/rag_v2/data.py` | Shared corpus loader for notebooks 05-09 | `load_base_corpus`, `load_papers_from_chunks` | `DEFAULT_INDEX_PATH`, `DEFAULT_CHUNKS_PATH` |
| `src/rag_v2/retrieval.py` | Retrieval primitives for new notebooks | `DenseRetriever`, `BM25Retriever`, `HybridRetriever`, `RetrievalBundle` | `EMBED_MODEL_DIM_TO_NAME`, `STOP_WORDS`, `alpha=0.7` |
| `src/rag_v2/graph.py` | Lightweight GraphRAG entity extraction and expansion | `_extract_entities`, `build_paper_entity_graph`, `detect_entity_communities`, `build_community_summaries`, `expand_with_graph` | `DEFAULT_TERMS` |
| `src/rag_v2/agentic.py` | Agentic grading, answering, faithfulness, rewrite, routing helpers | `llm_relevance_grade`, `llm_answer`, `llm_faithfulness`, `rewrite_query`, `route_query`, `AgenticResult` | LLM options: `temperature`, `num_ctx`, `num_predict`, `num_gpu` |
| `src/rag_v2/metrics.py` | Retrieval/generation metrics for notebooks 05-09 | `run_retrieval_eval`, `compute_retrieval_metrics`, `plot_retrieval_comparison`, `GenerationEvalRow` | keys include `latency_p50_ms`, `latency_p95_ms`, `f1@5`, `ndcg@5` |
| `src/rag_v2/multimodal.py` | PDF download, first-page rendering, OCR, vision call helpers | `download_arxiv_pdf`, `pdf_first_page_to_png`, `run_glm_ocr_cli`, `run_qwen_vision` | subprocess tools `pdftoppm`, CLI `ollama run glm-ocr` |
| `scripts/run_pipeline.py` | Non-notebook end-to-end runner with 4 phases | `phase1_ingest`, `phase2_retrieval`, `phase3_generate`, `phase4_evaluate`, `main` | `ARTIFACTS_DIR`, `LLM_MODEL`, `EMBED_MODEL`, `SAMPLE_QUERIES` |
| `scripts/check_docs_integrity.py` | Docs validation for links/references (QA) | script entrypoint | N/A |
| `scripts/check_docs_facts.py` | Docs fact checks and consistency QA | script entrypoint | N/A |
| `scripts/build_tutorial_pdf.py` | Build PDF from MkDocs print output | `run_mkdocs_build`, `find_print_page`, `render_pdf`, `main` | `DEFAULT_OUTPUT=docs/assets/agentic-rag-full-tutorial.pdf` |
| `notebooks/01_naive_rag.ipynb` | Part 1: corpus build, dense retrieval baseline | defines `naive_rag` in notebook | `EMBED_MODEL`, `RUN_EMBED_MODEL_COMPARISON`, `RAG_PROMPT` |
| `notebooks/02_advanced_rag.ipynb` | Part 2: BM25 + hybrid + rerank experiments | notebook flow using `src/retriever.py` | eval artifacts in `artifacts/eval_results/` |
| `notebooks/03_agentic_rag_langgraph.ipynb` | Part 3: LangGraph CRAG state-machine | `retrieve`, `grade_documents`, `web_search`, `generate_answer`, `grade_hallucination`, `run_agent` | `GraphState`, `MAX_REGENERATIONS=2`, `TRACE_DIR` |
| `notebooks/04_graph_rag.ipynb` | Part 4: GraphRAG with Chroma/Pinecone and agentic graph | `local_search`, `global_search`, `retrieve_node`, `global_retrieve_node`, `grade_node`, `web_search_node`, `generate_node`, `grade_hallucination_node` | `GraphRAGState`, `EMBED_MODEL='qwen3-embedding:4b'`, `LLM_MODEL`, `GUARDIAN_MODEL` |
| `notebooks/05_hybrid_rag.ipynb` | Part 5A: Hybrid RAG with `src/rag_v2` modules | notebook orchestration | output `artifacts/rag_v2/hybrid/05_hybrid_metrics.json` |
| `notebooks/06_graphrag.ipynb` | Part 5B: GraphRAG with `src/rag_v2/graph.py` | notebook orchestration | output `artifacts/rag_v2/graphrag/06_graphrag_metrics.json` |
| `notebooks/07_agentic_rag.ipynb` | Part 5C: Agentic routing with `src/rag_v2/agentic.py` | `graph_retrieve`, `cheap_relevance_grade`, `run_agent` | output `artifacts/rag_v2/agentic/07_agentic_metrics.json` |
| `notebooks/08_crag.ipynb` | Part 5D: Corrective RAG query rewrite loop | `cheap_relevance_grade`, `crag_answer` | output `artifacts/rag_v2/crag/08_crag_metrics.json` |
| `notebooks/09_multimodal_rag.ipynb` | Part 5E: OCR + vision path | `mm_retrieve` | output `artifacts/rag_v2/multimodal/09_multimodal_metrics.json` |
| `artifacts/eval_results/01_naive_rag_4000.json` | Baseline retrieval metrics schema example | N/A | `experiment_name`, `retriever_type`, `embed_model`, `llm_model`, `retrieval_metrics`, `generation_metrics`, `notes` |
| `artifacts/eval_results/03_agentic_rag_4000.json` | Agentic CRAG metrics example | N/A | `generation_metrics.faithfulness_rate`, `generation_metrics.web_search_rate` |
| `artifacts/rag_v2/multimodal/09_multimodal_metrics.json` | Multimodal run schema | N/A | top-level keys `summary`, `records`, `eval` |

## Module 3: Core Execution Flows

### Flow 1: Ingestion and index build (`src/ingest.py`)

Step-by-step:
1. `load_arxiv_papers(...)` or `load_hf_papers(...)` returns rows with keys:
`id`, `title`, `abstract`, `category`, `url`.
2. `chunk_documents(...)` splits abstracts into overlapping chunks and emits dicts:
`chunk_id`, `paper_id`, `title`, `category`, `text`, `chunk_index`.
3. `embed_texts(texts, model, batch_size=32)` calls `OLLAMA_CLIENT.embed(...)`, collects `response["embeddings"]`, converts to `np.float32`, and L2-normalizes.
4. `build_faiss_index(embeddings)` creates `faiss.IndexFlatIP(dim)` and adds all vectors.
5. `save_index_and_chunks(index, chunks, base_path)` writes:
`index.bin` and `chunks.pkl`.

Exact chunk structure from `chunk_documents`:

```python
{
  "chunk_id": f"{paper['id']}_chunk_{chunk_idx}",
  "paper_id": paper["id"],
  "title": paper["title"],
  "category": paper["category"],
  "text": chunk_text,
  "chunk_index": chunk_idx,
}
```

### Flow 2: Retrieval stack (`src/retriever.py`)

Step-by-step:
1. `DenseRetriever.retrieve(query, k)`:
- query vector via `embed_query(query, model=self.embed_model)`.
- FAISS call `scores, indices = self.index.search(query_vec, k)`.
- returns chunk dicts augmented with `score` and `retriever="dense"`.
2. `BM25Retriever.retrieve(query, k)`:
- tokenization via `_tokenise` (regex cleanup + stop-word removal).
- BM25 scores from `BM25Okapi`.
- returns chunk dicts with `retriever="bm25"`.
3. `HybridRetriever.retrieve(query, k)`:
- fetches from both retrievers.
- uses either `_alpha_fusion` or `_rrf_fusion`.
- outputs rows with `retriever="hybrid"` or `retriever="hybrid-rrf"`.
4. `Reranker.rerank(query, candidates, top_k)`:
- creates `(query, c["text"])` pairs.
- scores with cross-encoder.
- outputs top rows with `retriever="reranked"`.

### Flow 3: CLI end-to-end runner (`scripts/run_pipeline.py`)

Entrypoint sequence (`main`):
1. `phase1_ingest(n_papers=4000)`.
2. `phase2_retrieval(index, chunks)`.
3. `phase3_generate(index, chunks)`.
4. `phase4_evaluate(query, answer, retrieved, chunks)`.

Important constants in this script:
- `ARTIFACTS_DIR = artifacts/faiss_index`
- `LLM_MODEL = "granite4.1:8b"`
- `EMBED_MODEL = EMBED_MODEL_LITE`

Evaluation object persisted by this script uses `EvalResults` and is written to `artifacts/eval_results_pipeline_4000.json`.

### Flow 4: LangGraph CRAG flow (`notebooks/03_agentic_rag_langgraph.ipynb`)

`GraphState` fields in the notebook:
- `question: str`
- `documents: list`
- `filtered_documents: list`
- `retrieval_grade: str`
- `generation: str`
- `faithfulness_grade: str`
- `generation_attempts: int`
- `execution_trace: list`

Core node functions defined in notebook:
1. `retrieve(state)`
2. `grade_documents(state)`
3. `web_search(state)`
4. `generate_answer(state)`
5. `grade_hallucination(state)`
6. routing functions `route_after_grading(...)`, `route_after_hallucination_check(...)`

Runtime wrapper:
- `run_agent(question: str, verbose: bool = True) -> dict`
- initializes `initial_state` with empty/default values and invokes `rag_agent.invoke(initial_state)`.

### Flow 5: GraphRAG flow (`notebooks/04_graph_rag.ipynb` + `src/vectorstore.py` + `src/graph_builder.py`)

Notebook state type `GraphRAGState` fields:
- `question`, `retrieved`, `grade`, `answer`, `faith_grade`, `store`, `iteration`, `web_results`.

Node pipeline:
1. `retrieve_node` calls `local_search(...)`.
2. `grade_node` routes to `generate`, `global_retrieve`, or `web_search`.
3. `global_retrieve_node` injects community summaries as chunk-like rows.
4. `web_search_node` uses `DuckDuckGoSearchResults(max_results=3)` if available.
5. `generate_node` produces answer from current context.
6. `grade_hallucination_node` evaluates groundedness.

Store compatibility contract:
- Both `ChromaVectorStore.search(...)` and `PineconeVectorStore.search(...)` return chunk-like dicts with:
`chunk_id`, `paper_id`, `title`, `category`, `chunk_idx`, `text`, `score`, `retriever`.

### Flow 6: Part-5 modular flows (`src/rag_v2/*` used by notebooks 05-09)

1. Data load:
- `load_base_corpus()` reads `artifacts/faiss_index/index.bin` and `chunks.pkl`.
- `load_papers_from_chunks(...)` reconstructs paper-level rows with keys:
`id`, `title`, `category`, `text`, `n_chunks`.

2. Routing and correction:
- `route_query(question)` returns one of `"graph"`, `"bm25"`, `"hybrid"`.
- `llm_relevance_grade(...)` returns `(grade, confidence)`.
- `rewrite_query(...)` rewrites low-quality queries for CRAG loops.

3. Multimodal path:
- `download_arxiv_pdf(arxiv_id, output_path)`.
- `pdf_first_page_to_png(pdf_path, output_png)` using `pdftoppm`.
- `run_glm_ocr_cli(image_path)` using CLI `ollama run glm-ocr`.
- `run_qwen_vision(...)` for fallback-text or direct image vision prompt.

### Exact input/output shapes from committed artifacts

`artifacts/eval_results/01_naive_rag_4000.json`:
- Top-level keys: `experiment_name`, `retriever_type`, `embed_model`, `llm_model`, `retrieval_metrics`, `generation_metrics`, `notes`.
- `retrieval_metrics` keys: `recall@5`, `precision@5`, `mrr`, `k`, `n_queries`.

`artifacts/eval_results/03_agentic_rag_4000.json`:
- `generation_metrics` includes `faithfulness_rate` and `web_search_rate`.

`artifacts/rag_v2/multimodal/09_multimodal_metrics.json`:
- top-level: `summary`, `records`, `eval`.
- `records[]`: `arxiv_id`, `pdf_path`, `image_path`.
- `eval[]`: `question`, `sources`, `faithfulness`, `latency_ms`, `reason`, `llm_evaluated`.

## Module 4: Setup & Run Guide

### 4.1 Clean-machine installation path

1. Clone repo:

```bash
git clone https://github.com/pypi-ahmad/agentic-rag-arxiv-research-assistant.git
cd agentic-rag-arxiv-research-assistant
```

2. Pull required models (core path):

```bash
ollama pull qwen3-embedding:0.6b
ollama pull granite4.1:8b
```

3. Optional models (extended notebooks):

```bash
ollama pull qwen3-embedding:4b
ollama pull qwen3.5:4b
ollama pull glm-ocr
```

4. Create environment with `uv`:

```bash
uv python install 3.13.13
uv venv --python 3.13.13
source .venv/bin/activate
uv pip install -r requirements.txt
```

5. Optional kernel setup:

```bash
.venv/bin/python -m ipykernel install --user --name agentic-rag-env --display-name "Agentic RAG"
```

### 4.2 Required environment variables and config files

Required for optional Pinecone path:
1. `PINECONE_API_KEY` (used in `src/vectorstore.py` and notebook 04 Pinecone section).

No mandatory `.env` keys are enforced for core local FAISS/Chroma flows.

Important config files:
1. `requirements.txt` for Python dependencies.
2. `mkdocs.yml` for docs build and print-site pipeline.
3. Notebook constants inside each notebook (for model names, artifact directories, and evaluation settings).

### 4.3 Typical run sequences

Notebook-first tutorial sequence (from `docs/01_getting_started/tutorial_index.md`):
1. `notebooks/01_naive_rag.ipynb`
2. `notebooks/02_advanced_rag.ipynb`
3. `notebooks/03_agentic_rag_langgraph.ipynb`
4. `notebooks/04_graph_rag.ipynb`
5. `notebooks/05_hybrid_rag.ipynb`
6. `notebooks/06_graphrag.ipynb`
7. `notebooks/07_agentic_rag.ipynb`
8. `notebooks/08_crag.ipynb`
9. `notebooks/09_multimodal_rag.ipynb`

Scripted non-notebook path:

```bash
python scripts/run_pipeline.py
```

Docs + PDF path:

```bash
uv run python scripts/check_docs_integrity.py
uv run python scripts/check_docs_facts.py
uv run mkdocs build --strict
uv run python scripts/build_tutorial_pdf.py
```

### 4.4 Migration/seeding and external services

Database migrations:
- None (no relational migration framework in this repository).

Data/index seeding:
1. FAISS corpus artifacts are generated by notebook 01 or `scripts/run_pipeline.py` phase 1.
2. Graph/entity caches are persisted under `artifacts/graph/`.
3. Chroma collection is persisted under `artifacts/chromadb/`.
4. Pinecone index is created lazily by `PineconeVectorStore` when key and configuration are valid.

External/runtime dependencies:
1. Ollama server for embeddings/generation/judging.
2. Optional Pinecone service for cloud vector storage.
3. `pdftoppm` command-line tool required by multimodal helper `pdf_first_page_to_png`.
4. Chrome/Chromium required by `scripts/build_tutorial_pdf.py` for PDF rendering.

## Module 5: Study Plan & Practice Exercises

### 5.1 Ordered study plan

Phase 1: Build mental model and file map.
1. Read `README.md`.
2. Read `docs/01_getting_started/tutorial_index.md`.
3. Read `requirements.txt` and `mkdocs.yml`.

Phase 2: Learn core runtime.
1. Read `src/ingest.py` end to end.
2. Read `src/retriever.py` and `src/evaluator.py`.
3. Read `scripts/run_pipeline.py` and trace function call order.

Phase 3: Learn agentic and graph systems.
1. Read `notebooks/03_agentic_rag_langgraph.ipynb` focusing on `GraphState`, node functions, and routing.
2. Read `src/graph_builder.py` and `src/vectorstore.py`.
3. Read `notebooks/04_graph_rag.ipynb` state graph assembly.

Phase 4: Learn new technique modules.
1. Read `src/rag_v2/data.py`, `src/rag_v2/retrieval.py`, `src/rag_v2/metrics.py`.
2. Read `src/rag_v2/agentic.py` and `src/rag_v2/graph.py`.
3. Read `src/rag_v2/multimodal.py` and `notebooks/09_multimodal_rag.ipynb`.

Phase 5: Validate understanding through artifacts.
1. Compare schemas in `artifacts/eval_results/*.json` and `artifacts/rag_v2/*/*.json`.
2. Cross-check claims with `docs/08_reference/claims_traceability.md`.

### 5.2 Practice exercises

1. Trace the complete call chain that creates `artifacts/faiss_index/index.bin`.
2. Explain why `embed_texts` L2-normalizes vectors before `IndexFlatIP` search.
3. In `src/retriever.py`, compare alpha fusion and RRF. What data does each method ignore?
4. List every field in notebook 03 `GraphState` and explain which node updates it first.
5. In notebook 03, what exactly prevents infinite generation retries?
6. In notebook 04, write the route transitions from `grade` for each grade value.
7. Compare output schema differences between `01_naive_rag_4000.json` and `09_multimodal_metrics.json`.
8. Identify all places where `PINECONE_API_KEY` is used or checked in this repo.
9. Describe the fallback path when web search is unavailable in notebook 04.
10. Explain the multimodal fallback order in `run_qwen_vision`.

### 5.3 Model answer outlines

1. `scripts/run_pipeline.py` -> `phase1_ingest` -> `load_hf_papers` -> `chunk_documents` -> `embed_texts` -> `build_faiss_index` -> `save_index_and_chunks` writes `index.bin` and `chunks.pkl` under `artifacts/faiss_index`.
2. L2 normalization makes inner product equivalent to cosine similarity, so `IndexFlatIP` can rank by semantic angle without additional conversion.
3. Alpha fusion blends normalized score magnitudes (`alpha * dense + (1-alpha) * bm25`); RRF ignores score magnitudes and uses rank positions only.
4. Fields: `question`, `documents`, `filtered_documents`, `retrieval_grade`, `generation`, `faithfulness_grade`, `generation_attempts`, `execution_trace`. First updates: `retrieve` sets `documents`; `grade_documents` sets `filtered_documents` and `retrieval_grade`; `generate_answer` sets `generation`; `grade_hallucination` sets `faithfulness_grade`.
5. `MAX_REGENERATIONS = 2` and `generation_attempts` gate retries in routing function `route_after_hallucination_check`.
6. notebook 04 route map from `grade`: `relevant -> generate`, `try_global -> global_retrieve`, `web_search -> web_search`; then `global_retrieve -> grade`, `web_search -> generate`, `generate -> grade_hallu -> END`.
7. `01_naive_rag_4000.json` is experiment-level metrics (`retrieval_metrics`, `generation_metrics`, `notes`), while `09_multimodal_metrics.json` includes operational arrays (`records`, `eval`) and summary latency/OCR fields.
8. `src/vectorstore.py` checks env in `PineconeVectorStore.__init__`; notebook 04 reads `os.environ.get("PINECONE_API_KEY", "")` and toggles fallback behavior.
9. In `web_search_node`, if import/init fails then `WEB_AVAILABLE=False`; node returns `results=["Web search unavailable."]` and wraps into `web_chunks` with `retriever="web"`.
10. `run_qwen_vision` first tries text-mode using `fallback_text` and a prompt; if response is empty, it then attempts image-mode with `messages=[{"role":"user", "content": question, "images": [str(image_path)]}]`; on exception returns empty string.

## Understanding Checklist

Use this checklist after studying:

1. Can you explain the exact data contract from `load_hf_papers` output to `chunk_documents` output?
2. Can you explain why `embed_query` model must match the model used during index creation?
3. Can you draw the difference between `DenseRetriever`, `BM25Retriever`, `HybridRetriever`, and `Reranker` without looking at code?
4. Can you describe notebook 03 CRAG routing decisions (`retrieve -> grade -> generate/web_search -> hallucination grade -> retry/end`)?
5. Can you describe notebook 04 GraphRAG routing and how `global_retrieve` differs from `web_search`?
6. Can you state the exact required env var for Pinecone mode and what the fallback behavior is when it is missing?
7. Can you point to where retrieval and generation metrics are computed and serialized?
8. Can you explain the schema of at least two committed artifact JSON files from memory?
9. Can you explain the PDF export path from `mkdocs.yml` print-site plugin to `scripts/build_tutorial_pdf.py`?
10. Can you identify at least one repo-internal consistency risk (for example embedding-dimension assumptions) and where it appears in code?
