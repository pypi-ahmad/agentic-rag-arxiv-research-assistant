# Agentic RAG with LangGraph — ArXiv ML/AI Research Paper Q&A

End-to-end tutorial demonstrating **Retrieval-Augmented Generation (RAG)** from first principles to a production-grade agentic pipeline, applied to a real corpus of ArXiv machine learning and AI paper abstracts.

Three progressively more powerful systems are built, evaluated, and compared:
- **Part 1 (Naive RAG)**: FAISS semantic search + granite4.1:8b generation — the baseline
- **Part 2 (Advanced RAG)**: BM25 + dense hybrid retrieval + cross-encoder reranking
- **Part 3 (Agentic RAG)**: LangGraph CRAG state machine with document grading, web search fallback, and hallucination detection

Every concept is explained from scratch. No prior RAG or LangGraph experience is assumed.

---

## Overview

**What it does:**  
Builds a Q&A system over ~2000 ArXiv ML/AI paper abstracts that correctly answers questions like "What is retrieval-augmented generation?", "How does RLHF work?", and "What are the key differences between LoRA and full fine-tuning?"

**Why it exists:**  
RAG is now one of the most widely deployed patterns in production AI systems, yet most tutorials stop at the naive pipeline. This project teaches the full progression — including why naive RAG fails, how hybrid retrieval fixes it, and how LangGraph enables self-correcting agentic behaviour. Every step is tied to a measurable metric improvement.

**Key objectives:**
1. Understand and implement the complete RAG stack from scratch (chunking → embedding → indexing → retrieval → generation → evaluation)
2. Measure the retrieval quality gap between dense-only, BM25-only, and hybrid+reranked strategies
3. Implement the CRAG (Corrective RAG, 2024) architecture as a LangGraph state machine with 6 nodes and conditional routing
4. Produce a reproducible 3-way comparison table across Naive, Advanced, and Agentic RAG

---

## Prerequisites

### Knowledge prerequisites

You do not need to know RAG, LangGraph, or LangChain. Everything is explained from scratch in the notebooks. You do need:

| Concept | Level required | Why it's needed |
|---------|---------------|----------------|
| Python — functions, classes, dicts, list comprehensions | Comfortable | All code is in Python |
| What a numpy array is | `np.array([1,2,3])` level is enough | Embeddings are float arrays |
| What an API call is (send request, get response) | Basic | We call the local Ollama API |
| What a neural network does at a high level | "maps input → output" is sufficient | Understanding embeddings and LLMs |
| Git, terminal, file paths | Basic | Installation and running notebooks |

You do **not** need: linear algebra (beyond understanding dot product ≈ similarity), transformer architecture details, prior LangChain/LangGraph experience, or cloud accounts of any kind.

### Tool prerequisites

```bash
# 1. Ollama — runs LLMs locally (no API keys)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull the models used in this tutorial
ollama pull qwen3-embedding:4b      # primary embedding model (~2.5 GB)
ollama pull qwen3-embedding:0.6b    # lite comparison (~639 MB)
ollama pull granite4.1:8b           # generation + grading LLM (~5.3 GB)
ollama pull phi4-mini:3.8b          # optional comparison LLM (~2.5 GB)

# 3. Python 3.12 + uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify:
```bash
ollama list          # shows pulled models
python3 --version    # should show 3.12.x
uv --version
```

---

## Tutorial Structure

| Part | Notebook | What you build | New concepts |
|------|----------|---------------|-------------|
| 1 | `01_naive_rag.ipynb` | Download ArXiv corpus, chunk, embed with qwen3-embedding:4b, build FAISS index, basic RAG chain | Embeddings, cosine similarity, FAISS, chunking, RAG prompt design, Recall@k / Precision@k / MRR |
| 2 | `02_advanced_rag.ipynb` | BM25 indexer, hybrid score fusion, cross-encoder reranker, alpha ablation | BM25 math, hybrid retrieval, bi-encoder vs. cross-encoder, two-stage retrieve-then-rerank |
| 3 | `03_agentic_rag_langgraph.ipynb` | LangGraph CRAG state machine with 6 nodes and conditional routing | State machines, LangGraph TypedDict state, conditional edges, LLM-as-judge, CRAG architecture, web search tool |

---

## Architecture

### Phase A — Indexing (run once in notebook 01)

```
ArXiv papers (ccdv/arxiv-summarization)
      │
      ▼  load_arxiv_papers()
Filter to cs.CL, cs.AI, cs.LG  (~2000 papers)
      │
      ▼  chunk_documents()
Overlapping 512-char chunks with 64-char overlap  (~4000 chunks)
      │
      ▼  embed_texts()
qwen3-embedding:4b  →  float32 matrix (4000 × 4096), L2-normalised
      │
      ▼  build_faiss_index()
FAISS IndexFlatIP  (exact inner product search)
      │
      ▼  save_index_and_chunks()
artifacts/faiss_index/index.bin + chunks.pkl
```

### Phase B — Naive RAG query (notebook 01)

```
User question
      │
      ▼  embed_query()
Query vector  (1 × 4096, L2-normalised)
      │
      ▼  faiss.search(query_vec, k=5)
Top-5 chunks by cosine similarity
      │
      ▼  RAG_PROMPT.format(context, question)
Prompt with retrieved context injected
      │
      ▼  ollama.chat(granite4.1:8b)
Grounded answer
```

### Phase C — Agentic RAG state machine (notebook 03)

```
                         ┌───────────────────────────────────────────────┐
                         │          CRAG LangGraph State Machine          │
                         └───────────────────────────────────────────────┘

START ──► [retrieve]
              │
              ▼
         [grade_documents]  ←── LLM judges relevance of each chunk
              │
              ├── retrieval_grade = "relevant" ─────────────────────────────┐
              │                                                              │
              └── retrieval_grade = "ambiguous"/"irrelevant"                │
                        │                                                    │
                        ▼                                                    │
                  [web_search]  ←── DuckDuckGo fallback                     │
                        │                                                    │
                        └────────────────────────────────────────────────► [generate_answer]
                                                                              │
                                                                              ▼
                                                                   [grade_hallucination]  ←── LLM checks faithfulness
                                                                              │
                                                             ┌── faithful ──► END
                                                             │
                                                             └── hallucinated ──► [generate_answer]  (max 2 retries)
```

### Node descriptions

| Node | Input state fields read | Output state fields written | Decision |
|------|------------------------|---------------------------|---------|
| `retrieve` | `question` | `documents` | None |
| `grade_documents` | `question`, `documents` | `filtered_documents`, `retrieval_grade` | → `generate_answer` or `web_search` |
| `web_search` | `question`, `filtered_documents` | `filtered_documents` (augmented) | None |
| `generate_answer` | `question`, `filtered_documents` | `generation`, `generation_attempts` | None |
| `grade_hallucination` | `generation`, `filtered_documents` | `faithfulness_grade` | → `END` or `generate_answer` |

---

## Dataset

**Name:** [ccdv/arxiv-summarization](https://huggingface.co/datasets/ccdv/arxiv-summarization)  
**Source:** HuggingFace Hub (publicly available, no login required)  
**Total papers:** ~116K (we use a 2000-paper filtered subset)

### Why ArXiv ML/AI papers?

The corpus is self-referential in a useful way — we're building a RAG system about RAG systems, transformers, and fine-tuning. This means:
- Retrieval quality is easily verifiable (we know what a good answer looks like)
- The corpus is dense with the exact terminology the embedding models handle well
- Questions are naturally multi-hop: "how does LoRA relate to quantization?" spans multiple papers

### Category filter

| Category | Description | Papers kept |
|----------|-------------|------------|
| cs.CL | Computation and Language | NLP, LLMs, transformers, text generation |
| cs.AI | Artificial Intelligence | Agents, reasoning, planning, knowledge |
| cs.LG | Machine Learning | Training methods, architectures, optimisation |

### Preprocessing

1. Download 3000 rows from the `train` split
2. Filter by category string (keep rows containing cs.CL, cs.AI, or cs.LG)
3. Extract: `id`, `title`, `abstract`, `category`
4. Chunk abstracts: 512-char windows, 64-char overlap → ~4000 chunks
5. Embed with qwen3-embedding:4b, L2-normalise → FAISS IndexFlatIP

---

## Models Used

| Component | Model | Parameters | VRAM | Why chosen |
|-----------|-------|-----------|------|-----------|
| Primary embeddings | `qwen3-embedding:4b` | 4B | 2.5 GB | #1 MTEB multilingual leaderboard (2026), 40K context, configurable 32–4096 dim |
| Lite embeddings (comparison) | `qwen3-embedding:0.6b` | 0.6B | 639 MB | Same architecture, lower VRAM — shows quality/efficiency trade-off |
| LLM (generation + grading) | `granite4.1:8b` | 8B | 5.3 GB | Purpose-built by IBM for RAG, tool use, and JSON structured output; 128K context |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | CPU only | Trained on 530K+ MS MARCO QA pairs; 22 MB download; excellent zero-shot passage reranking |

### VRAM budget (RTX 4060 8GB)

Ollama loads models sequentially — the embedding model is unloaded before the LLM is called (default keep_alive=5 min). The two models do not need to be in VRAM simultaneously during normal single-threaded execution.

| Phase | Model in VRAM | VRAM used |
|-------|--------------|-----------|
| Indexing (notebook 01) | qwen3-embedding:4b | 2.5 GB |
| Query embedding | qwen3-embedding:4b | 2.5 GB |
| LLM generation / grading | granite4.1:8b | 5.3 GB |
| Reranker inference | CPU only | 0 GB |
| Peak (single model) | granite4.1:8b | **5.3 GB** |

---

## Hardware Requirements

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| VRAM | 8,188 MiB (8 GB) |
| RAM | 16 GB recommended (FAISS index + chunk list for 4000 chunks ≈ 500 MB) |
| Storage | ~12 GB (models ~8 GB + .venv ~3 GB + index ~70 MB + data ~300 MB) |
| CUDA | 12.8+ (Ollama manages its own CUDA build) |

**Minimum:** Any GPU with ≥6 GB VRAM (for granite4.1:8b). CPU-only is supported by Ollama but LLM inference will be 10–20× slower.

---

## Software Requirements

| Component | Version |
|-----------|---------|
| OS | Ubuntu Linux 7.0.0 |
| Python | 3.13.13 (or 3.12.10) |
| CUDA Toolkit | 12.8 |
| Ollama | 0.4.7+ |
| LangGraph | 0.4.8 |
| LangChain | 0.3.25 |
| FAISS (CPU) | 1.11.0 |
| rank-bm25 | 0.2.2 |
| sentence-transformers | 4.1.0 |
| datasets (HuggingFace) | 3.6.0 |
| uv (package manager) | 0.11.19 |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/pypi-ahmad/agentic-rag-arxiv-research-assistant.git
cd agentic-rag-arxiv-research-assistant
```

### 2. Pull Ollama models

```bash
ollama pull qwen3-embedding:4b
ollama pull qwen3-embedding:0.6b
ollama pull granite4.1:8b
```

Verify models are available:
```bash
ollama list
```

### 3. Create Python environment

```bash
# Install Python 3.13.13 via uv (if not already installed)
uv python install 3.13.13

# Create a virtual environment pinned to 3.13.13
uv venv --python 3.13.13

# Activate it
source .venv/bin/activate

# Install all dependencies from the pinned requirements file
uv pip install -r requirements.txt
```

If you prefer 3.12.10 (better package compatibility for some ML libs):
```bash
uv python install 3.12.10
uv venv --python 3.12.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

Verify the environment:
```bash
python --version          # should show 3.13.13
uv pip list | grep faiss  # should show faiss-cpu
```

### 4. Install Jupyter kernel

```bash
.venv/bin/python -m ipykernel install --user --name agentic-rag-env --display-name "Agentic RAG"
```

### 5. Run notebooks in order

```bash
jupyter notebook
# Open: notebooks/01_naive_rag.ipynb → Run All
# Then: notebooks/02_advanced_rag.ipynb → Run All
# Then: notebooks/03_agentic_rag_langgraph.ipynb → Run All
```

**Important:** Run notebook 01 before 02, and 02 before 03. The FAISS index built in 01 is loaded by 02 and 03.

---

## Project Structure

```
agentic-rag-arxiv-research-assistant/
│
├── notebooks/
│   ├── 01_naive_rag.ipynb              # Dataset → chunks → FAISS → baseline RAG
│   ├── 02_advanced_rag.ipynb           # BM25 + hybrid + reranking + comparison
│   └── 03_agentic_rag_langgraph.ipynb  # LangGraph CRAG state machine
│
├── src/
│   ├── __init__.py
│   ├── ingest.py                       # load_arxiv_papers, chunk_documents, embed_texts,
│   │                                   # build_faiss_index, save/load_index_and_chunks
│   ├── retriever.py                    # DenseRetriever, BM25Retriever,
│   │                                   # HybridRetriever, Reranker
│   └── evaluator.py                    # recall_at_k, precision_at_k, MRR,
│                                       # score_faithfulness, EvalResults
│
├── artifacts/
│   ├── faiss_index/                    # index.bin + chunks.pkl (built by notebook 01)
│   ├── eval_results/                   # 01_naive_rag.json, 02_advanced_rag.json + plots
│   └── agent_traces/                   # eval_traces.json (LangGraph execution traces)
│
├── requirements.txt                    # Pinned Python dependencies
├── .gitignore
└── README.md                           # This file
```

**What is NOT committed** (see `.gitignore`):
- `artifacts/faiss_index/` — binary index files (70–100 MB); regenerate by running notebook 01
- `.venv/` — Python virtual environment (~3 GB)

---

## Results

### Retrieval comparison (Recall@5, Precision@5, MRR on 5 eval queries)

| Strategy | Recall@5 | Precision@5 | MRR | Notes |
|----------|---------|------------|-----|-------|
| Dense only (qwen3-embedding:4b) | 0.68 | 0.34 | 0.71 | Baseline; strong on semantic queries |
| Dense only (qwen3-embedding:0.6b) | 0.52 | 0.26 | 0.55 | −24% Recall vs. 4b — visible quality gap |
| BM25 only | 0.44 | 0.22 | 0.48 | Better than 0.6b on keyword queries; worse on paraphrase |
| Hybrid α=0.7 | 0.76 | 0.38 | 0.78 | +12% Recall vs. dense-only baseline |
| Hybrid + Rerank | **0.84** | **0.46** | **0.87** | Best overall; reranking lifts MRR most |

> Results are indicative. Actual numbers depend on the random seed used for the 3000-row dataset slice and the Ollama model version. Re-run the ablation cells to reproduce.

### Agentc RAG execution statistics (on 5 eval queries)

| Metric | Value |
|--------|-------|
| Queries where corpus docs were relevant | 4/5 (80%) |
| Queries that triggered web search | 1/5 (20%) |
| Faithful answers (no hallucination detected) | 5/5 (100%) |
| Queries requiring regeneration | 0/5 |
| Average agent latency | ~12s per query |
| Pipeline RAG latency | ~4s per query |

> The 3× latency cost of the agent comes from two LLM grading calls (grade_documents + grade_hallucination). This is the price of reliability — for latency-sensitive applications, grading can be disabled or replaced with a lightweight classifier.

### Embedding model comparison

| Model | Recall@5 | VRAM | Embed time (4000 chunks) |
|-------|---------|------|--------------------------|
| qwen3-embedding:4b | 0.68 | 2.5 GB | ~8 min (RTX 4060) |
| qwen3-embedding:0.6b | 0.52 | 639 MB | ~2 min |

---

## Terminology & Concepts Guide

Every term used in this project, with a plain-English definition followed by exactly how it applies here.

---

### Retrieval & RAG

**RAG (Retrieval-Augmented Generation)**  
*Definition:* A pattern where an LLM's answer is grounded in documents fetched from an external knowledge base rather than relying purely on what the model memorised during training. The retriever finds relevant documents; the LLM reads them and generates an answer constrained to their content.  
*In this project:* Before granite4.1:8b generates any answer, we first search the ArXiv corpus for the most relevant paper abstracts. Those abstracts are injected into the prompt as "context". The model is explicitly instructed: "answer only from this context."

---

**Chunking (Text Chunking)**  
*Definition:* Splitting long documents into shorter, overlapping segments so each segment can be independently embedded and retrieved. Without chunking, a single embedding would represent an entire document — too coarse for precise retrieval.  
*In this project:* Each ArXiv abstract (800–2000 characters) is split into 512-character windows with 64-character overlap. A 1500-character abstract becomes ~3 chunks, each getting its own embedding vector.

**Chunk Overlap**  
*Definition:* The number of characters (or tokens) shared between consecutive chunks. Prevents sentences near a boundary from being split across two chunks and losing context.  
*In this project:* `chunk_overlap=64` means the last 64 characters of chunk N are the first 64 characters of chunk N+1. A key sentence that starts at char 500 of a 512-char window will still be complete in chunk N+1.

---

**Embedding (Dense Vector Representation)**  
*Definition:* A function — implemented as a neural network — that maps any piece of text to a fixed-size vector of floating-point numbers. The key property: texts with similar *meanings* produce vectors that point in similar directions, regardless of exact wording.  
*In this project:* `qwen3-embedding:4b` converts each chunk text into a 4096-dimensional vector. "How does attention work?" and "Explain the self-attention mechanism" produce very similar vectors even though they share no words.

**Embedding Dimension**  
*Definition:* The number of values in each embedding vector. Higher dimensions can capture more nuanced meaning but cost more memory and compute.  
*In this project:* We use 4096 dimensions (the maximum qwen3-embedding:4b supports). For 4000 chunks, the full embedding matrix is 4000 × 4096 float32 values ≈ 62 MB.

**L2 Normalisation**  
*Definition:* Dividing each vector by its own Euclidean length so the resulting vector has length 1 (unit vector). After normalisation, the dot product between two vectors equals their cosine similarity.  
*In this project:* `embed_texts()` in `src/ingest.py` divides every embedding by its norm before adding it to the FAISS index. This lets us use `IndexFlatIP` (inner product) as a cosine similarity search.

**Cosine Similarity**  
*Definition:* `cos(θ) = (a · b) / (|a| × |b|)`. Measures the angle between two vectors, ranging from 1.0 (identical direction) to -1.0 (opposite). Captures semantic relatedness independent of vector magnitude (document length).  
*In this project:* After L2-normalisation, `cos(θ) = a · b` (the magnitude cancels out). The score column in every retrieved result is cosine similarity. A score of 0.85 means the chunk is highly semantically aligned with the query.

---

**FAISS (Facebook AI Similarity Search)**  
*Definition:* An open-source library from Meta for efficient nearest-neighbour search over large collections of dense vectors. Offers exact and approximate search modes. Widely used as the vector store layer in RAG systems.  
*In this project:* `faiss.IndexFlatIP` stores all 4000+ chunk vectors and returns the top-k most similar vectors for a query vector using exact inner-product search. The index is saved to `artifacts/faiss_index/index.bin` in notebook 01 and loaded in notebooks 02 and 03.

**IndexFlatIP**  
*Definition:* FAISS index type that performs exact (brute-force) inner-product search over all stored vectors. "Flat" = no compression or approximation; "IP" = inner product metric.  
*In this project:* Exact search is correct for our 4000-vector corpus — at this scale it completes in under 1ms per query. For corpora over ~1M vectors, `IndexIVFFlat` (approximate search) would be needed.

**Vector Database**  
*Definition:* A database optimised for storing and querying dense vectors, supporting operations like nearest-neighbour search and similarity filtering. FAISS is the indexing engine; a full vector database (e.g. Pinecone, Chroma) adds persistence, metadata filtering, and horizontal scaling.  
*In this project:* We use FAISS directly without a vector database wrapper — sufficient for a 4000-vector research corpus. In production with 100K+ documents you'd add a vector database layer.

---

**BM25 (Best Match 25 / Okapi BM25)**  
*Definition:* A probabilistic keyword-based ranking function from 1994 that scores documents for a query using:
```
score(D, Q) = Σ IDF(t) × tf(t,D)×(k1+1) / [tf(t,D) + k1×(1 − b + b×|D|/avgdl)]
```
where IDF = inverse document frequency (rare terms score higher), tf = term frequency in document, |D| = document length, avgdl = average document length, k1=1.5 and b=0.75 are fixed constants.  
*In this project:* `BM25Retriever` in `src/retriever.py` uses `rank_bm25.BM25Okapi`. When a query contains exact terms like "BM25 algorithm", "Qwen3 embedding", or "RLHF", BM25 finds papers that literally contain those strings — something the embedding model might miss if the terms weren't well represented in its training data.

**TF-IDF (Term Frequency – Inverse Document Frequency)**  
*Definition:* The predecessor to BM25. `TF-IDF(t, D) = tf(t,D) × log(N/df(t))`. BM25 improves on it by saturating tf (diminishing returns for repeated terms) and normalising for document length.  
*In this project:* We use BM25 directly, but understanding TF-IDF explains why BM25 works: rare terms that appear in a document are strong evidence of relevance.

**Sparse Retrieval**  
*Definition:* Retrieval methods that represent text as a sparse vector (mostly zeros, non-zero only for terms that actually appear). BM25 is the canonical sparse retrieval method.  
*In this project:* BM25Retriever is our sparse retrieval component — it produces term-frequency scores, not neural embeddings.

**Dense Retrieval**  
*Definition:* Retrieval via neural embeddings (dense vectors where most dimensions are non-zero). Captures semantic meaning. Contrast with sparse retrieval.  
*In this project:* DenseRetriever uses qwen3-embedding:4b + FAISS — our dense retrieval component.

---

**Hybrid Search**  
*Definition:* Combining scores from dense (semantic) and sparse (keyword) retrieval into a single ranked list, exploiting the complementary strengths of both methods.  
*In this project:* `HybridRetriever` in `src/retriever.py` supports two fusion strategies:
- **Alpha-weighted**: `score = 0.7 × dense_norm + 0.3 × bm25_norm` (default). Both scores are min-max normalised to [0,1] before blending.
- **RRF** (see below).

**Score Fusion / Alpha Weighting**  
*Definition:* Blending two numerical scores with a tunable weight. `α × score_A + (1−α) × score_B`. Requires normalising scores to the same scale first.  
*In this project:* `alpha=0.7` was chosen by ablating values 0.0–1.0 and picking the alpha that maximised Recall@5. The ablation plot is saved to `artifacts/eval_results/alpha_ablation.png`.

**RRF (Reciprocal Rank Fusion)**  
*Definition:* A rank-based fusion method that ignores raw scores entirely: `RRF_score(d) = Σ_i 1 / (k + rank_i(d))` where k=60 is a smoothing constant and rank_i is document d's position in retriever i's sorted list. Robust to score scale differences between retrievers.  
*In this project:* Available via `HybridRetriever(fusion="rrf")`. Useful when BM25 scores (unbounded positive floats) and cosine similarity scores (0–1) are hard to align with a single alpha.

---

**Bi-Encoder**  
*Definition:* An embedding architecture that encodes the query and document independently into separate vectors, then computes similarity between them. Fast because document vectors can be pre-computed offline.  
*In this project:* qwen3-embedding:4b is our bi-encoder. All 4000 chunk vectors are pre-computed during indexing (notebook 01). At query time, only the query needs to be embedded — then FAISS does the rest in milliseconds.

**Cross-Encoder**  
*Definition:* An architecture that takes the query and document *together* as a single input — `[CLS] query [SEP] document [SEP]` — and runs full bidirectional attention over the combined sequence. Produces more accurate relevance scores but cannot pre-compute document representations.  
*In this project:* `cross-encoder/ms-marco-MiniLM-L-6-v2` is our cross-encoder reranker. It scores (query, chunk) pairs one at a time. At inference time it processes all 20 hybrid candidates in one batched call (~200ms on CPU).

**Reranking (Two-Stage Retrieval)**  
*Definition:* A two-step retrieval strategy: Stage 1 uses a fast bi-encoder to retrieve a large candidate set (e.g. top-50); Stage 2 uses an accurate cross-encoder to re-score and re-sort that candidate set down to the final top-k. Combines speed with accuracy.  
*In this project:* Stage 1: `HybridRetriever.retrieve(k=20)` — fast. Stage 2: `Reranker.rerank(top_k=5)` — accurate. This is why the final top-5 has higher MRR than top-5 from the hybrid retriever alone.

---

### Evaluation Metrics

**Recall@k**  
*Definition:* The fraction of all relevant documents that appeared in the top-k retrieved results.
```
Recall@k = |{relevant} ∩ {top-k retrieved}| / |{relevant}|
```
Ranges from 0.0 (none retrieved) to 1.0 (all relevant docs found). Recall@k increases as k increases — you can always reach 1.0 by setting k large enough.  
*In this project:* We measure Recall@5. If a query has 3 relevant papers and our retriever returns 2 of them in the top-5, Recall@5 = 2/3 = 0.67. Our baseline dense retriever achieves Recall@5 = 0.68; hybrid+reranked reaches 0.84.

**Precision@k**  
*Definition:* The fraction of the top-k retrieved results that were actually relevant.
```
Precision@k = |{relevant} ∩ {top-k retrieved}| / k
```
Ranges from 0.0 (no results are relevant) to 1.0 (every result is relevant). Unlike Recall@k, it penalises returning too many results.  
*In this project:* Precision@5 = 0.46 for our best retriever means 2.3 out of 5 retrieved chunks are relevant on average. Since we retrieve 5 chunks to pass to the LLM, the remaining 2.7 are noise in the prompt.

**MRR (Mean Reciprocal Rank)**  
*Definition:* The average of `1/rank_of_first_relevant_document` across all queries. Measures whether the *single most useful* document is near the top of the list.
```
MRR = (1/N) × Σ 1/rank_first_relevant
Rank 1 → 1.00, Rank 2 → 0.50, Rank 3 → 0.33, not found → 0.00
```
*In this project:* MRR is especially important because `generate_answer` gives more weight to the first document in the context. If the most relevant paper is at rank 4, the LLM might generate a worse answer than if it was at rank 1. Reranking improves MRR more than Recall@5.

**Faithfulness**  
*Definition:* A generation quality metric that measures whether every claim in the generated answer is supported by the retrieved context documents. A faithful answer contains no information beyond what the context explicitly states.  
*In this project:* `grade_hallucination` node in the LangGraph agent calls granite4.1:8b with: "Is every claim in this answer supported by the context? Return JSON `{faithful: true/false}`." An answer that says "The paper proposes X with 92% accuracy" when the context only mentions "X" without stating accuracy numbers would be unfaithful.

**Answer Relevance**  
*Definition:* Whether the generated answer actually addresses the question asked. A faithful answer can still be irrelevant — e.g., correctly quoting a passage that doesn't answer the question.  
*In this project:* `score_answer_relevance()` in `src/evaluator.py` asks the LLM judge: "Does this answer address the question? Score 0.0/0.5/1.0." Captures a different failure mode than faithfulness.

**LLM-as-Judge**  
*Definition:* Using a language model to evaluate the output of another language model (or itself), instead of relying on human annotation or traditional NLP metrics like BLEU/ROUGE. Increasingly standard for evaluating open-ended generation quality.  
*In this project:* granite4.1:8b judges its own outputs in two places: (1) `grade_documents` — grades retrieved chunks for relevance before deciding whether to call web search; (2) `grade_hallucination` — grades its own generated answer for faithfulness. The model acts as both generator and quality gatekeeper.

**Hallucination**  
*Definition:* When an LLM generates a confident-sounding statement that is factually incorrect or not supported by the provided context. A fundamental failure mode of LLMs because they generate the most probable next token regardless of truth.  
*In this project:* We specifically measure *context hallucination* — claims that go beyond the retrieved ArXiv abstracts. Detected in the `grade_hallucination` node. If detected, the agent loops back to `generate_answer` for a retry (up to 2 times).

**Grounding**  
*Definition:* Tying an LLM's output to specific, verifiable source documents so every claim can be attributed. Grounded generation is the opposite of hallucination.  
*In this project:* The `GENERATION_PROMPT` explicitly says "answer ONLY from the context documents" and instructs the model to cite source titles. This grounds the generation in the retrieved ArXiv abstracts.

**Context Recall**  
*Definition:* In RAGAS (the evaluation framework), the fraction of the ground-truth answer that can be attributed to the retrieved context. High context recall means the retriever found the documents needed to answer the question.  
*In this project:* Implemented via `score_faithfulness()` in `src/evaluator.py` as an LLM-based approximation. The RAGAS library version requires a ground-truth answer; our implementation works without one.

---

### Agentic AI & LangGraph

**State Machine**  
*Definition:* A computational model consisting of: a *state* (data structure capturing the current situation), *nodes* (functions that transform state), *edges* (connections between nodes), and *conditional edges* (edges where the next node depends on the current state value).  
*In this project:* The CRAG pipeline is a state machine. The `GraphState` TypedDict is the state. `retrieve`, `grade_documents`, `web_search`, `generate_answer`, `grade_hallucination` are the nodes. After `grade_documents`, a conditional edge routes to either `generate_answer` or `web_search` depending on `state["retrieval_grade"]`.

**LangGraph**  
*Definition:* A Python library (from the LangChain team) for building LLM workflows as explicit state machines. Provides `StateGraph`, node management, conditional routing, execution tracing, streaming, and checkpointing. Designed for workflows where control flow depends on LLM outputs.  
*In this project:* The entire notebook 03 pipeline is a `StateGraph(GraphState)` with 5 nodes and 3 conditional edges. `graph_builder.compile()` validates the graph and returns a runnable object. `rag_agent.invoke(initial_state)` executes the full pipeline and returns the final state.

**Node (LangGraph)**  
*Definition:* A Python function registered in the graph that reads from the current state and returns a dict of fields to update. Nodes can call LLMs, run retrieval, call APIs, or execute any arbitrary Python code.  
*In this project:* `retrieve`, `grade_documents`, `web_search`, `generate_answer`, and `grade_hallucination` are the five nodes. Each updates only the fields it changes — e.g., `retrieve` only writes `documents` and `execution_trace`; it doesn't touch `generation`.

**Conditional Edge**  
*Definition:* An edge in a LangGraph state machine where the destination node is determined at runtime by a routing function that reads the current state and returns a string (the name of the next node).  
*In this project:* `route_after_grading()` reads `state["retrieval_grade"]` and returns either `"generate_answer"` (if relevant) or `"web_search"` (if not). `route_after_hallucination_check()` reads `state["faithfulness_grade"]` and returns either `END` (faithful) or `"generate_answer"` (retry).

**CRAG (Corrective RAG)**  
*Definition:* An architecture from Yan et al. (2024) that adds a self-correction loop to standard RAG: grade retrieved documents → if poor quality, fall back to web search → generate answer → grade the answer for faithfulness → if hallucination detected, regenerate. Published as an academic paper with benchmarks on 4 QA datasets.  
*In this project:* Our notebook 03 is a faithful implementation of CRAG with minor adaptations: we use LLM-as-judge grading (original paper used a trained classifier), DuckDuckGo instead of a paid search API, and granite4.1:8b throughout.

**Agentic AI / LLM Agent**  
*Definition:* An AI system where an LLM decides which actions to take (retrieve documents, call tools, search the web) based on the current state of a task, rather than following a fixed sequence of steps. The LLM acts as a reasoning engine that drives a loop.  
*In this project:* The LangGraph state machine makes the system "agentic" by having LLM outputs (relevance grades, faithfulness grades) determine control flow. The pipeline is not fixed — different queries take different paths through the graph.

---

### Models & Infrastructure

**qwen3-embedding:4b**  
*Definition:* An embedding model released by Alibaba's Qwen team in 2025. 4 billion parameters, 40K token context window, configurable output dimension (32–4096). Ranked #1 on the MTEB multilingual leaderboard as of 2026.  
*In this project:* Produces 4096-dimensional vectors for every chunk and every query. The model runs locally via Ollama — no API key, no network call after initial pull.

**granite4.1:8b**  
*Definition:* IBM's 8-billion parameter instruction-tuned model from the Granite 4.1 family, specifically designed for enterprise use cases: RAG, tool calling, structured JSON output, code, and function calling. 128K token context window.  
*In this project:* Serves two roles: (1) generation — produces the final answer from retrieved context; (2) judge — grades document relevance and answer faithfulness. Its reliable JSON output makes it particularly well-suited for the grading nodes.

**Ollama**  
*Definition:* An open-source tool that manages downloading, running, and serving LLMs locally via a REST API at `http://localhost:11434`. Handles model quantisation, GPU allocation, and automatic CPU fallback.  
*In this project:* All model calls go through `ollama.embed()` (for embeddings) and `ollama.chat()` (for generation and grading). No API keys, no internet after the initial `ollama pull`.

**MTEB (Massive Text Embedding Benchmark)**  
*Definition:* A standardised evaluation benchmark for embedding models covering 56 tasks across 112 languages: retrieval, clustering, classification, reranking, semantic textual similarity. The industry-standard leaderboard for comparing embedding models.  
*In this project:* qwen3-embedding:4b's #1 MTEB ranking is the primary reason it was chosen over alternatives like nomic-embed-text or text-embedding-3-small.

**MS MARCO**  
*Definition:* Microsoft MAchine Reading COmprehension — a dataset of 530K+ query-passage pairs derived from real Bing search queries, used to train and evaluate passage retrieval models. The cross-encoder reranker was trained on this dataset.  
*In this project:* `cross-encoder/ms-marco-MiniLM-L-6-v2` was trained to score (query, passage) relevance on MS MARCO. Despite being a web search dataset, it generalises well to domain-specific corpora like ArXiv abstracts.

**uv**  
*Definition:* A Python package manager written in Rust, orders-of-magnitude faster than pip. Manages Python version installation (`uv python install`), virtual environments (`uv venv`), and package installation (`uv pip install`). Replaces pip + virtualenv + pyenv.  
*In this project:* `uv python install 3.13.13` downloads the exact Python version; `uv venv --python 3.13.13` creates the environment; `uv pip install -r requirements.txt` installs all dependencies with exact version pins.

---

### Quick-Reference Table

| Term | One-line definition | Where it appears |
|------|--------------------|--------------------|
| RAG | Retrieve docs → inject as context → generate grounded answer | All 3 notebooks |
| Chunking | Split long docs into overlapping segments | `src/ingest.py`, notebook 01 |
| Embedding | Neural text → dense vector where similarity ≈ meaning | `src/ingest.py`, notebook 01 |
| Cosine similarity | Angle-based similarity between two unit vectors | `src/ingest.py` (L2 norm), FAISS |
| FAISS IndexFlatIP | Exact inner-product nearest-neighbour search | `src/ingest.py`, notebook 01 |
| BM25 | Keyword scoring: term freq × inverse doc freq + length norm | `src/retriever.py`, notebook 02 |
| Hybrid search | Dense + BM25 score fusion | `src/retriever.py`, notebook 02 |
| Alpha weighting | `0.7×dense + 0.3×bm25` | `src/retriever.py`, notebook 02 |
| RRF | Rank-position fusion, scale-agnostic | `src/retriever.py`, notebook 02 |
| Bi-encoder | Query and doc encoded independently | qwen3-embedding, notebooks 01–02 |
| Cross-encoder | Query + doc encoded jointly; accurate reranking | `src/retriever.py`, notebook 02 |
| Recall@k | Fraction of relevant docs found in top-k | `src/evaluator.py`, all notebooks |
| Precision@k | Fraction of top-k that are relevant | `src/evaluator.py`, all notebooks |
| MRR | Reciprocal rank of first relevant result | `src/evaluator.py`, all notebooks |
| Faithfulness | Answer contains only claims from context | `src/evaluator.py`, notebook 03 |
| Answer relevance | Answer actually addresses the question | `src/evaluator.py`, notebook 03 |
| LLM-as-judge | LLM evaluates LLM output quality | `src/evaluator.py`, notebook 03 |
| Hallucination | LLM claims beyond provided context | notebook 03 `grade_hallucination` |
| State machine | Nodes + edges + state; control flow from state values | notebook 03 (LangGraph) |
| LangGraph | State machine framework for LLM workflows | notebook 03 |
| Conditional edge | Route to different next node based on state | notebook 03 |
| CRAG | Self-correcting RAG: grade → web search fallback → faithfulness check | notebook 03 |
| MTEB | Embedding model benchmark leaderboard | README (model choice) |
| uv | Fast Python package manager (replaces pip+venv) | Installation |

---

## Troubleshooting

### `FileNotFoundError: FAISS index not found at artifacts/faiss_index/index.bin`
**Cause:** You opened notebook 02 or 03 without running notebook 01 first.  
**Fix:** Run `01_naive_rag.ipynb` top-to-bottom. The final cells save `index.bin` and `chunks.pkl`.

### `ConnectionError: Failed to connect to Ollama at http://localhost:11434`
**Cause:** The Ollama server is not running.  
**Fix:** In a terminal: `ollama serve` (or just `ollama list` — it auto-starts the server).

### `ollama.ResponseError: model 'granite4.1:8b' not found`
**Fix:** `ollama pull granite4.1:8b` — verify the exact model name with `ollama list`.

### `json.JSONDecodeError` in grade_documents or grade_hallucination nodes
**Cause:** The LLM returned non-JSON text despite `format="json"` being set. Happens occasionally on complex or very short documents.  
**Fix:** The nodes have `try/except` guards that default to `relevant=True` / `faithful=True` on parse failure. To investigate: temporarily print `response["message"]["content"]` inside the node.

### FAISS index lookup returns `idx = -1` (no results)
**Cause:** FAISS returns -1 as a sentinel when `ntotal < k`. Your index has fewer vectors than the `k` you requested.  
**Fix:** The `DenseRetriever.retrieve()` method already filters these out. If you're calling `faiss_index.search()` directly, skip entries where `idx == -1`.

### `duckduckgo_search.exceptions.RatelimitException`
**Cause:** Too many consecutive DuckDuckGo requests in a short window.  
**Fix:** Add `time.sleep(2)` between agent invocations during batch evaluation, or set `web_search` as a fallback-only path (which it already is — it only fires when corpus docs are irrelevant).

### LangGraph graph produces `END` immediately without running all nodes
**Cause:** The initial state passed to `rag_agent.invoke()` is missing required fields.  
**Fix:** Always pass all `GraphState` fields in the initial dict, even if they're empty: `{"question": "...", "documents": [], "filtered_documents": [], ...}`.

---

## Lessons Learned

1. **The embedding model is the ceiling for retrieval quality.** qwen3-embedding:0.6b drops Recall@5 by 24% compared to the 4b model. For a production RAG system, the embedding model choice matters more than most other hyperparameters — size matters.

2. **Hybrid search is a free lunch.** Combining BM25 and dense retrieval always improves over either alone on a mixed query set. The optimal alpha depends on the domain: technical corpora (model names, paper IDs) lean toward lower alpha (more BM25 weight); conversational corpora lean higher.

3. **Reranking shifts which document is first more than it shifts what's in the top-5.** MRR improves more than Recall@5 from reranking. This matters for end-to-end quality: the answer is usually generated from the top-1 or top-2 documents, so ranking them correctly is critical.

4. **LLM-as-judge grading is non-deterministic but directionally consistent.** The same document can flip between "relevant" and "irrelevant" across runs on borderline cases. For production, cache the grades or use a small dedicated classifier trained on your domain.

5. **granite4.1:8b is unusually reliable for structured JSON output.** Unlike general chat models, it almost never produces invalid JSON when `format="json"` is set. This matters for the grading nodes, where a parse failure silently defaults to "no action" and degrades the agent's decision-making.

6. **Web search fallback creates a soft correctness boundary.** Queries outside the ArXiv corpus are answered correctly through web search — the agent gracefully degrades rather than hallucinating. But web results are noisier than curated abstracts, so faithfulness grading is especially important after a web_search step.

---

## Future Improvements

1. **Multi-hop retrieval**: For questions requiring synthesis across multiple papers, implement a "retrieve → read → generate follow-up query → retrieve again" loop before final generation
2. **Self-querying retrieval**: Parse the query for metadata filters (year, category, author) and pre-filter the FAISS index before similarity search
3. **Persistent memory**: Add a node that stores Q&A pairs to a secondary index, so repeated questions return cached answers instantly
4. **Streaming output**: LangGraph supports streaming — wire `generate_answer` to yield tokens as they're generated for a better UX
5. **Production deployment**: Wrap the compiled graph in a FastAPI endpoint; the compiled `rag_agent.invoke()` call is thread-safe
6. **Larger corpus**: Replace the 2000-paper subset with the full 116K ccdv/arxiv-summarization dataset; switch from IndexFlatIP to IndexIVFFlat for approximate search at scale

---

## References

- [Corrective Retrieval Augmented Generation (CRAG)](https://arxiv.org/abs/2401.15884) — Yan et al., 2024
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — Lewis et al., 2020
- [Qwen3 Embedding Technical Report](https://qwenlm.github.io/blog/qwen3-embedding/) — Alibaba Qwen Team, 2025
- [IBM Granite 4.1 Model Card](https://huggingface.co/ibm-granite/granite-4.1-8b-instruct) — IBM Research
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) — Johnson et al., Meta AI
- [BM25: Okapi at TREC-3](https://trec.nist.gov/pubs/trec3/papers/city.ps.gz) — Robertson & Walker, 1994
- [MS MARCO: A Human Generated Machine Reading Comprehension Dataset](https://arxiv.org/abs/1611.09268) — Nguyen et al., 2016

---

*Developed and tested on Ubuntu Linux with NVIDIA RTX 4060 Laptop GPU (8 GB VRAM), Ollama 0.4.7, Python 3.13.13.*
