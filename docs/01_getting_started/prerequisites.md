# Prerequisites

This page tells you exactly what you need — in terms of knowledge, tools, and hardware — before you open the first notebook. Nothing here is aspirational; these are the actual requirements tested on the machine where this project was built.

---

## Knowledge prerequisites

You do **not** need prior experience with RAG, LangChain, LangGraph, FAISS, or BM25. All of these are explained from scratch.

Here is what you *do* need:

| Concept | Level required | Why it matters |
|---------|----------------|----------------|
| Python — functions, classes, dicts, list comprehensions | Comfortable writing it | All code is Python |
| NumPy arrays | `np.array([1, 2, 3])` level is enough | Embeddings are float arrays |
| What an API call is (send a request, get a response) | Basic familiarity | We call the local Ollama REST API |
| What a neural network does at a high level | "maps input → output" is sufficient | Understanding embeddings and LLMs |
| Terminal, file paths, Git | Basic navigation | Installation and running notebooks |

You do **not** need:

- Linear algebra beyond understanding that dot product ≈ similarity
- Transformer architecture internals or attention math
- Prior LangChain or LangGraph experience
- Cloud accounts of any kind
- A formal ML or CS background

!!! tip "If you're fuzzy on embeddings"
    The Core Concepts section covers [Embeddings & Similarity](../02_core_concepts/embeddings.md) in detail before you encounter them in the notebooks. Read that page if the word "embedding" feels vague — you don't need to understand it before starting, but knowing the intuition early will make the notebook cells click faster.

---

## Tool prerequisites

### 1. Ollama

Ollama is the local model runtime. It downloads LLM and embedding models, manages GPU memory, and exposes a simple REST API at `http://localhost:11434`. You talk to it the same way you'd talk to OpenAI's API — but everything stays on your machine.

**Install Ollama:**

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify it's working:

```bash
ollama --version
# ollama version is 0.4.7
```

### 2. Pull the models

Once Ollama is installed, pull the two models this tutorial uses. These are the exact models tested in the notebooks:

```bash
# Embedding model — used in all three notebooks for encoding chunks and queries
ollama pull qwen3-embedding:0.6b   # ~639 MB

# LLM — used for answer generation and LLM-as-judge grading
ollama pull granite4.1:8b          # ~5.3 GB
```

!!! info "What these models do"
    `qwen3-embedding:0.6b` converts text into a 1024-dimensional vector. It's a lighter version of the full `qwen3-embedding:4b` model — same architecture, lower VRAM, slightly lower retrieval quality. The tutorial uses the 0.6b variant by default to fit on 6 GB VRAM cards.

    `granite4.1:8b` is IBM's instruction-tuned LLM optimised for RAG, structured JSON output, and tool use. It serves as both the answer generator and the judge that grades retrieved documents and generated answers.

The downloads are one-time. After `ollama pull`, the models are cached locally at `~/.ollama/models/`.

Verify the pulls completed:

```bash
ollama list
# NAME                        ID              SIZE    MODIFIED
# granite4.1:8b               ...             5.3 GB  ...
# qwen3-embedding:0.6b        ...             639 MB  ...
```

### 3. Python 3.13

The project is tested on Python 3.13.13. Python 3.12.10 also works.

The recommended way to manage Python versions in this project is `uv` (see below). You do not need to pre-install Python 3.13 separately — `uv` will download it for you.

### 4. uv

`uv` is a Python package manager written in Rust. It replaces `pip`, `virtualenv`, and `pyenv` with a single tool that is orders of magnitude faster than pip.

**Install uv:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify:

```bash
uv --version
# uv 0.11.19 (or later)
```

!!! note "Why uv instead of pip?"
    With pip, creating a virtual environment and installing the full dependency list for this project takes 3–5 minutes. With uv, it takes under 30 seconds. More importantly, `uv python install` lets you pin an exact Python version without needing pyenv or conda — which keeps the environment reproducible across machines.

---

## Hardware requirements

This project was developed and tested on the following hardware. All model VRAM figures are from actual runs.

| Component | Tested configuration | Minimum |
|-----------|---------------------|---------|
| GPU | NVIDIA RTX 4060 Laptop (8 GB VRAM) | Any GPU with ≥ 6 GB VRAM |
| RAM | 16 GB | 8 GB (tight; FAISS index + Python overhead) |
| Storage | ~12 GB free | ~12 GB (models ~8 GB, `.venv` ~3 GB, index ~70 MB, data ~300 MB) |
| CUDA | 12.8 | 12.x (Ollama manages its own CUDA build) |

**VRAM usage breakdown:**

Ollama loads one model at a time and unloads it after 5 minutes of inactivity. The embedding model and the LLM are never in VRAM simultaneously during normal execution.

| Phase | Model active | VRAM used |
|-------|-------------|-----------|
| Indexing (notebook 01) | `qwen3-embedding:0.6b` | ~639 MB |
| LLM generation / grading | `granite4.1:8b` | ~5.3 GB |
| Reranker inference (notebook 02) | CPU only | 0 GB |
| Peak (single model) | `granite4.1:8b` | **~5.3 GB** |

!!! warning "CPU-only fallback"
    Ollama supports CPU-only inference. If you don't have a compatible GPU, everything will still work — but `granite4.1:8b` generation will be 10–20× slower (expect 30–60 seconds per answer instead of 3–5 seconds). The embedding step in notebook 01 (~2100 chunks) will take 15–20 minutes on CPU instead of ~2 minutes on GPU.

    If you're on CPU and want faster iteration, consider `phi4-mini:3.8b` as a lighter LLM alternative — pull it with `ollama pull phi4-mini:3.8b` and swap the model name in the relevant cells.

---

## Software versions (for reference)

These are the exact versions in the tested environment. The `requirements.txt` file pins all of these.

| Package | Version |
|---------|---------|
| Python | 3.13.13 |
| LangGraph | 0.4.8 |
| LangChain | 0.3.25 |
| FAISS (CPU) | 1.11.0 |
| rank-bm25 | 0.2.2 |
| sentence-transformers | 4.1.0 |
| datasets (HuggingFace) | 3.6.0 |
| ollama (Python client) | latest |
| uv | 0.11.19 |

---

Once Ollama is installed, the models are pulled, and uv is ready, you have everything you need. Continue to [Installation](installation.md).
