# Installation

This page walks through every step needed to go from a freshly cloned repository to a running Jupyter environment. Each step includes the exact commands tested on this project's development machine (Ubuntu Linux, RTX 4060, CUDA 12.8).

If you haven't installed Ollama and pulled the models yet, do that first — see [Prerequisites](prerequisites.md).

---

## Step 1 — Clone the repository

```bash
git clone https://github.com/pypi-ahmad/agentic-rag-arxiv-research-assistant.git
cd agentic-rag-arxiv-research-assistant
```

You should now be inside the project directory. Verify with:

```bash
ls
# notebooks/  src/  artifacts/  docs/  requirements.txt  README.md  mkdocs.yml
```

---

## Step 2 — Pull Ollama models (if not done already)

The notebooks call Ollama directly, so the models must be available before any notebook cell runs. If you completed the Prerequisites page, you can skip this step.

```bash
ollama pull qwen3-embedding:0.6b
ollama pull granite4.1:8b
```

Verify both are ready:

```bash
ollama list
```

Expected output:

```
NAME                        ID              SIZE    MODIFIED
granite4.1:8b               abc123...       5.3 GB  a minute ago
qwen3-embedding:0.6b        def456...       639 MB  2 minutes ago
```

!!! note "Ollama server auto-start"
    Running `ollama list` or `ollama pull` automatically starts the Ollama background server if it isn't already running. You don't need to manually run `ollama serve` before opening the notebooks — the Python `ollama` client will reach it at `http://localhost:11434`.

---

## Step 3 — Create the Python environment with uv

`uv` manages the Python version and the virtual environment. Run these commands in the project root (where `requirements.txt` lives):

```bash
# Install Python 3.13.13 via uv (downloads it automatically if not present)
uv python install 3.13.13

# Create a virtual environment pinned to that Python version
uv venv --python 3.13.13

# Activate the environment
source .venv/bin/activate
```

Your shell prompt should now show `(agentic-rag-arxiv-research-assistant)` or similar, indicating the environment is active.

!!! tip "Python 3.12 alternative"
    If you prefer Python 3.12 for better compatibility with some ML libraries:

    ```bash
    uv python install 3.12.10
    uv venv --python 3.12.10
    source .venv/bin/activate
    ```

    Both versions are tested and produce identical results.

---

## Step 4 — Install dependencies

```bash
uv pip install -r requirements.txt
```

This installs all pinned dependencies. On a fast connection with GPU support, this typically takes under 60 seconds with uv.

**Verify the key packages installed correctly:**

```bash
python --version
# Python 3.13.13

uv pip list | grep faiss
# faiss-cpu    1.11.0

uv pip list | grep langgraph
# langgraph    0.4.8

uv pip list | grep rank-bm25
# rank-bm25    0.2.2
```

!!! warning "If you see CUDA-related errors during install"
    This project uses `faiss-cpu` — the CPU-only FAISS build. This is intentional: FAISS runs fast enough on CPU for a 4000-vector index (sub-millisecond per query), and using the CPU build avoids CUDA version conflicts. If `pip` tries to install `faiss-gpu`, something is wrong with your `requirements.txt` — check that you're using the file from the repo root.

---

## Step 5 — Register the Jupyter kernel

The virtual environment needs to be registered as a Jupyter kernel so you can select it from the notebook interface.

```bash
.venv/bin/python -m ipykernel install --user --name agentic-rag-env --display-name "Agentic RAG"
```

Expected output:

```
Installed kernelspec agentic-rag-env in /home/<user>/.local/share/jupyter/kernels/agentic-rag-env
```

You only need to do this once. The kernel persists across sessions.

---

## Step 6 — Launch Jupyter and verify

```bash
jupyter notebook
```

This opens the Jupyter interface in your browser. Navigate to `notebooks/` and open `01_naive_rag.ipynb`.

**Before running any cells, confirm the kernel is set correctly:**

- Look for the kernel name in the top-right corner of the notebook
- It should read **"Agentic RAG"** (the display name you registered)
- If it shows "Python 3" or something else, click on it and select "Agentic RAG" from the dropdown

!!! info "Kernel not appearing?"
    If "Agentic RAG" doesn't appear in the dropdown, run this command and refresh the Jupyter page:

    ```bash
    .venv/bin/python -m ipykernel install --user --name agentic-rag-env --display-name "Agentic RAG"
    ```

    Then in Jupyter: **Kernel → Change Kernel → Agentic RAG**.

---

## Verify the full setup with a quick smoke test

Before running an entire notebook, run this quick check in the first cell to confirm that all the critical pieces are working:

```python
# Paste this into a new notebook cell and run it
import faiss
import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langgraph.graph import StateGraph
import sys

print(f"Python: {sys.version}")
print(f"FAISS: {faiss.__version__}")

# Check Ollama is reachable and models are present
models = ollama.list()
model_names = [m.model for m in models.models]
print(f"Ollama models: {model_names}")

assert any("granite" in m for m in model_names), "granite4.1:8b not found — run: ollama pull granite4.1:8b"
assert any("qwen3-embedding" in m for m in model_names), "qwen3-embedding:0.6b not found — run: ollama pull qwen3-embedding:0.6b"

print("\nAll checks passed. Ready to run the notebooks.")
```

If you see `All checks passed` — you're ready to go.

---

## Project structure (what you're working with)

```
agentic-rag-arxiv-research-assistant/
│
├── notebooks/
│   ├── 01_naive_rag.ipynb              # Build FAISS index, baseline RAG, evaluation
│   ├── 02_advanced_rag.ipynb           # BM25 + hybrid + reranking
│   └── 03_agentic_rag_langgraph.ipynb  # LangGraph CRAG state machine
│
├── src/
│   ├── ingest.py                       # Paper loading, chunking, embedding, FAISS
│   ├── retriever.py                    # Dense, BM25, hybrid, reranker classes
│   └── evaluator.py                    # Recall@k, Precision@k, MRR, faithfulness
│
├── artifacts/
│   ├── faiss_index/                    # index.bin + chunks.pkl (built by notebook 01)
│   ├── eval_results/                   # JSON results + comparison plots
│   └── agent_traces/                   # LangGraph execution traces (notebook 03)
│
├── requirements.txt
└── mkdocs.yml
```

!!! note "The artifacts/ directory is not committed to git"
    `artifacts/faiss_index/` is in `.gitignore` — the binary index files are 70–100 MB and don't belong in git. You regenerate them by running notebook 01. The `eval_results/` plots and JSON files are committed so you can compare your results against the reference run.

---

Continue to [How to Use This Tutorial](how_to_use.md) to understand the notebook run order and how the docs connect to the code.
