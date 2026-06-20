# Agentic RAG — Zero to Hero

Build a production-style RAG assistant over ArXiv ML/AI papers with progressive architectures, reproducible artifacts, and measured outputs.

Current reference scope is **4,000 papers**, with **legacy 600-paper** results preserved for comparison.

---

## Start Here

- Project overview: [What Is This Project](01_getting_started/what_is_this_project.md)
- Setup requirements: [Prerequisites](01_getting_started/prerequisites.md)
- Environment setup: [Installation](01_getting_started/installation.md)
- Guided routes: [Learning Tracks](01_getting_started/learning_tracks.md)
- Full notebook map: [Tutorial Index](01_getting_started/tutorial_index.md)
- Download handbook: [Full Tutorial PDF](01_getting_started/handbook_pdf.md)

---

## Learning Path

```mermaid
graph LR
    A[Part 1<br/>Naive RAG] --> B[Part 2<br/>Advanced RAG]
    B --> C[Part 3<br/>Agentic RAG]
    C --> D[Part 4<br/>Graph RAG]
    D --> E[Part 5<br/>New Techniques]
```

| Part | Notebook(s) | Focus |
|---|---|---|
| Part 1 | `01_naive_rag.ipynb` | Dense FAISS baseline and evaluation |
| Part 2 | `02_advanced_rag.ipynb` | BM25 + hybrid fusion + reranking |
| Part 3 | `03_agentic_rag_langgraph.ipynb` | LangGraph CRAG control loop |
| Part 4 | `04_graph_rag.ipynb` | GraphRAG variants (ChromaDB/Pinecone/agentic graph) |
| Part 5 | `05_hybrid_rag.ipynb` to `09_multimodal_rag.ipynb` | Five standalone advanced techniques |

---

## Runtime Model

### Fully local core path

- Embeddings and generation run locally via Ollama.
- Parts 1-3 and Part 5 can run without cloud dependencies.

### Optional cloud path

- Part 4 includes a Pinecone backend option.
- Pinecone execution requires valid Pinecone credentials.

---

## Two Reader Paths

### Student path

1. [Learning Tracks](01_getting_started/learning_tracks.md)
2. [Tutorial Index](01_getting_started/tutorial_index.md)
3. Part 1 → Part 5 chapters in order

### Recruiter path

1. [Recruiter Brief](07_results/recruiter_brief.md)
2. [Full Benchmark Table](07_results/benchmark_table.md)
3. [Claims Traceability](08_reference/claims_traceability.md)
4. [Full Tutorial PDF](01_getting_started/handbook_pdf.md)

---

## Result Sources

- Current baseline metrics: `artifacts/eval_results/*_4000.json`
- Legacy reference metrics: `artifacts/eval_results/legacy_600/*.json`
- Part 5 metrics: `artifacts/rag_v2/*/*.json`

For interpretation guidance, use:
- [Full Benchmark Table](07_results/benchmark_table.md)
- [Reading the Numbers Honestly](07_results/reading_the_numbers.md)
- [Recruiter Brief](07_results/recruiter_brief.md)
