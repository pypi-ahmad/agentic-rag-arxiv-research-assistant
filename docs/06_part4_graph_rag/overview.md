# Overview

Part 4 upgrades the tutorial from chunk-level retrieval to **graph-aware retrieval** and
evaluates three systems on a shared **4,000-paper** ML/AI corpus:

1. GraphRAG with **ChromaDB** (local persistent vector store)
2. GraphRAG with **Pinecone** (managed serverless vector store)
3. **Agentic GraphRAG** with LangGraph routing and hallucination grading

This part keeps the historical 600-paper metrics from Parts 1-3 as a legacy baseline,
then adds a new 4,000-paper benchmark so readers can compare scale effects directly.

## What is new compared to Part 3

- Entity extraction from every paper abstract
- Knowledge graph construction (`paper ↔ entity`, plus `entity ↔ entity` co-occurrence)
- Community detection + community summarization
- Local graph search and global community search
- ChromaDB/Pinecone backend swap under a shared interface
- Agentic routing across local retrieval, global retrieval, and web fallback

## Outputs

When notebook 04 is executed end-to-end, Part 4 writes scale-aware result artifacts:

- `artifacts/eval_results/04_chromadb_graphrag_4000.json`
- `artifacts/eval_results/04_pinecone_graphrag_4000.json`
- `artifacts/eval_results/04_agent_graphrag_4000.json`

Legacy 600 artifacts remain under:

- `artifacts/eval_results/legacy_600/`
