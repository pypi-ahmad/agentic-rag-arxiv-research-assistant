# GraphRAG on Pinecone

Section 2 of notebook 04 reuses the same embeddings/graph logic but swaps the backend
from ChromaDB to Pinecone via the shared `VectorStore` interface.

## Goal

Keep retrieval logic constant and measure backend differences under identical 4,000-paper
conditions.

## Runtime requirements

- `PINECONE_API_KEY` set in environment
- Pinecone index creation/query permissions
- Same embedding model and dimension used in Section 1

## Artifact (runtime-generated)

- `artifacts/eval_results/04_pinecone_graphrag_4000.json`

## Comparison policy

Pinecone execution is optional but recommended when you need backend comparison with
the local ChromaDB path.
