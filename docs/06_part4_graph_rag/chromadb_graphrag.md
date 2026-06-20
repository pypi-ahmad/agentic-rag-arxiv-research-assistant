# GraphRAG on ChromaDB

Section 1 of notebook 04 builds a full GraphRAG pipeline on a local ChromaDB collection.

## Pipeline

1. Load 4,000 papers (`load_hf_papers(..., ml_filter=True)`)
2. Chunk abstracts and embed with `qwen3-embedding:4b`
3. Upsert vectors to ChromaDB
4. Extract entities with `granite4.1:8b` and cache progress
5. Build NetworkX knowledge graph
6. Detect communities and summarize them
7. Run local/global search for retrieval + generation
8. Evaluate retrieval/generation/judge metrics

## Why ChromaDB first

- Persistent local store
- No cloud dependency
- Fast iteration for notebook development
- Simple metadata-backed retrieval debugging

## Artifact (runtime-generated)

- `artifacts/eval_results/04_chromadb_graphrag_4000.json`
