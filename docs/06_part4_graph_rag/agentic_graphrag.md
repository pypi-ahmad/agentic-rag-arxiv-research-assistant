# Agentic GraphRAG (LangGraph)

Section 3 of notebook 04 wraps GraphRAG retrieval in a LangGraph agent loop.

## Routing stages

1. Local graph retrieval
2. Relevance grading
3. Global community retrieval fallback
4. Web fallback (only when needed)
5. Generation
6. Hallucination grading and retry gate

## Differences from Part 3 CRAG

- Uses graph-aware retrieval instead of pure FAISS dense retrieval
- Adds global community retrieval before web fallback
- Uses the same judge family across retrieval and answer checks in Part 4

## Artifact (runtime-generated)

- `artifacts/eval_results/04_agent_graphrag_4000.json`
