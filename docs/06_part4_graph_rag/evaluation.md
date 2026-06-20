# Evaluation Design

Part 4 applies the same evaluation structure to ChromaDB, Pinecone, and Agentic GraphRAG:

- Retrieval metrics: Precision@K, Recall@K, F1@K, MRR, NDCG@K
- Generation metrics: EM, BLEU, ROUGE-1/2/L, METEOR, BERTScore
- LLM-as-judge metrics: faithfulness and hallucination checks

## Data scale

- Primary baseline: 4,000-paper corpus (current)
- Legacy baseline: 600-paper corpus (historical, preserved for comparison)

## Outputs consumed by docs/README

- `01_naive_rag_4000.json`
- `02_advanced_rag_4000.json`
- `03_agentic_rag_4000.json`
- `04_chromadb_graphrag_4000.json`
- `04_pinecone_graphrag_4000.json`
- `04_agent_graphrag_4000.json`
- `legacy_600/*.json`
