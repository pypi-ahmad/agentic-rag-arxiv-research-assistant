# Results and Limits

This page reports Part 4 outcomes at 4,000-paper scale and keeps the historical
600-paper baseline visible for direct comparison.

## Result sources

- 4,000 baseline:
  - `04_chromadb_graphrag_4000.json`
  - `04_pinecone_graphrag_4000.json`
  - `04_agent_graphrag_4000.json`
- Legacy 600 baseline:
  - `legacy_600/01_naive_rag_legacy_600.json`
  - `legacy_600/02_advanced_rag_legacy_600.json`

## Interpretation guidance

- Use 600 numbers as a historical low-scale reference.
- Use 4,000 numbers as the current tutorial baseline.
- Focus on trend direction (coverage/robustness vs latency/cost), not one metric in isolation.

## Known constraints

- Part 4 runtime is significantly longer due to entity extraction and community summarization.
- Pinecone runs require valid cloud credentials and may be sensitive to quota/region settings.
