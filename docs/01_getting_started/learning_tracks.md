# Learning Tracks

Use the track that matches your goal and available time.

---

## Track 1 — Fast Foundations (2-3 hours)

Best for first-time RAG learners who want working intuition quickly.

1. Read [What Is This Project](what_is_this_project.md)
2. Complete [Installation](installation.md)
3. Run `01_naive_rag.ipynb`
4. Run `02_advanced_rag.ipynb`
5. Read [Reading the Numbers Honestly](../07_results/reading_the_numbers.md)

Expected outcome:
- Understand dense vs sparse vs hybrid retrieval
- Know how to read Recall@5, Precision@5, and MRR
- Reproduce baseline metrics from committed artifacts

---

## Track 2 — Full Zero-to-Hero (8-12 hours)

Best for serious learning and portfolio-level understanding.

1. Follow [Tutorial Index](tutorial_index.md) in order
2. Run all nine notebooks (or inspect the executed notebook copies in the `notebooks/` folder)
3. Read each corresponding chapter under `docs/03_*` to `docs/09_*`
4. Compare 4,000 baseline vs legacy 600 baseline artifacts
5. Study [All 5 Improvements Explained](../07_results/improvements_explained.md)

Expected outcome:
- End-to-end understanding of modern RAG variants
- Ability to explain tradeoffs (latency, quality, complexity)
- Evidence-backed project story for interviews

---

## Track 3 — Recruiter / Interview Review (45-90 minutes)

Best when you need a quick but credible project walkthrough.

1. Read [Recruiter Brief](../07_results/recruiter_brief.md)
2. Verify key numbers in [Full Benchmark Table](../07_results/benchmark_table.md)
3. Confirm evidence mapping in [Claims Traceability](../08_reference/claims_traceability.md)
4. Download [Full Tutorial PDF](handbook_pdf.md)

Expected outcome:
- Clear architecture and decision narrative
- Metric claims tied to concrete artifacts
- Interview-ready explanation of what was built and measured

---

## Recommended Runtime Profiles

| Profile | Use when | Minimum path |
|---|---|---|
| Local-only | No cloud credentials available | Parts 1-3 + Part 5 notebooks |
| Hybrid local + cloud | Pinecone evaluation is needed | Add Part 4 Pinecone variant |

---

## Quality and Verification Commands

```bash
uv run python scripts/check_docs_integrity.py
uv run python scripts/check_docs_facts.py
uv run mkdocs build --strict
```
