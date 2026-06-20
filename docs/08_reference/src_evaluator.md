# `src/evaluator.py` — Evaluation Metrics

Source: [`src/evaluator.py`](https://github.com/pypi-ahmad/agentic-rag-arxiv-research-assistant/blob/main/src/evaluator.py)

Standalone evaluation utilities used across notebook tracks. Retrieval metrics require no LLM; generation metrics call `granite4.1:8b` via Ollama as an LLM judge.

---

## Retrieval metrics

### `recall_at_k(retrieved_ids, relevant_ids, k)`

```python
def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
```

**Formula:** `|relevant ∩ top-k| / |relevant|`

Returns the fraction of all relevant documents that appeared in the top-k retrieved results.

| Score | Interpretation |
|-------|---------------|
| `1.0` | All relevant docs found in top-k |
| `0.5` | Half of relevant docs found |
| `0.0` | No relevant docs retrieved |

**Example:**
```python
retrieved = ["2401.15884", "2312.00752", "2310.01117", "2401.00001", "2312.11111"]
relevant  = ["2401.15884", "2312.00752", "2401.99999"]  # 3 relevant docs exist

recall_at_k(retrieved, relevant, k=5)
# → 2/3 = 0.667  (found 2 of 3 relevant in top-5)
```

---

### `precision_at_k(retrieved_ids, relevant_ids, k)`

```python
def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
```

**Formula:** `|relevant ∩ top-k| / k`

Returns the fraction of the top-k results that were actually relevant.

!!! note "Recall vs Precision"
    Recall asks: "did we find everything relevant?"  
    Precision asks: "is everything we returned relevant?"  
    For RAG with k=5, precision matters more — 5 irrelevant chunks in the prompt hurt generation quality even if many relevant docs exist elsewhere.

---

### `mrr(queries_retrieved, queries_relevant)`

```python
def mrr(queries_retrieved: list[list[str]], queries_relevant: list[list[str]]) -> float:
```

**Formula:** `(1/N) × Σ 1/rank_first_relevant`

Mean Reciprocal Rank — the average of the reciprocal rank of the first relevant document across all queries.

| First relevant at rank | Reciprocal rank |
|----------------------|-----------------|
| 1 | 1.000 |
| 2 | 0.500 |
| 3 | 0.333 |
| 5 | 0.200 |
| Not found | 0.000 |

**Why MRR matters more than Recall@k for RAG:** The LLM generator pays most attention to the first document in the context. Getting the single most relevant document to rank 1 has a greater impact on answer quality than having 4 of 5 relevant docs in the top-5.

---

### `compute_retrieval_metrics(retriever, eval_queries, k)`

```python
def compute_retrieval_metrics(
    retriever,
    eval_queries: list[dict],
    k: int = 5,
) -> dict:
```

Convenience wrapper that runs a full eval loop and returns a metrics dict.

**`eval_queries` format:**
```python
[
    {
        "question": "How does RLHF work?",
        "relevant_ids": ["2203.02155", "2304.01852"]
    },
    ...
]
```

**Returns:**
```python
{
    "recall@5": 0.367,
    "precision@5": 0.190,
    "mrr": 0.441,
    "k": 5,
    "n_queries": 20
}
```

---

## Generation metrics (LLM-as-judge)

### `score_faithfulness(question, context_docs, answer, model)`

```python
def score_faithfulness(
    question: str,
    context_docs: list[dict],
    answer: str,
    model: str = "granite4.1:8b",
) -> dict:
```

Uses `granite4.1:8b` to judge whether every claim in `answer` is supported by `context_docs`.

**Prompt pattern:** *"Does this answer contain any claims not supported by the context? Return JSON `{faithful: true/false, unsupported_claims: int}`"*

**Returns:**
```python
{"faithful": True, "unsupported_claims": 0}
# or
{"faithful": False, "unsupported_claims": 3}
```

!!! warning "LLM judge limitations"
    The judge is non-deterministic — the same answer can score differently across runs on borderline cases. For production use, cache grades or use a smaller dedicated classifier trained on your domain.

---

### `score_answer_relevance(question, answer, model)`

```python
def score_answer_relevance(question: str, answer: str, model: str = "granite4.1:8b") -> float:
```

Scores whether `answer` actually addresses `question`. Returns `0.0`, `0.5`, or `1.0`.

!!! note "Faithfulness ≠ Relevance"
    An answer can be perfectly faithful (every claim is in the context) but irrelevant (the retrieved context was about the wrong topic). Both metrics are needed for a complete picture.

---

## `EvalResults` dataclass

```python
@dataclass
class EvalResults:
    experiment_name: str
    retriever_type: str
    embed_model: str
    llm_model: str
    retrieval_metrics: dict = field(default_factory=dict)
    generation_metrics: dict = field(default_factory=dict)
    notes: str = ""

    def save(self, path: Path) -> None: ...
    def load(cls, path: Path) -> "EvalResults": ...
```

Structured container for experiment results. Serialises to / from JSON via `.save()` / `.load()`.

All notebook eval results are saved under `artifacts/eval_results/`:

| File | Contents |
|------|----------|
| `artifacts/eval_results/01_naive_rag_4000.json` | Dense baseline metrics at 4,000-paper scale |
| `artifacts/eval_results/02_advanced_rag_4000.json` | Advanced retrieval metrics at 4,000-paper scale |
| `artifacts/eval_results/03_agentic_rag_4000.json` | Agentic CRAG metrics at 4,000-paper scale |
