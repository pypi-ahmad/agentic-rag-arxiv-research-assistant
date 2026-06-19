# Evaluation

Part 3 evaluation measures agent behaviour, not retrieval rank. This is a fundamentally
different question from Parts 1 and 2. Instead of asking "did the right document appear
in the top-5?", we ask "did the agent produce a faithful, grounded answer?"

---

## Why 10 queries, not 20?

Parts 1 and 2 used a 20-query evaluation set with near-instant scoring: retrieval
metrics like Recall@5 and MRR are computed by comparing ranked lists to ground-truth
labels, which takes milliseconds.

Part 3 queries are expensive. Each query runs five LLM calls (retrieve is fast, but
`grade_documents` makes 10 judge calls, `generate_answer` makes 1, `grade_hallucination`
makes 1, with possible retries). Average latency is ~15 seconds per query.

At that rate, 20 queries takes ~5 minutes. For a tutorial notebook that a reader runs
interactively, that is too long. 10 queries gives a representative sample in ~2.5
minutes — enough to observe meaningful variation in agent behaviour without making
the notebook impractical to run.

!!! info "Latency composition"
    The ~15s per query breaks down roughly as: 10 judge calls in grade_documents
    (~10s) + 1 generation call (~3s) + 1 faithfulness call (~2s). A retry adds
    another ~5s. The bottleneck is the per-document grading loop in grade_documents.

---

## The evaluation queries

The 10 queries are designed to cover different retrieval regimes:

1. High-frequency concepts well-covered in the corpus (*"What is RLHF?"*)
2. Specific paper contributions (*"What does the LoRA paper propose?"*)
3. Comparative questions (*"How does RAG compare to fine-tuning?"*)
4. Technique explanations (*"How does the attention mechanism work?"*)
5. Application questions (*"What are challenges in deploying LLMs in production?"*)
6. Recent techniques likely to have sparse coverage (*"What is speculative decoding?"*)
7. Multi-concept queries (*"What methods improve reasoning in LLMs?"*)
8. Definition questions with clear corpus answers (*"What is knowledge distillation?"*)
9. Broader survey questions (*"What are evaluation metrics for text generation?"*)
10. Technique deep-dives (*"How do mixture-of-experts models scale?"*)

This spread tests all routing paths: most queries should be fully served by the corpus,
with the harder or newer queries exercising the agent's faithfulness check.

---

## The run_agent() function

A wrapper function handles the per-query boilerplate — constructing the initial state,
invoking the agent, and extracting the metrics we care about:

```python
def run_agent(question: str) -> dict:
    """Run the CRAG agent on a single question and return a results dict.

    Args:
        question: The user's research question.

    Returns:
        Dict with keys: question, generation, retrieval_grade,
        faithfulness_grade, generation_attempts, web_search_triggered,
        execution_trace.

    Example:
        result = run_agent("What is RLHF?")
        print(result["generation"])
        print(result["faithfulness_grade"])
    """
    initial_state: GraphState = {
        "question": question,
        "documents": [],
        "filtered_documents": [],
        "retrieval_grade": "",
        "generation": "",
        "generation_attempts": 0,
        "faithfulness_grade": "",
        "execution_trace": [],
    }

    result = rag_agent.invoke(initial_state)

    return {
        "question": question,
        "generation": result["generation"],
        "retrieval_grade": result["retrieval_grade"],
        "faithfulness_grade": result["faithfulness_grade"],
        "generation_attempts": result["generation_attempts"],
        "web_search_triggered": result["retrieval_grade"] == "irrelevant",
        "execution_trace": result["execution_trace"],
    }
```

**Key design decisions:**

- The initial state always starts with empty lists and empty strings. Do not reuse
  state across queries — each query must start clean.
- `web_search_triggered` is derived from `retrieval_grade` rather than stored
  explicitly. The agent node that runs web search does not set a boolean flag, but
  `retrieval_grade == "irrelevant"` is exactly the condition under which web search
  runs. This avoids adding a redundant field to `GraphState`.
- The function returns the `execution_trace` so the caller can inspect the full
  reasoning chain for any query.

---

## The agent_results structure

The evaluation loop runs `run_agent()` for each query and collects results:

```python
agent_results = {}

for question in eval_questions:
    print(f"Running: {question[:60]}...")
    agent_results[question] = run_agent(question)
    print(f"  → retrieval: {agent_results[question]['retrieval_grade']}, "
          f"faithful: {agent_results[question]['faithfulness_grade']}")
```

After the loop, `agent_results` is a dict keyed by question. Accessing any result:

```python
r = agent_results["What is RLHF?"]

r["generation"]           # the answer text
r["retrieval_grade"]      # "relevant" or "irrelevant"
r["faithfulness_grade"]   # "faithful" or "hallucinated"
r["generation_attempts"]  # 1 or 2
r["web_search_triggered"] # True or False
r["execution_trace"]      # list of strings
```

---

## What the metrics measure

These metrics measure agent behaviour, not retrieval quality. This is an important
distinction:

| Metric | Measures | How computed |
|---|---|---|
| Relevant retrieval rate | Did the local corpus produce useful evidence? | `retrieval_grade == "relevant"` per query |
| Web search rate | How often did the corpus fail completely? | `web_search_triggered` per query |
| Faithful answer rate | Did the answer avoid hallucination? | `faithfulness_grade == "faithful"` per query |
| Retry rate | How often did the agent need a second attempt? | `generation_attempts == 2` per query |

None of these require human-labelled ground truth. The LLM judges produce the labels.
This means evaluation is fully automated and fast — but the labels are only as good as
the judge. A judge that is lenient on relevance inflates the faithful answer rate; one
that is strict on hallucination deflates it. The judge quality is itself a variable,
not a ground truth.

!!! note "Faithfulness is not factual accuracy"
    A faithful answer is one where every claim can be traced back to the retrieved
    context. It is possible to produce a faithful but factually incorrect answer if
    the source documents themselves contain errors. Faithfulness measures internal
    consistency (answer grounded in context), not external truth.

---

## Saving results

Results are saved to `artifacts/eval/03_agentic_rag.json` for comparison with Parts
1 and 2:

```python
import json
from pathlib import Path

Path("artifacts/eval").mkdir(parents=True, exist_ok=True)
with open("artifacts/eval/03_agentic_rag.json", "w") as f:
    json.dump({
        "n_queries": len(eval_questions),
        "relevant_retrieval_rate": sum(
            1 for r in agent_results.values()
            if r["retrieval_grade"] == "relevant"
        ) / len(agent_results),
        "web_search_rate": sum(
            1 for r in agent_results.values()
            if r["web_search_triggered"]
        ) / len(agent_results),
        "faithful_rate": sum(
            1 for r in agent_results.values()
            if r["faithfulness_grade"] == "faithful"
        ) / len(agent_results),
        "results": agent_results,
    }, f, indent=2)
```

---

## What's next

[Results and Limitations](results_and_limits.md) presents the full results table,
compares agent behaviour before and after Improvement 5, and discusses what the agentic
architecture adds over Part 2 — and where its remaining limits are.
