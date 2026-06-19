# CRAG Architecture

This page explains the research paper behind the agent design, maps our implementation
to the paper, and documents the `GraphState` TypedDict — the shared memory that every
node reads from and writes to.

---

## The CRAG paper

**Corrective Retrieval-Augmented Generation** (Yan et al., 2024) diagnoses a
fundamental problem with standard RAG: the retriever is not always right, and the
generator does not know when it is not.

In a standard RAG pipeline, whatever the retriever returns goes directly into the
prompt. If the top-5 chunks are tangentially related or outright wrong for the query,
the LLM generates an answer anyway — and it often sounds confident. The pipeline has
no mechanism to detect retrieval failure before generation.

CRAG's solution has three stages:

1. **Evaluate retrieval quality.** A lightweight evaluator (in the paper, a fine-tuned
   T5; in our implementation, an LLM judge) scores each retrieved document against the
   query. Documents are classified as correct, incorrect, or ambiguous.

2. **Take corrective action based on the score.** If at least one document scores
   "correct", proceed to generation. If the retrieval is ambiguous or entirely
   incorrect, fall back to a web search to supplement or replace the corpus results.

3. **Generate from quality-controlled context.** Generation only happens after the
   context has passed through the evaluator gate. Bad evidence cannot reach the LLM.

Our implementation adds a fourth stage not in the original paper: a post-generation
faithfulness check that loops back if the answer introduces unsupported claims.

---

## Our implementation vs the paper

| CRAG (Yan et al.) | Our implementation |
|---|---|
| Fine-tuned T5 relevance scorer | granite4.1:8b with a JSON prompt |
| 3-way classification: correct / incorrect / ambiguous | 2-way: relevant / irrelevant (see [Threshold Improvement](threshold_improvement.md)) |
| Web search to supplement ambiguous results | Web search only on fully irrelevant results |
| No post-generation check | grade_hallucination node with retry loop |
| Cloud LLM for generation | Local granite4.1:8b via Ollama |

The most important difference is the evaluator. The paper's T5 scorer is fast and
trained specifically for document relevance. Our LLM judge is slower but requires no
fine-tuning, which keeps the entire system local and zero-cost to deploy.

---

## The GraphState TypedDict

All five nodes communicate through a single `TypedDict`. Every node receives the
complete state and returns only the keys it modifies.

```python
from typing import TypedDict, Optional

class GraphState(TypedDict):
    question: str
    documents: list[dict]
    filtered_documents: list[dict]
    retrieval_grade: str
    generation: str
    generation_attempts: int
    faithfulness_grade: str
    execution_trace: list[str]
```

### Field-by-field explanation

**`question: str`**
The user's query. Set once at invocation and never modified. Every node that needs the
question reads it from here rather than being passed it as a parameter — there is only
one place the question lives.

**`documents: list[dict]`**
The raw retrieved documents from the `retrieve` node. Each dict has at minimum a
`"content"` key with the chunk text. This is the unfiltered output of the
`DenseRetriever`.

**`filtered_documents: list[dict]`**
The documents that survive grading. After `grade_documents` runs, this contains only
the chunks that the LLM judge rated as relevant. If `web_search` runs, its results are
appended here too. This is what `generate_answer` uses as context — never `documents`
directly.

**`retrieval_grade: str`**
Either `"relevant"` or `"irrelevant"`. Set by `grade_documents` and read by the
conditional edge routing function. This is the single value that determines whether the
agent goes to `generate_answer` or `web_search` next.

**`generation: str`**
The LLM's generated answer. Set by `generate_answer`. On a hallucination-triggered
retry, it gets overwritten with the new attempt.

**`generation_attempts: int`**
A counter incremented by `generate_answer` each time it runs. The `grade_hallucination`
routing function checks this to enforce the two-attempt maximum — without it, a
persistently hallucinating LLM could loop forever.

**`faithfulness_grade: str`**
Either `"faithful"` or `"hallucinated"`. Set by `grade_hallucination` and read by its
conditional edge. Determines whether the graph terminates or loops back.

**`execution_trace: list[str]`**
A list of strings that each node appends to, recording what it did and why. This is
purely for debugging — it has no effect on routing. After a query completes, you can
inspect `result["execution_trace"]` to see the complete step-by-step log of what the
agent decided and why.

---

## The execution_trace in practice

Debugging a graph without a trace is painful. You can see the final answer but not why
the agent took the path it did. The trace makes every decision visible:

```python
result = rag_agent.invoke(initial_state)

for step in result["execution_trace"]:
    print(step)
```

Example output for a query that passes through the happy path:

```
retrieve: found 10 documents for 'attention mechanisms in transformers'
grade_documents: doc 0 → relevant (score: high match on attention mechanism description)
grade_documents: doc 1 → relevant (score: mentions self-attention and query-key-value)
grade_documents: doc 2 → irrelevant (score: about image classification, not attention)
...
grade_documents: 4 relevant out of 10, grade=relevant
generate_answer: attempt 1
grade_hallucination: 0 unsupported claims → faithful
```

And for a query that triggers the web fallback and then a retry:

```
retrieve: found 10 documents for 'LLM watermarking 2024'
grade_documents: 0 relevant out of 10, grade=irrelevant
web_search: querying DuckDuckGo for 'LLM watermarking 2024'
web_search: appended 3 web results to filtered_documents
generate_answer: attempt 1
grade_hallucination: 2 unsupported claims → hallucinated
generate_answer: attempt 2
grade_hallucination: 0 unsupported claims → faithful
```

The trace turns an opaque black box into a legible audit trail.

---

## What's next

[LLM as Judge](llm_as_judge.md) goes inside the two grading nodes — the exact prompts
used, how JSON output is parsed, and why granite4.1:8b was chosen as the judge model.
