# LLM as Judge

Two nodes in the graph make binary decisions using a language model: `grade_documents`
decides whether retrieved chunks are relevant, and `grade_hallucination` decides whether
the generated answer is faithful to the context. This page explains both nodes in
detail — the prompts, the parsing logic, and why reliable JSON output matters.

---

## Why use an LLM as a judge?

The alternative to an LLM judge is a heuristic: score by cosine similarity, count
keyword overlap, or use a fine-tuned classifier. These approaches can work, but they
have a ceiling. A chunk that scores 0.88 cosine similarity might still be about the
wrong subtopic. An LLM can read the chunk and the question together and make a semantic
judgment that a vector score cannot.

The tradeoff is speed. Each LLM call takes 1–3 seconds. With 10 retrieved documents,
`grade_documents` makes 10 LLM calls per query. This is the primary driver of the ~15s
average latency in Part 3.

---

## The grade_documents node

This node iterates over every document in `state["documents"]`, asks the LLM whether
the document is relevant to the question, counts the relevant ones, and sets
`retrieval_grade`.

```python
GRADING_PROMPT = """You are a relevance grader. Given a user question and a document,
assess whether the document contains information relevant to answering the question.

Question: {question}
Document: {document}

Respond with ONLY valid JSON in this exact format:
{{
  "score": "yes" or "no",
  "reasoning": "one sentence explaining your decision"
}}"""

def grade_documents(state: GraphState) -> dict:
    question = state["question"]
    documents = state["documents"]
    filtered_documents = []
    n_relevant = 0
    trace = list(state["execution_trace"])

    for doc in documents:
        prompt = GRADING_PROMPT.format(
            question=question,
            document=doc["content"][:800]   # truncate long chunks
        )
        response = ollama.chat(
            model="granite4.1:8b",
            messages=[{"role": "user", "content": prompt}]
        )
        try:
            result = json.loads(response["message"]["content"])
            if result.get("score") == "yes":
                filtered_documents.append(doc)
                n_relevant += 1
                trace.append(f"grade_documents: doc → relevant")
            else:
                trace.append(f"grade_documents: doc → irrelevant")
        except json.JSONDecodeError:
            # If parsing fails, treat as irrelevant — fail safe
            trace.append(f"grade_documents: doc → irrelevant (parse error)")

    retrieval_grade = "relevant" if n_relevant >= 1 else "irrelevant"
    trace.append(
        f"grade_documents: {n_relevant} relevant out of {len(documents)}, "
        f"grade={retrieval_grade}"
    )

    return {
        "filtered_documents": filtered_documents,
        "retrieval_grade": retrieval_grade,
        "execution_trace": trace,
    }
```

**Line-by-line walkthrough:**

- The prompt truncates each chunk to 800 characters. Full chunks can be 512 characters
  of content, but metadata fields can push the total higher. Truncation prevents
  the prompt from exceeding the model's context window on pathological chunks.
- `json.loads()` is wrapped in a try/except. If the model returns prose instead of
  JSON, we fail safe: treat the document as irrelevant rather than crashing.
- The threshold `n_relevant >= 1` is the 2-tier threshold introduced as Improvement 5.
  See [Threshold Improvement](threshold_improvement.md) for the history and the
  measured impact.
- Only `filtered_documents` — not `documents` — flows to `generate_answer`. Raw
  ungraded chunks never reach the generation prompt.

---

## The grade_hallucination node

After generation, this node checks whether the answer makes claims that are not
supported by the retrieved context. It counts "unsupported claims" — specific
assertions in the answer that cannot be traced back to any document in
`filtered_documents`.

```python
HALLUCINATION_PROMPT = """You are a faithfulness checker. Given an answer and the
source documents used to generate it, identify claims in the answer that are NOT
supported by the documents.

Source documents:
{context}

Answer to check:
{answer}

Respond with ONLY valid JSON in this exact format:
{{
  "unsupported_claims": <integer count of unsupported claims>,
  "reasoning": "brief explanation"
}}"""

def grade_hallucination(state: GraphState) -> dict:
    filtered_documents = state["filtered_documents"]
    generation = state["generation"]
    trace = list(state["execution_trace"])

    context = "\n\n".join(
        doc["content"] for doc in filtered_documents
    )

    prompt = HALLUCINATION_PROMPT.format(
        context=context[:3000],   # cap context length
        answer=generation
    )
    response = ollama.chat(
        model="granite4.1:8b",
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        result = json.loads(response["message"]["content"])
        unsupported = int(result.get("unsupported_claims", 0))
    except (json.JSONDecodeError, ValueError):
        unsupported = 0   # fail safe: assume faithful if parse fails

    faithfulness_grade = "faithful" if unsupported == 0 else "hallucinated"
    trace.append(
        f"grade_hallucination: {unsupported} unsupported claims "
        f"→ {faithfulness_grade}"
    )

    return {
        "faithfulness_grade": faithfulness_grade,
        "execution_trace": trace,
    }
```

**Key decisions:**

- The context is capped at 3000 characters. With up to 10 filtered documents, the
  total context can be large. Capping prevents latency spikes while still giving the
  judge enough to work with.
- The fail-safe on parse error assumes faithful (returns 0 unsupported claims). This
  is the correct conservative choice: we should only retry when we are confident the
  answer hallucinated, not because the judge failed to produce valid JSON.
- `unsupported_claims` is an integer, not a boolean. This is intentional: zero means
  faithful, any positive number means hallucinated. Future work could use the count to
  calibrate confidence rather than applying a hard threshold at 1.

---

## Why granite4.1:8b for judging?

Several properties make granite4.1:8b a good judge model for this task:

**Reliable JSON.** The model follows the JSON format instruction consistently. In
testing, parse failures were rare (under 5% of calls). Many smaller models produce
well-formed prose but inconsistent JSON, requiring more fragile parsing logic.

**Calibrated conservatism.** When the document and question are clearly unrelated,
the model says "no" confidently. When there is a partial match, it tends toward "no"
rather than inflating relevance — which is what you want from a gating mechanism.

**Speed.** At 8B parameters running locally, inference takes 1–3 seconds per call.
A larger model would produce better judgments but would make the ~15s average latency
substantially worse.

!!! tip "Improving judge accuracy"
    The single biggest lever for improving faithfulness from 70% is prompt engineering
    on the judge prompts. The current prompts are intentionally simple. Techniques like
    chain-of-thought ("reason step by step before deciding") or few-shot examples with
    correct labels can push judge accuracy significantly higher without changing the
    graph architecture.

---

## What's next

[Web Search Fallback](web_search_fallback.md) covers the `web_search` node — when it
fires, how DuckDuckGo results are formatted as documents, and the tradeoffs of
injecting uncontrolled web content into the context.
