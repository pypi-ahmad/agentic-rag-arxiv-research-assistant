# Retrieval and Generation

With the FAISS index saved to disk, the pipeline is ready to answer questions. This page
covers two things: how the `DenseRetriever` class turns a user question into a ranked list
of chunks, and how the `naive_rag()` function assembles those chunks into a prompt and calls
`granite4.1:8b` for the final answer.

---

## The `DenseRetriever` class

`DenseRetriever` lives in `src/retriever.py`. Its job is to wrap the FAISS index and the
chunks list behind a clean `.retrieve(query, k)` interface.

```python
class DenseRetriever:
    def __init__(
        self,
        index,              # faiss.IndexFlatIP from build_faiss_index()
        chunks: list[dict], # same list saved by save_index_and_chunks()
        embed_model: str = EMBED_MODEL_PRIMARY,
    ) -> None:
        self.index = index
        self.chunks = chunks
        self.embed_model = embed_model
```

The two constructor arguments — `index` and `chunks` — are the exact objects returned by
`load_index_and_chunks()`. They share an implicit contract: `chunks[i]` is the text that
produced the vector at row `i` in the index.

!!! warning "Always use the same embedding model"
    The `embed_model` parameter must match the model used to build the index.
    If you embed the corpus with `qwen3-embedding:0.6b` and then query with
    `qwen3-embedding:4b`, the vectors live in different vector spaces — the
    similarity scores will be meaningless and retrieval will silently break.
    The code logs a reminder but cannot enforce this at runtime.

---

## `embed_query()` and `faiss.search()`

When `.retrieve()` is called, it performs two steps:

**Step 1 — embed the query**

```python
query_vec = embed_query(query, model=self.embed_model)
```

`embed_query()` is defined in `src/ingest.py`. It calls `ollama.embed()` with a single
string, L2-normalises the result, and returns a float32 array of shape `(1, 1024)`. The
shape `(1, 1024)` rather than `(1024,)` is deliberate — FAISS's `.search()` expects a
2D input where the first dimension is the number of queries.

**Step 2 — search the FAISS index**

```python
scores, indices = self.index.search(query_vec, k)
```

FAISS scans all stored vectors, computes the inner product between `query_vec` and each one,
and returns the `k` highest scores. Because the vectors are L2-normalised, inner product
equals cosine similarity. `scores[0]` is a float array of shape `(k,)` and `indices[0]` is
the corresponding integer row positions into the chunks list.

**Step 3 — map back to chunk dicts**

```python
for score, idx in zip(scores[0], indices[0]):
    if idx == -1:
        continue
    chunk = dict(self.chunks[idx])
    chunk["score"] = float(score)
    chunk["retriever"] = "dense"
    results.append(chunk)
```

The `idx == -1` guard handles the edge case where the index contains fewer than `k` vectors.
Each returned chunk is a copy of the original dict, augmented with a `"score"` key so the
caller can see how similar each result was.

---

## The RAG prompt template

The retrieved chunks are assembled into a structured prompt before being passed to the LLM:

```python
RAG_PROMPT = """You are a research assistant specialising in machine learning and AI.
Answer the user's question using ONLY the information provided in the context below.
If the context does not contain enough information to answer, say:
"The retrieved documents do not contain enough information to answer this question."
Do not use any knowledge beyond what is in the context.

Context:
{context}

Question: {question}

Answer:"""
```

The `ONLY` instruction is the most important line in the template. Without it, `granite4.1:8b`
will blend retrieved evidence with its parametric knowledge — producing answers that sound
correct but may include facts not present in the corpus. The explicit fallback phrase also
matters: it gives the model a graceful exit when the retrieved context is genuinely unhelpful,
rather than forcing it to hallucinate.

Each retrieved chunk is formatted as a numbered document block:

```python
context_parts = []
for i, chunk in enumerate(retrieved, 1):
    context_parts.append(f"[Document {i}] {chunk['title']}\n{chunk['text']}")
context = "\n\n".join(context_parts)
```

The `[Document N]` prefix helps the model distinguish between sources and occasionally
reference them in its answer.

---

## `ollama.chat()` — the generation call

```python
response = ollama.chat(
    model=llm_model,
    messages=[{"role": "user", "content": prompt}],
)
answer = response["message"]["content"].strip()
```

`ollama.chat()` is a blocking synchronous call. It sends the full prompt to `granite4.1:8b`
running locally and waits for the complete response. For a typical RAG answer (100–300 words),
this takes 3–8 seconds on an RTX 4060.

The return value from `naive_rag()` bundles the answer with its provenance:

```python
return {
    "question": query,
    "answer":   answer,
    "contexts": [c["text"] for c in retrieved],   # raw chunk texts
    "sources":  [c["title"] for c in retrieved],  # paper titles
    "scores":   [c["score"] for c in retrieved],  # similarity scores
}
```

Keeping `contexts` and `sources` in the return dict is important for evaluation: the
faithfulness scorer in `src/evaluator.py` needs the raw retrieved texts to check whether
the answer is grounded in them.

---

## What's next

Now that we can retrieve and generate, the next question is: how good is it?
Head to [Evaluation](evaluation.md) to see how the 20-query eval set is constructed,
what `compute_retrieval_metrics()` computes, and what the three metric numbers actually mean.
