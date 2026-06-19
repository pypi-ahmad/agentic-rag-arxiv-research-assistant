# Chunking

Before we embed anything, we need to decide what unit of text to embed. This page explains
why we don't embed whole abstracts, what the 512/64 parameters mean, and how `chunk_documents()`
implements overlapping windows.

---

## Why chunk at all?

Short answer: granularity. A user asking "how does LoRA reduce memory?" wants the two sentences
that explain the rank-decomposition trick — not an entire 300-word abstract that spends half
its words on motivation and evaluation.

When you embed a full abstract, the resulting vector represents the centroid of all the ideas
in the document. A paper that covers LoRA, QLoRA, and adapter modules will produce a single
vector that sits roughly equidistant from all three topics. If the query is about LoRA
specifically, the full-document vector may not be the closest match.

By splitting the abstract into smaller pieces, each chunk's vector captures a narrower slice
of meaning. The retriever can surface the exact passage that answers the question, not just
the paper that roughly covers the topic.

!!! note "Abstracts are already short"
    At 150–400 words, an ArXiv abstract typically produces only 1–3 chunks at 512 characters.
    This is intentionally coarse — we are not splitting paragraphs mid-sentence, we are keeping
    full semantic units while still giving the embedding model a focused input. For full papers
    (methods, results, discussion sections), you would use smaller chunks: 256–384 characters.

---

## The 512 / 64 parameters

```python
DEFAULT_CHUNK_SIZE    = 512   # target characters per chunk (~128 tokens)
DEFAULT_CHUNK_OVERLAP = 64    # characters shared between consecutive chunks
```

**Chunk size — 512 characters**

At roughly 4 characters per token, 512 characters is about 128 tokens. This is well within
`qwen3-embedding:0.6b`'s context window and gives the model a meaningful unit of text to
work with — usually one or two sentences of a typical abstract paragraph.

**Overlap — 64 characters**

Overlap is the number of characters that are repeated at the end of one chunk and the
start of the next. Consider what happens without overlap at a chunk boundary:

```
... The model is trained using a contrastive [BOUNDARY] objective that pushes representations
of the same image under different augmentations closer together ...
```

Without overlap, the phrase "contrastive objective" is split across two chunks. A query about
"contrastive objectives in self-supervised learning" might match neither chunk strongly because
neither chunk contains the full phrase. With 64-character overlap, the end of the first chunk
includes enough context for the phrase to survive intact in at least one of the two chunks.

**Visualised overlap:**

```
Abstract text:  |-------- abstract (400 chars) --------|

Chunk 0:        |<--- 512 chars --->|
                                ^--- boundary
Chunk 1:                 |<--- 512 chars --->|
                 |<-64->| overlap
```

The step between chunk starts is `chunk_size - chunk_overlap = 512 - 64 = 448` characters.

!!! tip "Choosing your own parameters"
    For production use, tune chunk size based on two constraints:
    (1) the embedding model's max input length, and
    (2) the typical length of a relevant passage in your domain.
    Overlap of 10–15% of chunk size is a good starting point (64 / 512 ≈ 12.5% here).
    Larger overlap = more chunks = more index memory + slower embedding time.

---

## The `chunk_documents()` function

```python
def chunk_documents(
    papers: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict]:
```

The implementation is a simple sliding window over characters:

```python
step = chunk_size - chunk_overlap   # 448
start = 0
chunk_idx = 0
while start < len(text):
    end = min(start + chunk_size, len(text))
    chunk_text = text[start:end].strip()
    if len(chunk_text) > 50:   # skip tiny trailing fragments
        chunks.append({...})
        chunk_idx += 1
    start += step
```

The `len(chunk_text) > 50` guard discards the short trailing fragment that appears when
a text's length is not a clean multiple of the step size. A 5-word fragment at the end of
an abstract carries almost no retrievable information.

---

## What a chunk dict looks like

```python
{
    "chunk_id":    "arxiv_train_05042_chunk_0",
    "paper_id":    "arxiv_train_05042",
    "title":       "We propose a novel attention mechanism for transformer models...",
    "category":    "cs.AI",
    "text":        "We propose a novel attention mechanism for transformer models that "
                   "dynamically weights each head based on task context. Our method, "
                   "HydraHead, reduces inference cost by 23% while maintaining ...",
    "chunk_index": 0,
}
```

The `chunk_id` is unique across the entire corpus. The `paper_id` links back to the source
paper — this is what the evaluator uses to decide whether a retrieved chunk counts as a hit.
The `text` field is what gets passed to `embed_texts()`.

For a 600-paper corpus with abstracts averaging ~300 characters, `chunk_documents()` typically
produces around 700–900 chunks total. Most abstracts produce one chunk; longer ones produce two.

---

## What's next

With chunks in hand, the next step is turning each chunk's text into a 1024-dimensional
vector. Head to [Embedding and Indexing](embedding_indexing.md) to see how `embed_texts()`
batches calls to Ollama, why we L2-normalise the vectors, and how `build_faiss_index()`
stores them for fast retrieval.
