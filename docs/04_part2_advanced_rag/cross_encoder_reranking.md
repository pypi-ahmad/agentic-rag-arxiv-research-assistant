# Cross-Encoder Reranking

Retrieval gives us a ranked list of candidates. Reranking asks: given the query *and* a specific document together, how relevant is this document really? That joint question is something a retriever — which scores query and documents independently — cannot answer well.

This page explains the two-stage retrieve-then-rerank architecture, walks through the `Reranker` class in `src/retriever.py`, and covers the latency tradeoff that makes this pattern practical.

---

## Why retrieval alone isn't enough

When `DenseRetriever` embeds a query, it produces one vector. Each document was already embedded at index time into its own vector. Relevance is approximated by the cosine distance between those two independently-produced vectors.

The problem: the encoding happens in isolation. The query encoder never sees the document; the document encoder never sees the query. They're strangers who never met.

A **cross-encoder** does the opposite: it encodes the query and document *together*, concatenated as a single input sequence. Every attention head in every layer can attend to both the query tokens and the document tokens simultaneously. The result is a single relevance score that captures fine-grained query-document interactions that independent encoding misses.

```
Bi-encoder (retrieval):
    query → encoder → q_vec ─┐
                               ├── cosine_sim → score
    doc   → encoder → d_vec ─┘

Cross-encoder (reranking):
    [query] [SEP] [document] → encoder → relevance score
```

The catch: you cannot pre-compute cross-encoder scores. Every query requires a fresh forward pass over every candidate. For 600,000 documents that's prohibitive; for 20 candidates, it runs in under a second.

This is the two-stage strategy: retrieve fast over the full corpus, then rerank the short candidate list accurately.

---

## The Reranker class

```python
class Reranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        logger.info(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        if not candidates:
            return []

        # Build (query, passage) pairs for the cross-encoder
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)

        # Zip scores back to chunks and sort descending
        scored = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )

        results = []
        for score, chunk in scored[:top_k]:
            chunk = dict(chunk)
            chunk["score"] = float(score)
            chunk["retriever"] = "reranked"
            results.append(chunk)

        return results
```

The key line is `self.model.predict(pairs)`. `CrossEncoder.predict()` from `sentence-transformers` accepts a list of `(query, passage)` tuples and returns a numpy array of logits — one per pair — in a single batched forward pass. This is faster than calling `predict()` 20 times individually because the model processes them as a batch on the same device.

---

## Why k=20 candidates, not k=5?

In the notebook, `HybridRetriever.retrieve()` is called with `k=20` before reranking, not `k=5`. This is deliberate.

The reranker fixes rank order within the candidate set — but it can't promote a document that wasn't retrieved in the first place. If the correct paper is ranked 7th by hybrid retrieval and you only pass top-5 to the reranker, that paper is gone.

Fetching `k=20` for reranking and returning `top_k=5` means:

- Retrieval casts a wide net (Recall@20 is higher than Recall@5)
- Reranking sharpens the final top-5 from that wider pool

The tradeoff is latency: 20 cross-encoder pairs instead of 5. On CPU with the MiniLM model, 20 pairs takes roughly 80–120ms. That's acceptable for interactive use.

---

## The ms-marco-MiniLM-L-6-v2 model

The default model is `cross-encoder/ms-marco-MiniLM-L-6-v2` from Hugging Face.

**MS MARCO** (Microsoft Machine Reading Comprehension) is a large-scale dataset of real Bing search queries paired with passages and human-judged relevance labels. A model trained on MS MARCO learns what "relevant passage for a query" means from millions of real web search examples.

**MiniLM-L-6** means a distilled BERT-style transformer with 6 layers and 22MB on disk. It's fast enough to run on CPU without a GPU (the primary reason it was chosen for this tutorial — no VRAM is consumed during reranking, leaving GPU memory free for the LLM).

!!! note "Zero-shot transfer to ArXiv"
    The model was trained on web search passages, not academic paper abstracts. It is applied here zero-shot — with no fine-tuning on the ArXiv corpus. Despite this domain gap, it still improves MRR from 0.287 to 0.363 on the RRF+Rerank configuration. Fine-tuning on in-domain data (query-abstract pairs from ArXiv) would likely improve this further.

---

## Latency profile

Measured on a single CPU core (Intel Core i7), 20 candidate pairs:

| Step | Latency |
|---|---|
| Dense retrieval (FAISS, 600 papers) | ~5ms |
| BM25 scoring (rank_bm25, 600 papers) | ~8ms |
| Hybrid fusion | <1ms |
| Cross-encoder reranking (20 pairs, CPU) | ~90–120ms |
| **Total pipeline** | **~120ms** |

Most of the latency is in reranking. For a chatbot-style interface where the user waits for an LLM response anyway, 120ms is imperceptible. For batch evaluation over 20 queries, it adds a few seconds.

---

## Impact on the benchmark

| Strategy | Recall@5 | MRR |
|---|---|---|
| Hybrid RRF | 0.267 | 0.287 |
| Hybrid RRF + Rerank | 0.333 | 0.363 |

Reranking adds +0.066 Recall@5 and +0.075 MRR over RRF alone. The reranker rescues documents that were scored too conservatively by rank-based fusion — their cross-encoder score promotes them into the final top-5.

The reranker does not recover all of RRF's gap versus BM25 alone (MRR 0.363 vs 0.441 for BM25 improved), which is why Part 2's headline result is "improved BM25 tokenisation" rather than reranking. But the reranker adds consistent, predictable improvement on top of any upstream retrieval strategy, making it a low-risk addition to any pipeline.
