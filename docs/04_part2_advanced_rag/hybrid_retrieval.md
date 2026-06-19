# Hybrid Retrieval

Dense and BM25 retrieval have complementary failure modes. Dense search misses exact technical terms; BM25 misses paraphrase and semantic similarity. The natural response is to run both and combine their results — that's hybrid retrieval.

But combining two ranked lists with different score scales is non-trivial. This page walks through both fusion strategies in `src/retriever.py`, explains the alpha-ablation experiment, and looks at a surprising result: the theoretically more robust method (RRF) did not beat the simpler one on this corpus.

---

## The HybridRetriever class

```python
class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        alpha: float = 0.7,
        fusion: str = "alpha",  # "alpha" or "rrf"
        rrf_k: int = 60,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.alpha = alpha
        self.fusion = fusion
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int = 10) -> list[dict]:
        fetch_k = k * 2

        dense_results = self.dense.retrieve(query, k=fetch_k)
        bm25_results = self.bm25.retrieve(query, k=fetch_k)

        if self.fusion == "rrf":
            return self._rrf_fusion(dense_results, bm25_results, k)
        else:
            return self._alpha_fusion(dense_results, bm25_results, k)
```

`retrieve()` always fetches `k*2` candidates from each retriever. If the final output is top-5, each retriever returns top-10. This ensures that a document ranked 6th by dense and 3rd by BM25 still makes it into the fusion pool — it might deserve to be in the final top-5 once scores are combined.

---

## Strategy 1: Alpha-weighted score fusion

The idea is simple: normalise both score sets to [0, 1], then take a weighted average.

```python
def _alpha_fusion(self, dense_results, bm25_results, k):
    dense_map = {r["chunk_id"]: r["score"] for r in dense_results}
    bm25_map  = {r["chunk_id"]: r["score"] for r in bm25_results}

    all_ids = set(dense_map) | set(bm25_map)

    # Min-max normalise dense scores
    d_vals = list(dense_map.values())
    d_min, d_max = min(d_vals), max(d_vals)

    # Min-max normalise BM25 scores
    b_vals = list(bm25_map.values())
    b_min, b_max = min(b_vals), max(b_vals)

    def norm_dense(s):
        return (s - d_min) / max(d_max - d_min, 1e-10)

    def norm_bm25(s):
        return (s - b_min) / max(b_max - b_min, 1e-10)

    fused = []
    for cid in all_ids:
        chunk = dict(all_chunks[cid])
        d_score = norm_dense(dense_map.get(cid, d_min))
        b_score = norm_bm25(bm25_map.get(cid, b_min))
        chunk["score"] = self.alpha * d_score + (1 - self.alpha) * b_score
        chunk["retriever"] = "hybrid"
        fused.append((chunk["score"], chunk))

    fused.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in fused[:k]]
```

Step by step:

1. Build a `chunk_id → score` lookup for each retriever's results
2. Take the **union** of all candidate IDs (a document only in BM25 is still a candidate)
3. Min-max normalise each score set independently so both live in [0, 1]
4. For each candidate, compute `alpha × dense_norm + (1-alpha) × bm25_norm`
5. A document that only appeared in one retriever gets score `0` for the other (mapped to the minimum)
6. Sort by fused score and return top-k

With `alpha=0.7`, the final score is "70% semantic similarity, 30% BM25 keyword match".

---

## Why alpha=0.7?

The alpha value was determined by ablation — running the full 20-query evaluation at several values and recording MRR:

| Alpha | MRR |
|---|---|
| 0.3 (BM25-heavy) | 0.271 |
| 0.5 (equal weight) | 0.289 |
| 0.7 (dense-heavy) | 0.308 |
| 0.9 (almost dense) | 0.301 |

MRR peaks around `alpha=0.7`. This reflects the corpus: with 600 recent ML paper abstracts, semantic similarity carries more signal than keyword overlap for most queries. The 30% BM25 contribution helps with exact-term queries without overwhelming the semantic component.

---

## Strategy 2: Reciprocal Rank Fusion (RRF)

RRF is a rank-based fusion method. It ignores raw scores entirely and only uses each document's position in each retriever's sorted list.

```python
def _rrf_fusion(self, dense_results, bm25_results, k):
    scores: dict[str, float] = {}
    all_chunks: dict[str, dict] = {}

    for results in [dense_results, bm25_results]:
        for rank, chunk in enumerate(results):
            cid = chunk["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
            all_chunks[cid] = chunk

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    result = []
    for cid, score in ranked:
        chunk = dict(all_chunks[cid])
        chunk["score"] = score
        chunk["retriever"] = "hybrid-rrf"
        result.append(chunk)
    return result
```

Step by step:

1. For each retriever's result list, iterate through `(rank, chunk)` pairs
2. Add `1 / (60 + rank + 1)` to that document's running RRF score
3. The constant `60` (from the original RRF paper, Cormack et al. 2009) controls how steeply the contribution falls off with rank
4. A document ranked 1st contributes `1/61 ≈ 0.016`; ranked 10th contributes `1/71 ≈ 0.014`; ranked 100th contributes `1/161 ≈ 0.006`
5. Documents appearing in both lists get contributions from both — their scores sum

The appeal of RRF is that it sidesteps score normalisation entirely. You never need to worry about the fact that BM25 scores are raw values in [0, ∞) while dense scores are cosine similarities in [0, 1].

---

## The surprising result: RRF did not win

| Strategy | Recall@5 | MRR |
|---|---|---|
| Dense baseline | 0.250 | 0.299 |
| BM25 improved | 0.367 | 0.441 |
| Hybrid alpha=0.7 | 0.267 | 0.308 |
| Hybrid RRF | 0.267 | 0.287 |

RRF scored lower than both BM25 alone and alpha fusion. This is counter-intuitive — RRF is theoretically more robust than score-level fusion. Why didn't it win?

**The answer lies in what RRF throws away.** Alpha fusion, after min-max normalisation, preserves score *magnitude*. A dense score of 0.95 vs 0.60 tells you the first document is far more similar — and that information survives into the fused score. RRF collapses those two documents into ranks 1 and 2, treating the gap between them as `1/61 - 1/62 ≈ 0.0003`. On a corpus where dense embeddings carry strong signal (well-represented ML terminology), discarding magnitude hurts.

!!! note "When RRF wins"
    RRF is the right choice when retriever score scales are incomparable or unstable — for example, when fusing results from different embedding models, different corpora, or sparse retrievers with very different vocabulary sizes. For two retrievers that are both well-calibrated on the same corpus, alpha fusion with good normalisation can outperform it.

The practical takeaway: **test both** on your corpus. The answer depends on how well-calibrated your retrievers are, not on the theory.
