# BM25 in Practice

BM25 (Best Match 25) is a classical keyword-scoring algorithm that has been the gold standard in information retrieval since 1994. Before neural embeddings existed, BM25 was how search engines ranked documents. It is fast, interpretable, and exceptionally good at one thing: scoring documents that share exact words with the query.

This page walks through the `BM25Retriever` class in `src/retriever.py`, explains the tokenisation bug that limited its performance in the original version, and shows the fix that produced the biggest single improvement in the entire benchmark.

---

## How BM25 scores a document

For each query term `t`, BM25 computes:

```
score(D, Q) = Σ_t  IDF(t) × (tf(t,D) × (k1+1)) / (tf(t,D) + k1×(1 - b + b×|D|/avgdl))
```

Where:

- `IDF(t)` = `log((N - df(t) + 0.5) / (df(t) + 0.5))` — rare terms score higher
- `tf(t, D)` = how many times term `t` appears in document `D`
- `|D|` = document length; `avgdl` = average document length across the corpus
- `k1=1.5`, `b=0.75` are the standard Okapi BM25 constants

In plain English: a term contributes high score when it appears often *in this document* (high tf) but rarely *across all documents* (high IDF). Long documents are penalised by the length normalisation term `b×|D|/avgdl`.

This is why BM25 complements dense retrieval. Dense embeddings capture meaning regardless of exact words. BM25 rewards exact term overlap — which is exactly what you want when a query contains specific technical terms like `LoRA`, `RLHF`, or `SQuAD`.

---

## The BM25Retriever class

```python
class BM25Retriever:
    def __init__(self, chunks: list[dict]) -> None:
        self.chunks = chunks

        # OLD tokenisation (whitespace only — kept for reference):
        # tokenised_corpus = [chunk["text"].lower().split() for chunk in chunks]

        # IMPROVED tokenisation (lowercase + strip punctuation + stop words):
        tokenised_corpus = [_tokenise(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenised_corpus)
        logger.info(f"Built BM25 index over {len(chunks)} chunks.")

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        # OLD: query_tokens = query.lower().split()
        # IMPROVED: same tokeniser as the corpus — must match for consistent scoring
        query_tokens = _tokenise(query)
        scores = self.bm25.get_scores(query_tokens)

        top_k_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_k_indices:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            chunk["retriever"] = "bm25"
            results.append(chunk)

        return results
```

The constructor builds the BM25 index once — `BM25Okapi(tokenised_corpus)` tokenises and stores term frequencies for every chunk. At query time, `bm25.get_scores(query_tokens)` returns a raw score for every document in the corpus as a numpy array, which is then sorted to get the top-k.

!!! note "Index cost is O(corpus)"
    Building the BM25 index is a one-time O(N) operation over the corpus. Each query is O(|query terms| × N) for score computation. For 600 papers and typical short queries, this is fast enough that it does not need caching.

---

## The old tokenisation: three failure modes

The original implementation used Python's default whitespace split:

```python
# OLD — do not use
tokenised_corpus = [chunk["text"].lower().split() for chunk in chunks]
query_tokens = query.lower().split()
```

This looks harmless but introduces three problems:

**Failure mode 1 — Punctuation creates phantom tokens.**
`"attention."` and `"attention"` are different tokens under whitespace split. If the query contains `attention` but the corpus contains `attention.` (end of sentence), BM25 scores zero for that term even though they're the same word. In a paper abstract, virtually every noun appears at least once with trailing punctuation.

**Failure mode 2 — Stop words consume scoring budget.**
Words like `the`, `is`, `of`, `in` appear in every document with high term frequency but near-zero IDF (since they appear in *all* documents, `log((N - N + 0.5)/(N + 0.5)) ≈ 0`). Their direct contribution to BM25 is negligible. The problem is that they pollute the token set: when the query is `"what is the attention mechanism"`, half the query tokens are stop words that score near-zero everywhere and add noise to the final ranking.

**Failure mode 3 — Mixed case breaks rare-term matching.**
ML papers often use mixed case: `"Transformer"` at the start of a sentence, `"transformer"` mid-sentence. Under whitespace split + lowercase, these are the same. But any uppercase-only tokenisation (even accidental) would split them.

---

## The new `_tokenise()` function

```python
BM25_STOP_WORDS: frozenset[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "as", "this", "that", "these",
    "those", "it", "its", "we", "our", "they", "their", "which", "who",
    "and", "or", "but", "not", "no", "nor", "so", "yet", "both", "each",
    "about", "than", "more", "also", "such", "into", "its", "over",
})


def _tokenise(text: str) -> list[str]:
    """Lowercase, strip non-alphanumeric chars, remove stop words and single-char tokens."""
    tokens = re.sub(r"[^a-z0-9]", " ", text.lower()).split()
    return [t for t in tokens if t not in BM25_STOP_WORDS and len(t) > 1]
```

Three steps in one line:

1. `text.lower()` — case-normalise the whole string first
2. `re.sub(r"[^a-z0-9]", " ", ...)` — replace every non-alphanumeric character with a space, so `attention.` becomes `attention ` and hyphens in compound terms are treated as separators
3. Filter: drop tokens that are stop words or single characters

The stop-word list is intentionally narrow — only function words and auxiliary verbs. Technical nouns like `not` in "not pre-trained" are excluded deliberately because negation can matter for relevance.

!!! important "Corpus and query must use the same tokeniser"
    BM25 scoring requires that query tokens and corpus tokens are in the same vocabulary. If you tokenise the corpus with `_tokenise()` but send the query through `.lower().split()`, you'll be searching for `"attention."` in an index that contains `"attention"` — and scoring zero for every document. The `retrieve()` method calls `_tokenise(query)` for exactly this reason.

---

## Impact

This change — three lines of code — produced the biggest single improvement in the 20-query benchmark:

| Tokenisation | Recall@5 | MRR |
|---|---|---|
| Dense baseline | 0.250 | 0.299 |
| BM25 (whitespace split) | 0.300 | 0.333 |
| BM25 (improved `_tokenise`) | **0.367** | **0.441** |

MRR jumps from 0.299 (dense baseline) to 0.441 (+47%). The improvement is entirely from better preprocessing — the BM25 algorithm, the corpus, and the queries are identical.
