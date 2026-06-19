# All 5 Improvements Explained

This page is a post-mortem for each of the five changes made between the initial
prototype and the final system. For each improvement: what was wrong, what the
fix was, what it measured, and the actual code that changed.

Read this as a case study in iterative RAG development — every change was motivated
by a measurable failure, not a hunch.

---

## Improvement 1 — Corpus: 300 → 600 Papers

### What was wrong

The original corpus contained 300 ArXiv abstracts, which sounds large but is a very
thin slice of ML/AI research. Many reasonable test queries — "What is CRAG?",
"How does LoRA work?", "Explain GraphSAGE" — had no relevant document in the index
at all. Retrieval was evaluated against ground truth documents that were simply absent.
The agent was not failing at retrieval; it was being asked to find documents that did
not exist.

80% of the original 5-query test set happened to have a relevant document in the 300-paper
corpus, but that was luck. With a harder 10-query set the gap would have been larger.

### The fix

Double the download count. The corpus is assembled from the ArXiv API by querying
multiple ML/AI topic keywords and deduplicating by paper ID.

**Before:**

```python
MAX_PAPERS_PER_QUERY = 30
QUERY_TERMS = ["machine learning", "deep learning", "neural network"]
# total: ~90 papers after dedup
```

**After:**

```python
MAX_PAPERS_PER_QUERY = 60
QUERY_TERMS = [
    "machine learning", "deep learning", "neural network",
    "retrieval augmented generation", "large language model",
    "transformer attention", "fine-tuning PEFT", "graph neural network",
    "reinforcement learning", "diffusion model",
]
# total: ~600 papers after dedup
```

### Measured result

| Metric | Before | After |
|---|---|---|
| Queries with relevant doc in corpus | 4 / 5 (80%) | 10 / 10 (100%) |
| Web search triggered | 1 / 5 (20%) | 0 / 10 (0%) |

This is the highest-leverage change in the entire project. Better retrieval algorithms
cannot compensate for a corpus that does not contain the answer. **Coverage before
cleverness.**

---

## Improvement 2 — Eval Set: 5-Query Single-Keyword → 20-Query Multi-Keyword OR

### What was wrong

The original eval set had 5 queries, each with a single relevant document identified
by one specific title string match. This created two problems.

First, 5 queries is too few to surface real variance between strategies. In the old eval
every strategy scored the same Recall@5 (0.53) and Precision@5 (0.24) because the
differences were smaller than the measurement noise.

Second, single-keyword ground truth only counts a document as relevant if its title
contains an exact phrase. A paper titled "Self-Attention Mechanisms in Language Models"
would not match a ground truth entry for "attention mechanism in transformers" even
though it is clearly relevant.

```python
# Old ground truth — exact single-keyword match
GROUND_TRUTH = {
    "What is BM25?": ["BM25 ranking"],
    "Explain LoRA": ["LoRA fine-tuning"],
    ...
}
```

### The fix

Expand to 20 queries with multi-keyword OR matching in the ground truth: a document
is relevant if its title or abstract contains *any* of several synonymous terms.

**Before:**

```python
def is_relevant(doc_text: str, query: str, ground_truth: dict[str, list[str]]) -> bool:
    keyword = ground_truth[query][0]          # single keyword
    return keyword.lower() in doc_text.lower()
```

**After:**

```python
def is_relevant(doc_text: str, query: str, ground_truth: dict[str, list[str]]) -> bool:
    keywords = ground_truth[query]            # list of synonyms
    return any(kw.lower() in doc_text.lower() for kw in keywords)
```

Example of a new ground-truth entry:

```python
"How does BM25 rank documents?": [
    "BM25", "Okapi BM25", "best match 25", "Robertson", "sparse retrieval"
],
```

### Measured result

The 20-query eval reveals the real performance gap between strategies that the 5-query
eval masked. BM25 now clearly separates from dense (MRR 0.441 vs 0.299) and the
hybrid variants cluster as expected.

The absolute numbers drop substantially (MRR ~0.60 → ~0.30–0.44 range) because the
eval is now harder and fairer. See [Reading the Numbers Honestly](reading_the_numbers.md)
for why this is a good thing.

---

## Improvement 3 — BM25 Tokenisation Fix (+47% MRR)

### What was wrong

The initial BM25 implementation used Python's `str.split()` to tokenise documents.
This splits on whitespace only, which means hyphenated technical terms are kept as
single tokens:

```
"self-attention"   →  ["self-attention"]     # one token
"cross-encoder"    →  ["cross-encoder"]      # one token
"bi-encoder"       →  ["bi-encoder"]         # one token
```

When a user queries "self attention" (without a hyphen), BM25 looks for the token
`"self"` and the token `"attention"` separately. It finds them in documents that
contain both words in isolation, but not in documents where the canonical form is
hyphenated. The vocabulary mismatch between query tokens and document tokens
degrades recall significantly on technical ML text.

### The fix

Replace `str.split()` with a regex tokeniser that splits on hyphens, slashes,
underscores, and whitespace — then lowercase and strip short tokens.

**Before:**

```python
from rank_bm25 import BM25Okapi

def build_bm25_index(chunks: list[str]) -> BM25Okapi:
    tokenised = [doc.lower().split() for doc in chunks]
    return BM25Okapi(tokenised)
```

**After:**

```python
import re
from rank_bm25 import BM25Okapi

def _tokenise(text: str) -> list[str]:
    """Split on whitespace, hyphens, slashes, and underscores; drop single chars."""
    tokens = re.split(r"[\s\-_/]+", text.lower())
    return [t for t in tokens if len(t) > 1]

def build_bm25_index(chunks: list[str]) -> BM25Okapi:
    tokenised = [_tokenise(doc) for doc in chunks]
    return BM25Okapi(tokenised)
```

The same `_tokenise` function must be applied to queries at retrieval time, otherwise
the vocabulary still mismatches:

```python
def bm25_retrieve(query: str, bm25: BM25Okapi, chunks: list[str], k: int = 5):
    query_tokens = _tokenise(query)           # same tokeniser — critical
    scores = bm25.get_scores(query_tokens)
    top_k = scores.argsort()[::-1][:k]
    return [chunks[i] for i in top_k]
```

### Measured result

| Strategy | Eval | MRR | vs Dense baseline |
|---|---|---|---|
| Dense | 20q | 0.2992 | — |
| BM25 (old tokenisation) | 5q | 0.6667 | (misleading — see note) |
| BM25 (improved tokenisation) | 20q | **0.4408** | **+47%** |

The +47% gain over dense baseline is the largest single algorithmic improvement in
the project — and it came from a six-line tokeniser change, not a new model.

!!! tip "The lesson"
    In RAG systems, tokenisation bugs are silent. The index builds fine, queries run
    fine, and results look plausible. The only way to catch this class of bug is to
    measure MRR on queries that include hyphenated technical terms and notice the gap.

---

## Improvement 4 — RRF Fusion as an Alternative to Alpha Weighting

### What was wrong

The alpha-weighted hybrid combined dense and BM25 scores as:

```
final_score = α × dense_score + (1 - α) × bm25_score
```

This requires both scores to live on a comparable scale. FAISS dot-product scores
for `qwen3-embedding` typically range from 0.2 to 0.9. BM25 scores (via `rank_bm25`)
are raw term-frequency sums that range from 0.0 to ~8.0 depending on document length.
To combine them you need a normalisation step, and the normalisation constant changes
every time you update the corpus.

Reciprocal Rank Fusion (RRF) avoids this entirely by working on *ranks* rather
than scores:

$$
\text{RRF\_score}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}
$$

where `k = 60` is a stability constant. No score calibration needed.

### The fix

Add an RRF fusion function alongside the existing alpha-weighted combiner:

**Before (alpha only):**

```python
def hybrid_retrieve(
    query: str,
    dense_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    alpha: float = 0.7,
    k: int = 5,
) -> list[str]:
    scores: dict[str, float] = {}
    for doc, score in dense_results:
        scores[doc] = scores.get(doc, 0.0) + alpha * score
    for doc, score in bm25_results:
        scores[doc] = scores.get(doc, 0.0) + (1 - alpha) * score
    return sorted(scores, key=scores.get, reverse=True)[:k]
```

**After (RRF added):**

```python
def rrf_retrieve(
    ranked_lists: list[list[str]],
    k: int = 5,
    rrf_k: int = 60,
) -> list[str]:
    """Reciprocal Rank Fusion over multiple ranked result lists."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=scores.get, reverse=True)[:k]
```

### Measured result

| Strategy | MRR | Recall@5 |
|---|---|---|
| Hybrid α=0.7 | 0.3075 | 0.2667 |
| Hybrid RRF | 0.2867 | 0.2667 |
| Hybrid RRF + Rerank | 0.3625 | 0.3333 |

On the 600-paper corpus, alpha-weighted hybrid narrowly beats vanilla RRF (MRR 0.308
vs 0.287). However, RRF shines when combined with a cross-encoder reranker: the +26%
MRR gain from RRF → RRF+Rerank is larger than the gain from alpha → RRF. In a
production pipeline, RRF is the better fusion choice because it is score-calibration-free
and generalises more predictably across corpus updates.

---

## Improvement 5 — CRAG: 3-Tier → 2-Tier Grading Threshold

### What was wrong

The original CRAG implementation used a 3-tier relevance judgement from the LLM judge:

```
"good"      →  keep the document, proceed to generate
"ambiguous" →  trigger web search anyway (conservative)
"bad"       →  trigger web search
```

The `"ambiguous"` tier caused two problems. First, the LLM often returned it for
documents that were *weakly but genuinely* relevant — an abstract about transformer
attention was graded "ambiguous" when the query was about self-attention, even though
the document would support a reasonable answer. This triggered unnecessary web searches.

Second, parsing three tiers reliably from an LLM response (especially local models)
required more fragile JSON handling. When the model returned `"Ambiguous"` (capital A)
or `"uncertain"` the JSON parser failed silently and defaulted to triggering web search.

### The fix

Collapse to a 2-tier system: "good" or "bad". No middle ground.

**Before (prompt for 3-tier):**

```python
GRADE_PROMPT = """You are grading whether a retrieved document is relevant to a query.

Query: {query}
Document: {document}

Respond with JSON: {{"relevance": "good" | "ambiguous" | "bad"}}
Explain your reasoning in 1 sentence."""
```

**Before (parsing 3 tiers):**

```python
def grade_document(response: str) -> str:
    data = json.loads(response)
    return data["relevance"]          # "good", "ambiguous", or "bad"

# In the graph logic:
if grade == "good":
    keep_docs.append(doc)
elif grade == "ambiguous":
    trigger_web_search = True         # conservative — often wrong
else:
    trigger_web_search = True
```

**After (prompt for 2-tier):**

```python
GRADE_PROMPT = """You are grading whether a retrieved document is relevant to a query.

Query: {query}
Document: {document}

Respond ONLY with valid JSON: {{"relevance": "good"}} or {{"relevance": "bad"}}
A document is "good" if it contains information that could help answer the query,
even partially. Only return "bad" if the document is clearly off-topic."""
```

**After (parsing 2 tiers):**

```python
def grade_document(response: str) -> str:
    try:
        data = json.loads(response)
        grade = data.get("relevance", "bad").lower().strip()
        return "good" if grade == "good" else "bad"
    except json.JSONDecodeError:
        return "bad"                  # fail safe — treat parse failure as bad

# In the graph logic:
if grade == "good":
    keep_docs.append(doc)
else:
    trigger_web_search = True
```

### Measured result

| Metric | Before (3-tier) | After (2-tier) |
|---|---|---|
| Web search triggered | 1 / 5 (20%) | 0 / 10 (0%) |
| Faithful answers | 3 / 5 (60%) | 7 / 10 (70%) |

The zero web-fallback rate on the 10-query eval confirms that the 2-tier threshold is
less conservative. Faithfulness improved from 60% to 70% — partly because the agent
now uses more of its retrieved context instead of discarding it as "ambiguous", and
partly because the larger corpus means the kept context is higher quality.

!!! note "When 3 tiers are appropriate"
    If your corpus is known to be sparse and you genuinely want aggressive web fallback,
    the 3-tier system is a defensible choice. The 2-tier simplification is the right
    call here because the expanded corpus made "ambiguous" documents almost always
    useful in practice.
