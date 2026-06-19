# BM25 and Sparse Retrieval

Semantic search with embeddings is powerful, but it has a specific and predictable
failure mode. Understanding it explains why BM25 exists as a complement to vector
search — and why the two together outperform either alone.

---

## The Limits of Semantic Search

Consider these two queries:

1. "How do transformer models handle long sequences?"
2. "What does the Mamba SSM paper claim about RWKV?"

Query 1 is a conceptual question. Even if the relevant papers use different words —
"long-context window", "extended attention span", "sequence length scaling" — an
embedding model can find them, because all those phrases map to nearby regions of
vector space.

Query 2 is an exact-term question. The words "Mamba", "SSM", and "RWKV" are
proper nouns — specific model names. They appeared rarely or never in the embedding
model's training data. The model cannot reliably distinguish "Mamba SSM" from any
other sequence model just by the *meaning* of the name; the name itself is the signal.

This is where semantic search loses to classical keyword search:

- **Model names**: Mamba, RWKV, LLaMA, Mistral, Qwen
- **Dataset names**: SQuAD, MMLU, HumanEval, BEIR
- **Acronyms**: RAG, LoRA, QLoRA, PEFT, RLHF
- **Paper IDs**: arXiv:2305.10403
- **Rare technical terms** that the embedding model has never seen in context

For all of these, a retriever that matches exact tokens beats one that reasons about
meaning.

!!! note "The core insight"
    Embeddings encode *generalised meaning*. Keyword search encodes *exact tokens*.
    They are complementary, not competing. This is why Part 2 of this tutorial
    introduces hybrid retrieval: dense + sparse together.

---

## What Is Sparse Retrieval?

A "sparse" representation of a document is a vector indexed by vocabulary. Most entries
are zero (the word does not appear); a few are non-zero (the word does appear).

Imagine a vocabulary of 100,000 English words. A 200-word paper abstract might contain
150 unique words — so its sparse vector has 99,850 zeros and 150 non-zero values.
Compare this to a dense embedding: 1024 non-zero values for every document, regardless
of content.

| Property | Dense (embedding) | Sparse (BM25 / TF-IDF) |
|---|---|---|
| Vector size | Fixed (e.g. 1024) | Vocabulary size (~100K+) |
| Non-zero entries | All 1024 | A few hundred per doc |
| Captures | Semantic meaning | Exact token overlap |
| Handles new vocab | Poorly (OOV) | Well (new term = new dimension) |
| Main algorithm | Cosine similarity | BM25 |

---

## The BM25 Formula, Explained Step by Step

BM25 (Best Match 25) was published in 1994 and has remained the standard for
keyword search for thirty years. Here is the scoring formula:

For a query Q with terms $q_1, q_2, \ldots, q_n$ and a document D:

$$
\text{score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}
$$

That looks intimidating. Let's unpack each piece.

### Term Frequency: f(q, D)

$f(q, D)$ is simply the count of how many times the query term $q$ appears in document $D$.

A document that uses the word "attention" 10 times is probably more about attention
than one that uses it once. Term frequency captures this.

### Inverse Document Frequency: IDF(q)

$$\text{IDF}(q) = \log\!\left(\frac{N - \text{df}(q) + 0.5}{\text{df}(q) + 0.5}\right)$$

Where N is the total number of documents and df(q) is the number of documents that
contain term q.

IDF penalises terms that appear in many documents. The word "the" appears in every
document — its IDF is nearly zero, so it contributes almost nothing to the score.
The term "Mamba" appears in very few documents — its IDF is high, so a match on it
strongly boosts the score.

This is the "rare terms matter more" principle.

### Saturation: the k₁ term

Notice the numerator and denominator both contain $f(q, D)$. When $f(q, D)$ is
small, the fraction grows quickly. But as $f(q, D)$ gets very large, the fraction
approaches $(k_1 + 1)$ — it saturates.

!!! example "Why saturation matters"
    If document A mentions "attention" 5 times and document B mentions it 50 times,
    document B is not 10x more relevant. After a point, more occurrences stop adding
    information. BM25 models this saturation explicitly. Plain TF-IDF does not —
    which is one of the reasons BM25 outperforms TF-IDF.

The default value $k_1 = 1.5$ means scores saturate at around 3–5 term occurrences.

### Length Normalisation: the b term

$$1 - b + b \cdot \frac{|D|}{\text{avgdl}}$$

$|D|$ is the length of the document; $\text{avgdl}$ is the average document length
across the corpus. The parameter $b = 0.75$ controls how strongly length is penalised.

A long document has a higher chance of containing a query term just by chance. Length
normalisation corrects for this: a term that appears once in a short abstract is
treated as more significant than the same term appearing once in a 10-page paper.

!!! info "BM25 vs TF-IDF: the short version"
    TF-IDF: score = TF × IDF. Simple product, no saturation, no length normalisation.
    BM25: adds saturation (k₁) and length normalisation (b). These two additions are
    why BM25 consistently outperforms TF-IDF on benchmark retrieval tasks.

---

## How BM25 Is Built in This Project

`BM25Retriever` in `src/retriever.py` wraps the `rank_bm25` library:

```python
# src/retriever.py
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, chunks: list[dict]) -> None:
        tokenised_corpus = [_tokenise(chunk["text"]) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenised_corpus)

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        query_tokens = _tokenise(query)
        scores = self.bm25.get_scores(query_tokens)
        top_k_indices = np.argsort(scores)[::-1][:k]
        ...
```

`BM25Okapi` is the most common variant of BM25 (the "Okapi" refers to the Okapi
system at City University London where the formula was developed). It uses
$k_1 = 1.5$ and $b = 0.75$ as defaults — well-validated parameters across many
document retrieval benchmarks.

---

## Our Tokenisation Improvement

Tokenisation determines what "terms" BM25 scores on. The choice matters more than
most tutorials acknowledge.

### The Old Approach (whitespace split only)

```python
# OLD — naive whitespace split
tokens = text.lower().split()
```

**Problems with this approach:**

- `"Transformer."` and `"Transformer"` are different tokens (trailing punctuation).
- `"the"`, `"is"`, `"of"` consume IDF budget — they appear in nearly every document,
  their IDF is close to zero, but the model still processes them on every query.
- `"a"` and `"A"` would be different tokens if `.lower()` is missed anywhere.

### The New Approach (lowercase + punctuation strip + stop words)

```python
# src/retriever.py — _tokenise()

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

What each step does:

1. `.lower()` — case-fold everything so `"Transformer"` and `"transformer"` are
   the same token.
2. `re.sub(r"[^a-z0-9]", " ", ...)` — replace every non-alphanumeric character
   with a space. Punctuation, hyphens, colons all become word boundaries.
   `"chain-of-thought"` becomes three tokens: `["chain", "thought"]` (after stop
   word removal).
3. Stop word filter — removes ~50 common function words that carry no semantic
   signal for BM25. This frees up the IDF budget for terms that actually matter.
4. Single-character filter (`len(t) > 1`) — drops leftover single characters
   from punctuation splitting.

!!! example "Before and after tokenisation"
    Input: `"The attention mechanism in transformer-based LLMs."`

    **Old (whitespace split):**
    `["the", "attention", "mechanism", "in", "transformer-based", "llms."]`

    **New (improved):**
    `["attention", "mechanism", "transformer", "based", "llms"]`

    The improved version removes stop words ("the", "in"), strips punctuation
    from "llms.", and splits the hyphenated compound — giving BM25 cleaner,
    denser signal to work with.

The same `_tokenise()` function is applied to **both** the corpus at index-build
time and the query at search time. This symmetry is mandatory: BM25 scores
are only meaningful when the corpus tokens and the query tokens come from the
same vocabulary.

---

## When BM25 Wins and When It Loses

**BM25 wins when:**

- The query contains specific technical terms, proper nouns, or acronyms that
  are unlikely to have rich semantic representations in the embedding space.
- The relevant document uses the exact same terminology as the query — no
  paraphrasing involved.
- The answer is "find me everything about LoRA" rather than "find me things
  conceptually similar to parameter-efficient fine-tuning".

**BM25 loses when:**

- The query and the relevant document use completely different vocabulary.
  ("How do LLMs generate text?" vs a paper discussing "autoregressive decoding"
  — zero keyword overlap, but highly relevant.)
- The query is long and conversational — many tokens, most of which are stop words
  that BM25 will discard.
- The relevant documents are short — length normalisation compresses scores and
  ranking becomes noisy at very short chunk lengths.

This complementary failure-mode profile is exactly why hybrid retrieval (combining
dense and BM25) is the subject of Part 2. BM25 catches what embeddings miss;
embeddings catch what BM25 misses.

---

## Summary

| Concept | Detail |
|---|---|
| BM25 variant | BM25Okapi (k₁ = 1.5, b = 0.75) |
| Library | `rank_bm25` |
| Tokenisation | lowercase + punct strip + stop word removal |
| Index built over | All chunk texts (~1,200 chunks) |
| Scores | Raw BM25 values (not normalised to 0–1) |
| Best for | Exact-term queries, proper nouns, acronyms |
| Worst for | Paraphrase and conceptual queries |

The next part of this tutorial builds the complete Naive RAG pipeline, combining
everything you have learned here: embed the corpus, index it in FAISS, and generate
answers with Granite 4.1 — end to end, fully local.
