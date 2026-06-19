# Loading Data

Before any embedding or indexing can happen, we need documents. This page explains where
they come from, how they are filtered to stay on-topic, and what the data looks like
when it arrives in Python.

---

## The dataset — `ccdv/arxiv-summarization`

The corpus comes from [ccdv/arxiv-summarization](https://huggingface.co/datasets/ccdv/arxiv-summarization)
on HuggingFace. It contains roughly 203,000 academic papers scraped from ArXiv, each stored as
a `(article, abstract)` pair. The `article` field is the full paper text; the `abstract` field
is the structured summary the authors wrote.

We use only the `abstract` field because:

- Abstracts are self-contained. They summarise the paper's problem, method, and results in
  150–400 words. A question about "how LoRA reduces memory" is answerable from the abstract
  alone.
- Abstracts are fast to embed. Processing full papers would require a much larger
  chunk budget and significantly longer indexing time.
- Noise is lower. Full papers contain section headings, references, and mathematical notation
  that degrade embedding quality.

!!! info "Definition — Embedding quality"
    Embedding quality refers to how well a text's vector captures its meaning.
    Dense boilerplate text (reference lists, captions, repeated headers) can dilute
    the semantic signal in a vector, making it harder to distinguish a paper on
    "attention mechanisms" from one on "image segmentation."

The dataset spans all of ArXiv — physics, mathematics, chemistry, computer science. For
a research assistant focused on ML and AI, papers about topology or organic chemistry add
noise without adding relevant signal. The ML keyword filter solves this.

---

## The `load_hf_papers()` function

```python
def load_hf_papers(
    n_samples: int = 500,
    split: str = "train",
    ml_filter: bool = True,
    scan_multiplier: int = 10,
) -> list[dict]:
```

This function does three things:

**1. Load a slice of the dataset**

```python
scan_size = n_samples * scan_multiplier
offset = 5000  # skip earliest rows (heavy math/stats bias)
raw = load_dataset(
    "ccdv/arxiv-summarization",
    split=f"{split}[{offset}:{offset + scan_size}]",
)
```

We ask for `n_samples * scan_multiplier` rows (default: 600 × 10 = 6,000) because only
about 10–15% of all ArXiv papers are ML-related. The offset of 5,000 skips the earliest
rows, which are disproportionately maths and statistics papers.

**2. Apply the ML keyword filter**

```python
ML_KEYWORDS = [
    "neural network", "deep learning", "machine learning", "transformer",
    "attention mechanism", "language model", "reinforcement learning",
    "classification", "object detection", "generative model", "diffusion",
    "fine-tuning", "pre-training", "bert", "gpt", "llm", "embedding",
    # ... 14 more keywords
]

if ml_filter:
    abstract_lower = abstract.lower()
    if not any(kw in abstract_lower for kw in ML_KEYWORDS):
        continue
```

This is OR logic — a paper is kept if its abstract contains *any* of the listed keywords.
The list covers the major subfields: NLP (`bert`, `gpt`, `language model`), computer vision
(`object detection`, `convolutional`), generative models (`diffusion`, `generative model`),
and general ML (`gradient descent`, `training`, `benchmark`).

**3. Construct normalised paper dicts**

The HuggingFace dataset does not always have a clean `title` field. We extract the first
sentence of the abstract as a stand-in title:

```python
first_sentence = abstract.split(".")[0].strip()
title = first_sentence[:120] + ("..." if len(first_sentence) > 120 else "")
paper_id = f"arxiv_{split}_{offset + idx:05d}"
```

---

## What a paper dict looks like

Every paper is stored as a plain Python dict. Here is a representative example:

```python
{
    "id":       "arxiv_train_05042",
    "title":    "We propose a novel attention mechanism for transformer models that...",
    "abstract": "We propose a novel attention mechanism for transformer models that "
                "dynamically weights each head based on task context. Our method, "
                "HydraHead, reduces inference cost by 23% while maintaining benchmark "
                "accuracy on GLUE and SuperGLUE. Experiments on eight downstream "
                "classification tasks confirm that head specialisation is learnable "
                "without additional supervision.",
    "category": "cs.AI",
    "url":      "https://arxiv.org/abs/arxiv_train_05042",
}
```

The `id` field is the primary key. It links paper dicts to their chunks (which carry a
`paper_id` field) and to evaluation ground-truth (the `relevant_ids` list in each eval query).

!!! note "The arxiv.org primary path"
    `load_hf_papers()` is the fallback. The primary function `load_arxiv_papers()` tries
    to fetch from `arxiv.org` first (filtered to `cs.CL`, `cs.AI`, `cs.LG`). If the network
    is unavailable, it automatically falls back to `load_hf_papers()`. For offline use or
    reproducibility, call `load_hf_papers()` directly.

---

## What's next

Now that we have 600 paper dicts, the next step is to split each abstract into smaller
pieces that the embedding model can handle well. Head to [Chunking](chunking.md) to see
how the 512-character window strategy works and why overlap matters.
