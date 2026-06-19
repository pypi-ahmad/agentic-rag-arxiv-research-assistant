# Further Reading

Foundational papers, benchmark datasets, and next-step projects for going deeper
on every major topic in this tutorial.

---

## Foundational Papers

### CRAG — Corrective Retrieval-Augmented Generation

**Yan, S., Gu, J., Zhu, Y., & Ling, Z. (2024).** *Corrective Retrieval Augmented Generation.* arXiv:2401.15884.

The paper that introduced the CRAG architecture implemented in Part 3 of this tutorial.
Key contributions: the relevance grading step that evaluates each retrieved document
before generation, the web search fallback for queries the corpus cannot answer, and
the knowledge refinement step that strips irrelevant passages from kept documents.
The paper uses a different implementation (GPT-3.5, Bing search, a fine-tuned retrieval
evaluator) but the core architecture maps directly to the LangGraph implementation here.

- [arXiv:2401.15884](https://arxiv.org/abs/2401.15884)

---

### RAG — Retrieval-Augmented Generation (original)

**Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020).** *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* arXiv:2005.11401. NeurIPS 2020.

The paper that coined the term RAG. Proposes combining a pre-trained retriever (DPR)
with a seq2seq generator (BART) in an end-to-end trainable system. The "RAG-Sequence"
and "RAG-Token" variants introduced here are the conceptual ancestors of every RAG
system built since. Essential reading for understanding why the paradigm exists and
what it was originally designed to solve.

- [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- [NeurIPS 2020 proceedings](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html)

---

### BM25

**Robertson, S., & Walker, S. (1994).** *Some simple effective approximations to the 2-Poisson model for probabilistic weighted retrieval.* SIGIR 1994.

The original BM25 paper. Robertson and Walker derive the BM25 formula from first
principles using a 2-Poisson model of term frequency distributions. The "25" in BM25
refers to this being the 25th parameterisation in a series of experiments. This paper
is short (4 pages) and worth reading to understand why the saturation function and
length normalisation terms in BM25 take the specific forms they do.

Also see the later formalisation:

**Robertson, S., & Zaragoza, H. (2009).** *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval, 3(4), 333–389.

- [SIGIR 1994 (via ACM)](https://dl.acm.org/doi/10.5555/188490.188561)
- [Robertson & Zaragoza 2009 (FnTIR)](https://www.nowpublishers.com/article/Details/INR-019)

---

### MS MARCO

**Bajaj, P., Campos, D., Craswell, N., Deng, L., Gao, J., Liu, X., Majumder, R., McNamara, A., Mitra, B., Nguyen, T., Rosenberg, M., Song, X., Stocker, M., Tiwary, S., & Wang, M. (2016).** *MS MARCO: A Human Generated MAchine Reading COmprehension Dataset.* arXiv:1611.09268.

The dataset used to train the cross-encoder reranker used in Part 2 of this tutorial
(`cross-encoder/ms-marco-MiniLM-L-6-v2`). MS MARCO contains 1 million real Bing
queries with relevant passages selected by human annotators from web documents.
Understanding the training distribution of your reranker helps explain where it excels
(web-style queries) and where it may underperform (highly technical domain-specific queries).

- [arXiv:1611.09268](https://arxiv.org/abs/1611.09268)
- [MS MARCO website](https://microsoft.github.io/msmarco/)

---

### Dense Passage Retrieval (DPR)

**Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. (2020).** *Dense Passage Retrieval for Open-Domain Question Answering.* arXiv:2004.04906. EMNLP 2020.

The paper that established the modern bi-encoder paradigm for dense retrieval. DPR
trains separate encoders for questions and passages using in-batch negatives, achieving
large gains over BM25 on open-domain QA. The intuition for when dense retrieval wins
over BM25 (semantic paraphrase, question-to-passage vocabulary mismatch) comes from
this work.

- [arXiv:2004.04906](https://arxiv.org/abs/2004.04906)

---

### MTEB — Massive Text Embedding Benchmark

**Muennighoff, N., Tazi, N., Magne, L., & Reimers, N. (2022).** *MTEB: Massive Text Embedding Benchmark.* arXiv:2210.07316. EACL 2023.

The benchmark used to evaluate and compare embedding models, including `qwen3-embedding`.
MTEB covers 56 datasets across 8 task categories (retrieval, clustering, classification,
reranking, summarisation, pair classification, STS, and bitext mining). Consulting the
MTEB leaderboard before choosing an embedding model is the recommended practice.

- [arXiv:2210.07316](https://arxiv.org/abs/2210.07316)
- [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

---

## Next Steps

### Multi-hop RAG

Standard RAG retrieves documents in a single shot. Multi-hop RAG decomposes complex
questions into sub-questions, retrieves for each, and synthesises the sub-answers.
For example, "Which 2023 paper on LoRA achieved the highest GLUE score?" requires
first finding LoRA papers from 2023, then finding the evaluation results within them.

Starting point: **IRCoT (Interleaving Retrieval with Chain-of-Thought)** —
Trivedi et al. (2022), arXiv:2212.10509.

---

### Self-querying retrieval

Instead of using the user's raw query for retrieval, prompt an LLM to generate a better
query — one that includes synonyms, removes ambiguity, or adds metadata constraints
(e.g., "papers from 2023 about transformer attention"). LangChain's `SelfQueryRetriever`
provides a ready implementation.

Relevant: **Query2Doc** — Wang et al. (2023), arXiv:2303.07678.

---

### Streaming responses

The current implementation generates a complete answer before displaying it. For
interactive applications, streaming the tokens as they are generated provides a much
better user experience. Ollama supports streaming via the `stream=True` parameter, and
LangChain provides streaming callbacks for LangGraph nodes.

```python
# Streaming with Ollama
for chunk in ollama.chat(model="granite4.1:8b", messages=[...], stream=True):
    print(chunk["message"]["content"], end="", flush=True)
```

---

### Production deployment

The current system is a research prototype. Moving to production requires:

- **Vector database**: replace FAISS (in-memory) with Pinecone, Weaviate, or Qdrant
  for persistence, multi-user access, and metadata filtering.
- **Serving**: wrap the LangGraph graph in a FastAPI endpoint.
- **Caching**: cache embeddings and retrieval results for repeated queries.
- **Monitoring**: log retrieval quality, answer faithfulness, and latency per query.
  MLflow or W&B work well for this.
- **Index updates**: add a scheduled job that pulls new ArXiv papers and upserts
  their embeddings into the vector store.

---

## Projects to Explore

### LlamaIndex

A data framework for LLM-powered applications, with a particularly strong focus on
document ingestion, chunking strategies, and index management. LlamaIndex has built-in
support for hierarchical chunking, sentence-window retrieval, and document summary
indexing — strategies that go beyond the simple abstract-as-chunk approach used here.

[llamaindex.ai](https://www.llamaindex.ai) · [GitHub](https://github.com/run-llama/llama_index)

---

### DSPy

A framework for *programming* (not prompting) LLMs. Instead of manually writing and
tuning prompts, DSPy uses optimisers that automatically find the best prompts given a
training set and a metric. Particularly useful for systematically improving CRAG-style
prompts (the grading prompt, the generation prompt, the hallucination prompt) without
manual iteration.

[dspy.ai](https://dspy.ai) · [GitHub](https://github.com/stanfordnlp/dspy)

---

### RAGAS — RAG Assessment

An evaluation framework specifically designed for RAG pipelines. RAGAS computes four
metrics automatically using LLM-as-judge:

- **Faithfulness** — are the claims in the answer supported by the retrieved context?
- **Answer relevance** — does the answer actually address the question?
- **Context precision** — what fraction of the retrieved context is relevant?
- **Context recall** — does the retrieved context contain all the information needed?

RAGAS is the recommended next step for a more rigorous automated eval than the
MRR/Recall@k metrics used in this tutorial.

[docs.ragas.io](https://docs.ragas.io) · [GitHub](https://github.com/explodinggradients/ragas)

---

### Sentence Transformers

The library used under the hood for the cross-encoder reranker. The full library
provides bi-encoders, cross-encoders, and dense retrieval utilities, with over 6,000
pre-trained models on HuggingFace. If you want to fine-tune your own embedding model
on domain-specific text (e.g., ArXiv ML papers), Sentence Transformers provides the
training infrastructure.

[sbert.net](https://www.sbert.net) · [GitHub](https://github.com/UKPLab/sentence-transformers)
