# Glossary

Every technical term used in this tutorial, alphabetically sorted.

---

**Alpha weighting** — A score-fusion technique that combines two ranked result lists
by computing a weighted sum of their scores: `final = α × score_A + (1-α) × score_B`.
The parameter α controls how much each source contributes. α=0.7 means 70% dense,
30% BM25. Because the two sources may produce scores on different scales, alpha
weighting typically requires normalisation before fusion. In this project: used in
`HybridRetriever` to blend FAISS dense scores and BM25 scores, with α=0.7 as the
default.

---

**Agentic AI** — An AI architecture where an LLM does not just respond to a single
prompt but takes a sequence of actions, observes results, and decides what to do next
based on intermediate outcomes. The LLM acts as a *reasoner* that drives a loop rather
than a function that maps input to output. Agentic systems can use tools, call external
APIs, perform web searches, and grade their own outputs. In this project: the CRAG
systems in `03_agentic_rag_langgraph.ipynb` and `07_agentic_rag.ipynb` are agentic pipelines where the LLM grades
retrieved documents, decides whether to search the web, generates an answer, and grades
its own answer for hallucinations.

---

**ArXiv** — A free, open-access repository of preprint scientific papers, primarily
in physics, mathematics, computer science, and quantitative biology. Most ML and AI
research is posted here before (and sometimes instead of) formal journal publication.
In this project: the current baseline uses 4,000 ML/AI abstracts from
`ccdv/arxiv-summarization` (ML-filtered), with legacy 600-paper results retained
for historical comparison.

---

**Bi-encoder** — A neural network architecture for information retrieval where the
query and each document are encoded *independently* by the same (or separate) encoder,
producing fixed-length vector representations. Similarity is then computed by a simple
dot product or cosine similarity between the two vectors. Because documents can be
pre-encoded, bi-encoders scale efficiently to millions of documents. The trade-off is
that the query and document never attend to each other during encoding. In this project:
`qwen3-embedding` is the bi-encoder used to produce 1024-dimensional vectors for every
chunk and for every query.

---

**BM25** — (Best Match 25) A probabilistic keyword-based ranking function that scores
documents by how well they match a query's terms, weighted by term frequency (TF) and
inverse document frequency (IDF), with length normalisation. The "25" refers to the
25th iteration of the BM series of retrieval models developed at City University London.
BM25 is the standard baseline for information retrieval and is hard to beat on
keyword-heavy queries. In this project: implemented via the `rank_bm25` library in
`src/retriever.py`, using a custom tokeniser that splits on hyphens and slashes to
handle technical ML vocabulary.

---

**ChromaDB** — An open-source, embedded vector database with a Python-first API.
It stores embeddings alongside metadata and supports similarity search, filtering,
and persistence to disk. ChromaDB is popular for prototypes and local development
because it requires no separate server process. In this project: used in Part 4
GraphRAG as one backend variant, alongside an optional Pinecone backend.

---

**Chunking** — The process of splitting a long document into smaller, overlapping or
non-overlapping segments before embedding. LLM context windows and embedding models
both have token limits, so raw documents must be chunked before indexing. Chunk size
is a critical hyperparameter: too small and individual chunks lose context; too large
and the embedding captures too much irrelevant material. In this project: each ArXiv
abstract is treated as a single chunk (abstracts are naturally short, typically
150–250 words), so no splitting is required. The chunk *is* the abstract.

---

**Conditional edge** — In LangGraph, an edge whose target node is determined at
runtime by a function that inspects the current graph state. Unlike regular edges
(which always route to the same next node), conditional edges implement branching
logic — the equivalent of an `if` statement in a state machine. In this project: used
in the CRAG graph to route from the `grade_documents` node to either `generate`
(if relevant documents were found) or `web_search` (if all documents were graded bad).

---

**Cosine similarity** — A measure of similarity between two non-zero vectors that
computes the cosine of the angle between them:
`cos(θ) = (a · b) / (‖a‖ × ‖b‖)`. The result is 1.0 for identical direction
(same meaning), 0.0 for perpendicular vectors (unrelated topics), and -1.0 for
opposite directions. Cosine similarity is the standard metric for embedding-based
retrieval because it is scale-invariant — a long document and a short document with
the same meaning produce vectors of different magnitude but the same direction. In
this project: `IndexFlatIP` in FAISS computes dot products, which equals cosine
similarity when vectors are L2-normalised (which `qwen3-embedding` produces by default).

---

**CRAG** — (Corrective Retrieval-Augmented Generation) An agentic RAG architecture
introduced by Yan et al. (2024) that adds a self-correction loop to standard RAG. The
system grades each retrieved document for relevance. If all documents are graded
irrelevant, it triggers a web search to supplement the corpus. After generating an
answer, the system grades the answer for faithfulness and can regenerate if hallucination
is detected. In this project: implemented in the legacy LangGraph track
(`03_agentic_rag_langgraph.ipynb`) and extended in Part 5 CRAG (`08_crag.ipynb`),
with LLM-as-judge grading for retrieval and final answers.

---

**Cross-encoder** — A neural network architecture for relevance scoring where the
query and a candidate document are concatenated and processed *together* by the encoder.
Unlike a bi-encoder, the query and document attend to each other during encoding,
allowing the model to capture fine-grained relevance signals. Cross-encoders are
significantly more accurate than bi-encoders but cannot pre-encode documents — they
must run at query time for every candidate, making them too slow for first-stage
retrieval over large corpora. They are used as a reranker over a small candidate set
(typically the top 20–50 from first-stage retrieval). In this project: the
`cross-encoder/ms-marco-MiniLM-L-6-v2` model from `sentence-transformers` is used to
rerank the top-20 candidates from RRF fusion, selecting the final top-5.

---

**Dense retrieval** — A retrieval paradigm that represents both queries and documents
as dense floating-point vectors (embeddings) and finds relevant documents by nearest-
neighbour search in the embedding space. Dense retrieval excels at semantic similarity:
it can find documents whose meaning matches the query even when they share no
vocabulary. Its weakness is exact keyword matching — low-frequency technical terms
that appear rarely in training data produce poorly discriminative embeddings. In this
project: FAISS with `qwen3-embedding` is the dense retrieval base for the core
pipeline and is reused across advanced variants.

---

**DSPy** — A framework for programming (rather than prompting) language models, using
optimisers to automatically tune prompts and few-shot examples based on a downstream
metric. In this project: not used, but listed in Further Reading as a natural next step
for systematically optimising the CRAG prompts.

---

**DuckDuckGo** — A privacy-focused web search engine. In this project: the CRAG agent
uses the `duckduckgo_search` Python library to perform web searches when the local
corpus cannot answer a query. The library wraps DuckDuckGo's public API and returns
organic search results as structured Python objects.

---

**Embedding** — A dense, fixed-length numerical vector that represents a piece of text
in a high-dimensional space where semantic similarity corresponds to geometric proximity.
Embedding models are neural networks trained to map text to vectors such that similar
texts produce similar vectors. Modern embedding models produce 384 to 4096-dimensional
vectors. In this project: `qwen3-embedding` (via Ollama) produces 1024-dimensional
vectors for every abstract chunk and for every query at retrieval time.

---

**FAISS** — (Facebook AI Similarity Search) An open-source library by Meta AI Research
for efficient similarity search and clustering of dense vectors. FAISS implements
dozens of index types with different accuracy/speed/memory trade-offs. It is the
standard tool for embedding-based retrieval at scale. In this project: `IndexFlatIP`
(exact inner-product search) is used for tutorial-scale corpora, and the index is
saved to `artifacts/faiss_index/index.bin` after Part 1 for downstream loading.

---

**Faithfulness** — A property of an LLM-generated answer: the answer is *faithful* if
every claim it makes is supported by (or at least not contradicted by) the retrieved
documents. An unfaithful answer introduces information that is not in the context —
this is a form of hallucination. In this project: faithfulness is evaluated by the
LLM-as-judge node in the CRAG graph, which checks whether the generated answer is
grounded in the provided document chunks.

---

**granite4.1** — (specifically `granite4.1:8b`) IBM's Granite 4.1 8-billion-parameter
instruction-tuned language model, available through Ollama. Used as the generative LLM
for producing answers from retrieved context. Granite 4.1 is a capable open-weights
model that runs locally on consumer hardware. In this project: the generation node
calls `ollama.chat()` with `granite4.1:8b` to produce all answers, and the same model
acts as the LLM judge for document grading and hallucination detection.

---

**Grounding** — The process of anchoring an LLM's response to specific, verifiable
source material. A grounded answer refers to and is consistent with the retrieved
documents. Grounding is what RAG buys over pure LLM generation: the model cannot
fabricate facts that contradict the explicit text in the context window. In this
project: grounding is enforced by always passing retrieved chunks as context in the
prompt, and verified by the faithfulness grading step in the CRAG agent.

---

**Hallucination** — A generated statement that is factually incorrect, fabricated, or
not supported by the provided context. Hallucination arises because LLMs are trained to
predict plausible next tokens, not to verify factual accuracy. Hallucination is not
random noise — it tends to produce confident, internally consistent, plausible-sounding
text, which makes it hard to detect without a reference to check against. In this
project: the CRAG agent includes a hallucination-detection step where the LLM judge
reads the generated answer alongside the source chunks and flags answers that introduce
unsupported claims.

---

**Hybrid search** — A retrieval strategy that combines results from multiple retrieval
systems (typically dense and sparse) to compensate for the weaknesses of each. Dense
retrieval handles semantic paraphrase; sparse retrieval handles exact keyword matching.
Their combination typically outperforms either alone on diverse query sets. Fusion can
be done by score weighting (alpha blend) or by rank fusion (RRF). In this project:
hybrid retrieval is the core contribution of Part 2, combining FAISS dense search and
BM25 sparse search.

---

**IDF** — (Inverse Document Frequency) A component of BM25 and TF-IDF scoring that
measures how rare a term is across the entire corpus. IDF for term `t` is
`log((N - df_t + 0.5) / (df_t + 0.5))` where `N` is the total number of documents and
`df_t` is the number of documents containing `t`. Common words ("the", "is", "of")
have low IDF; rare technical terms ("GraphSAGE", "CRAG", "LoRA") have high IDF. In
BM25, high-IDF terms contribute much more to the relevance score than low-IDF terms.
In this project: IDF is computed automatically by `rank_bm25` over the active
corpus used for the run (legacy 600 or current 4,000 baseline).

---

**IndexFlatIP** — The specific FAISS index type used in this project. "Flat" means the
index stores all vectors in a flat array and performs an exhaustive (exact) search —
no approximation. "IP" stands for inner product, which equals cosine similarity when
vectors are L2-normalised. The trade-off: exact accuracy but O(n) search time. For
4,000 documents this is still practical for exact tutorial-scale retrieval; for
millions of documents you would switch to an
approximate index like `IndexIVFFlat` or `IndexHNSWFlat`. In this project: `IndexFlatIP`
is built in `01_naive_rag.ipynb` and saved to `artifacts/faiss_index/index.bin`.

---

**LangGraph** — A library built on top of LangChain that enables the construction of
stateful, multi-step agent workflows as directed graphs. Nodes are Python functions
that read from and write to a shared state dictionary. Edges define the control flow
between nodes, with conditional edges enabling runtime branching. LangGraph compiles
the graph before execution (producing a `CompiledStateGraph`) and manages state
persistence across steps. In this project: the CRAG state machine in Part 3 is
implemented as a LangGraph graph with nodes for retrieval, grading, web search,
generation, and hallucination checking.

---

**LLM-as-judge** — A technique where a language model evaluates the output of another
language model (or its own output). The judge LLM is given an evaluation rubric,
the input, and the output to assess, and returns a structured score or label. LLM-as-
judge is widely used in RAG evaluation because human annotation is slow and expensive.
Its weakness is that the judge inherits the LLM's biases — it may prefer longer answers
or authoritative-sounding language regardless of correctness. In this project: the
same `granite4.1:8b` model acts as judge for document relevance grading and answer
faithfulness checking in the CRAG agent.

---

**LoRA** — (Low-Rank Adaptation) A parameter-efficient fine-tuning technique that
injects small trainable rank-decomposition matrices into the weight matrices of a
frozen pre-trained model. LoRA dramatically reduces the number of trainable parameters
(typically by 10,000×) and thus GPU memory requirements during fine-tuning. In this
project: LoRA appears as a topic in the ArXiv corpus (several papers on LoRA/QLoRA
are indexed), and "How does LoRA reduce memory consumption?" is one of the evaluation
queries.

---

**MTEB** — (Massive Text Embedding Benchmark) A comprehensive benchmark for evaluating
text embedding models across dozens of tasks including retrieval, clustering, classification,
and semantic textual similarity. MTEB is the standard reference for comparing embedding
models. Higher MTEB retrieval score generally predicts better RAG performance. In this
project: `qwen3-embedding` was selected partly based on its strong MTEB performance
relative to its size, making it suitable for local deployment.

---

**MRR** — (Mean Reciprocal Rank) An evaluation metric for ranked retrieval systems.
For each query, the reciprocal rank is `1 / rank` where `rank` is the position of the
first relevant document in the result list. MRR is the average of these reciprocal ranks
across all queries. A system that always returns the relevant document at rank 1 has
MRR = 1.0; one that always puts it at rank 5 has MRR = 0.2. MRR is the primary metric
in this project because RAG performance depends on surfacing at least one relevant
document quickly. In this project: MRR is computed in `src/evaluator.py` over the 20-
query evaluation set.

---

**MS MARCO** — (Microsoft MAchine Reading COmprehension) A large-scale dataset of
real Bing search queries paired with passages from web documents, used for training
and evaluating information retrieval models. MS MARCO is the dominant training set
for cross-encoder rerankers. In this project: `cross-encoder/ms-marco-MiniLM-L-6-v2`
is a reranker trained on MS MARCO, used to rerank the top-20 RRF candidates.

---

**Node** — In LangGraph, a node is a Python function (or callable) that takes the
current graph state as input, performs some work, and returns a (partial) state update.
Nodes are the "units of computation" in the graph — each does one conceptually distinct
thing: retrieve, grade, search, generate, or check. In this project: the CRAG graph
has five nodes: `retrieve`, `grade_documents`, `web_search`, `generate`, and
`grade_generation`.

---

**Ollama** — A tool for running large language models locally on consumer hardware.
Ollama handles model downloading, quantisation, and serving behind a REST API that
mirrors the OpenAI SDK interface. It supports dozens of open-weight models including
Llama, Mistral, Qwen, Granite, and specialised embedding models. In this project:
Ollama serves both the `qwen3-embedding` embedding model and the `granite4.1:8b`
generation/judge model. All inference is local — no cloud API keys required.

---

**Pinecone** — A fully managed cloud vector database service. Pinecone handles
indexing, storage, and similarity search at scale without infrastructure management.
It supports metadata filtering, namespaces, and multiple index types (serverless and
pod-based). In this project: used as an optional backend in Part 4 GraphRAG.
The core baseline remains runnable locally without Pinecone.

---

**Precision@k** — The fraction of the top-k retrieved documents that are relevant:
`Precision@k = (relevant documents in top k) / k`. Unlike Recall@k, Precision@k
measures *efficiency* — how much noise is in your results. A system with Precision@5
= 0.40 means on average 2 of the top 5 results are relevant. In this project:
Precision@5 is reported alongside Recall@5 in the full benchmark table, with the
best score of 0.19 (BM25) on the 20-query honest eval.

---

**qwen3-embedding** — Alibaba's Qwen3 embedding model, available in various sizes.
It produces 1024-dimensional L2-normalised vectors and achieves strong results on the
MTEB benchmark. The model is designed for retrieval tasks and multilingual text. In
this project: `qwen3-embedding` (the base variant) runs via Ollama and embeds all 600
paper chunks during the indexing step in `01_naive_rag.ipynb`. The same model embeds
every query at retrieval time.

---

**RAGAS** — A framework for evaluating RAG pipelines using automatic metrics including
faithfulness, answer relevance, context precision, and context recall. RAGAS uses
LLM-as-judge internally to compute each metric. In this project: not used directly,
but listed in Further Reading as the recommended next step for a more rigorous
automated eval.

---

**RAG** — (Retrieval-Augmented Generation) An architecture that enhances LLM-generated
responses by first retrieving relevant documents from an external knowledge base and
providing them as context in the prompt. RAG addresses two fundamental LLM limitations:
hallucination (the model invents facts it does not know) and knowledge cutoff (the model
was trained on data up to a fixed date). In this project: RAG is the central subject
of the entire tutorial, progressing from a naive one-stage pipeline (Part 1) through
advanced retrieval techniques (Part 2) to an agentic self-correcting loop (Part 3).

---

**Recall@k** — The fraction of all relevant documents that appear in the top-k results:
`Recall@k = (relevant documents in top k) / (total relevant documents)`. Recall@k
measures *coverage* — does the system find something useful within k results?
A system with Recall@5 = 0.37 surfaces 37% of the relevant documents in the top 5
positions. In this project: Recall@5 is computed in `src/evaluator.py`. The best
result on the honest 20-query eval is 0.3667 from BM25.

---

**Reranking** — A two-stage retrieval pattern where a fast first-stage retriever
(dense or sparse) produces a large candidate set (e.g., top 20), and a slower but
more accurate second-stage model (cross-encoder) rescores and reorders the candidates
to produce the final top-k. Reranking decouples scale (handled by first-stage) from
accuracy (handled by second-stage). In this project: `cross-encoder/ms-marco-MiniLM-L-6-v2`
reranks the top-20 results from RRF fusion, improving MRR from 0.2867 to 0.3625 (+26%).

---

**RRF** — (Reciprocal Rank Fusion) A rank fusion algorithm that combines multiple
ranked result lists without requiring score normalisation.
`RRF_score(d) = Σ 1/(k + rank_r(d))` where the sum is over all ranking systems and
`k=60` is a stability constant. Documents that appear high in multiple lists get the
highest combined scores. RRF is parameter-free, robust to score distribution shifts,
and scales easily to more than two retrieval systems. In this project: `rrf_retrieve()`
in `src/retriever.py` fuses FAISS dense rankings and BM25 rankings before the optional
reranking step.

---

**Self-querying** — A RAG technique where the LLM rewrites or decomposes the user's
question before retrieval — for example, extracting structured metadata filters or
breaking a complex multi-hop question into sub-queries. In this project: not implemented,
but listed in Further Reading as a recommended extension.

---

**Sparse retrieval** — A retrieval paradigm that represents documents as sparse
high-dimensional vectors where each dimension corresponds to a vocabulary term and
the value is a TF-IDF or BM25 weight. Most dimensions are zero (hence "sparse").
Sparse retrieval performs exact vocabulary matching and is highly effective for
keyword-heavy queries. Its weakness is vocabulary mismatch: if the query uses a synonym
not present in the document, the document is not found. In this project: BM25 is the
sparse retrieval component, implemented via `rank_bm25`.

---

**State machine** — A computational model consisting of a finite set of states, a set
of transitions between states triggered by events or conditions, and an initial state.
At any point in time the machine is in exactly one state. Agentic LLM pipelines are
naturally modelled as state machines because the agent must decide at each step which
action to take based on current observations. In this project: the CRAG pipeline is
a state machine where the state includes the query, retrieved documents, grades, web
results, and the generated answer. LangGraph is the framework used to define and
execute this state machine.

---

**TF-IDF** — (Term Frequency — Inverse Document Frequency) A classical text-weighting
scheme that scores how important a term is to a document within a corpus.
TF (term frequency) measures how often the term appears in the document. IDF measures
how rare the term is across the corpus. Their product gives terms that are both frequent
in the document and rare overall — the most discriminative terms. BM25 is a more
sophisticated probabilistic extension of TF-IDF with better handling of document length
and term saturation. In this project: BM25 (via `rank_bm25`) uses TF-IDF-style weighting
internally; TF-IDF itself is not used directly.

---

**uv** — A fast Python package manager and project management tool written in Rust.
`uv` replaces `pip`, `venv`, `pip-tools`, and `pyenv` with a unified, significantly
faster interface. It resolves dependencies in milliseconds and creates reproducible
lockfiles. In this project: the entire project uses `uv` for environment setup and
dependency management. The `requirements.txt` lists top-level dependencies; `uv pip install -r requirements.txt` installs them into a `.venv` created with `uv venv`.

---

**Vector database** — A database specialised for storing, indexing, and querying
embedding vectors. Unlike relational databases (optimised for exact matches and joins),
vector databases are optimised for approximate nearest-neighbour (ANN) search in high-
dimensional spaces. They typically support metadata filtering alongside vector search.
Examples include FAISS (library, not a server), ChromaDB, Pinecone, Weaviate, and Qdrant.
In this project: FAISS (`IndexFlatIP`) is used as a vector index (exact search, no server).
