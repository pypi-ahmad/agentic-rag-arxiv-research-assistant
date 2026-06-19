# Troubleshooting

Exact errors encountered during development, with their root cause and fix.
Errors are ordered from most to least common.

---

## `ConnectionError` — Ollama not running

**Exact error:**

```
httpx.ConnectError: [Errno 111] Connection refused
```

or

```
ollama._types.ResponseError: model not found, try pulling it first
```

or from the `ollama` Python library:

```
ConnectionError: HTTPConnectionPool(host='localhost', port=11434): Max retries exceeded
```

**Cause:** The Ollama daemon is not running. The `ollama` Python library connects to
`localhost:11434` by default. Every embedding call and every LLM call will fail if
Ollama is not up.

**Fix:**

```bash
# Start Ollama in the background
ollama serve &

# Verify it is running
curl http://localhost:11434/api/tags

# Check that the required models are pulled
ollama list
# Should show: qwen3-embedding and granite4.1:8b
```

If the models are not listed:

```bash
ollama pull qwen3-embedding
ollama pull granite4.1:8b
```

---

## `ollama.ResponseError` — Model not found

**Exact error:**

```
ollama._types.ResponseError: model 'granite4.1:8b' not found, try pulling it first
```

**Cause:** The model name in the notebook does not match the model name as stored in
Ollama. This happens after a fresh install, after manually pulling a differently-tagged
version, or after Ollama updates change model naming conventions.

**Fix:**

```bash
# See exact model names as Ollama knows them
ollama list

# Pull the correct model (check the exact tag)
ollama pull granite4.1:8b
ollama pull qwen3-embedding
```

Then in the notebook, ensure the model name string matches exactly:

```python
# Correct
response = ollama.chat(model="granite4.1:8b", messages=[...])

# Wrong — tag mismatch
response = ollama.chat(model="granite4.1", messages=[...])        # missing :8b
response = ollama.chat(model="granite4.1:latest", messages=[...]) # wrong tag
```

---

## `FileNotFoundError` — FAISS index not found

**Exact error:**

```
FileNotFoundError: [Errno 2] No such file or directory: 'artifacts/faiss_index.bin'
```

or

```
FileNotFoundError: [Errno 2] No such file or directory: 'artifacts/chunks.pkl'
```

**Cause:** Parts 2 and 3 of the tutorial load the FAISS index and chunk list that are
saved at the end of Part 1 (`01_naive_rag.ipynb`). If you open a Part 2 or Part 3
notebook without having run Part 1 first, the `artifacts/` directory will be empty.

**Fix:** Run `01_naive_rag.ipynb` from top to bottom before opening any later notebook.
The final cells save:

- `artifacts/faiss_index.bin` — the FAISS index
- `artifacts/chunks.pkl` — the list of chunk strings
- `artifacts/metadata.pkl` — the list of metadata dicts

You only need to run Part 1 once. After that, Parts 2 and 3 can be run independently
as many times as needed.

```bash
# Verify the artifacts exist
ls -lh artifacts/
# Should show: faiss_index.bin, chunks.pkl, metadata.pkl
```

---

## `json.JSONDecodeError` — grade_documents parsing failure

**Exact error:**

```python
json.JSONDecodeError: Expecting value: line 1 column 1 (char 0)
```

or

```python
json.JSONDecodeError: Extra data: line 2 column 1 (char 47)
```

**Cause:** The LLM judge (granite4.1:8b) returned text that is not valid JSON. This
happens because:

1. The model prefixed the JSON with a reasoning preamble ("Let me think about this...")
2. The model returned markdown-fenced JSON (` ```json\n{...}\n``` `)
3. The model returned multiple JSON objects or trailing text after the JSON
4. The model returned `None` or an empty string on a very short document

**Fix:** Extract JSON from the response text rather than parsing it directly.

```python
import json
import re

def parse_grade(response_text: str) -> str:
    """Extract relevance grade from LLM response, robust to extra text."""
    # Try direct parse first
    try:
        data = json.loads(response_text.strip())
        return data.get("relevance", "bad").lower()
    except json.JSONDecodeError:
        pass

    # Try to extract JSON object from text
    match = re.search(r'\{[^}]+\}', response_text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return data.get("relevance", "bad").lower()
        except json.JSONDecodeError:
            pass

    # Fallback: scan for the keyword directly
    text_lower = response_text.lower()
    if '"good"' in text_lower or "'good'" in text_lower or ": good" in text_lower:
        return "good"
    return "bad"
```

Also: add explicit instructions in the prompt to not include any text before or after
the JSON:

```python
GRADE_PROMPT = """...
Respond ONLY with valid JSON on a single line. No explanation. No markdown.
Example: {"relevance": "good"}
"""
```

---

## FAISS `idx = -1` sentinel — missing or corrupted results

**Symptom:** Retrieval returns documents with `idx=-1`, or slicing the chunk list
raises `IndexError: list index out of range`.

**Exact error:**

```
IndexError: list index out of range
# or the returned chunk is from index -1, which is the last element
```

**Cause:** FAISS `IndexFlatIP.search()` returns `-1` as a sentinel value for any
result slot it could not fill — this happens when `k` (number of requested results)
is larger than the number of vectors in the index. If you ask for `k=5` results from
an index with only 3 vectors, the last two slots contain `idx=-1` and `distance=-inf`.

```python
# Naive code that breaks:
distances, indices = index.search(query_vector, k=5)
results = [chunks[i] for i in indices[0]]  # crashes if i == -1
```

**Fix:** Filter out the `-1` sentinel before accessing the chunk list:

```python
distances, indices = index.search(query_vector, k=5)
results = [chunks[i] for i in indices[0] if i != -1]
```

This is also the correct defensive pattern for production: never assume the index
has at least `k` entries.

---

## `RatelimitException` — DuckDuckGo web search throttled

**Exact error:**

```
duckduckgo_search.exceptions.RatelimitException: https://duckduckgo.com 202 Ratelimit
```

**Cause:** DuckDuckGo rate-limits automated search requests. The `duckduckgo_search`
library wraps the public (unofficial) DuckDuckGo API. Sending more than a few queries
per minute in quick succession triggers the rate limit. This most commonly occurs when
running the CRAG agent on multiple queries in a loop without any delay.

**Fix:** Add a retry with exponential backoff, or a simple sleep between web search
calls:

```python
import time
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException

def web_search_with_retry(query: str, max_results: int = 3, retries: int = 3) -> list[str]:
    """Search DuckDuckGo with retry on rate limit."""
    for attempt in range(retries):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            return [r["body"] for r in results if r.get("body")]
        except RatelimitException:
            wait = 2 ** attempt  # 1s, 2s, 4s
            print(f"Rate limited. Waiting {wait}s before retry {attempt + 1}/{retries}...")
            time.sleep(wait)
    return []  # return empty list after all retries fail
```

In interactive notebook use, the simplest fix is to add `time.sleep(2)` before the
DuckDuckGo call. The rate limit is per-IP and resets quickly.

---

## `AttributeError` — `CompiledStateGraph` has no attribute `'graph'`

**Exact error:**

```
AttributeError: 'CompiledStateGraph' object has no attribute 'graph'
```

**Cause:** LangGraph 0.4+ changed the public API for inspecting the compiled graph.
Code written for LangGraph 0.3.x that accesses `compiled_graph.graph` to list nodes
or visualise the graph will raise this error on 0.4+.

**Affected pattern:**

```python
# Old code — LangGraph ≤ 0.3
app = workflow.compile()
print(app.graph.nodes.keys())          # AttributeError on 0.4+
app.graph.draw_mermaid_png(...)        # AttributeError on 0.4+
```

**Fix:** Use `graph_builder.nodes.keys()` (on the `StateGraph` before compilation) to
list nodes, and use the compiled graph's own visualisation methods:

```python
# List nodes — access the builder, not the compiled graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
# ...

print(workflow.nodes.keys())           # works on the builder

# Compile
app = workflow.compile()

# Visualise — use the compiled graph's method directly
png_bytes = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_bytes)
```

To check your LangGraph version:

```bash
python -c "import langgraph; print(langgraph.__version__)"
```

If you need to support both old and new versions:

```python
try:
    nodes = list(app.graph.nodes.keys())       # 0.3.x
except AttributeError:
    nodes = list(workflow.nodes.keys())        # 0.4+
```

---

## `arxiv.HTTPError` 503 / 429 — ArXiv API unavailable or rate-limited

**Exact error:**

```
arxiv.arxiv.HTTPError: Page request resulted in HTTP 503: Service Unavailable
```

or

```
arxiv.arxiv.HTTPError: Page request resulted in HTTP 429: Too Many Requests
```

**Cause:** The ArXiv API is occasionally unavailable (503) or is being rate-limited
(429) when too many requests are sent in a short period. This typically happens during
the data collection step in `01_naive_rag.ipynb` when fetching 600 papers across
multiple keyword queries.

**Fix:** The project includes a HuggingFace fallback. If the ArXiv API fails, the
ingest script downloads a pre-built subset of ArXiv ML abstracts from the HuggingFace
Hub (the `ccdv/arxiv-summarization` or similar dataset):

```python
import arxiv
import time

def fetch_arxiv_with_fallback(queries: list[str], max_per_query: int = 60) -> list[dict]:
    papers = []
    for query in queries:
        try:
            search = arxiv.Search(query=query, max_results=max_per_query)
            for result in search.results():
                papers.append({
                    "title": result.title,
                    "abstract": result.summary,
                    "arxiv_id": result.entry_id,
                })
            time.sleep(3)  # be polite to the API
        except arxiv.HTTPError as e:
            print(f"ArXiv API error for '{query}': {e}. Using HuggingFace fallback.")
            papers.extend(_load_huggingface_fallback())
            break
    return papers

def _load_huggingface_fallback() -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("ccdv/arxiv-summarization", split="train", streaming=True)
    return [
        {"title": row["section_names"], "abstract": row["abstract"], "arxiv_id": "hf-fallback"}
        for _, row in zip(range(600), ds)
    ]
```

For 503 errors specifically, adding `time.sleep(10)` between query batches is usually
sufficient. The ArXiv API allows ~3 requests per second but returns 503 under load.

---

## General debugging tips

**Check Ollama model names exactly:**

```bash
ollama list
# Copy-paste the model name from this output — do not retype it
```

**Verify FAISS index is not empty:**

```python
import faiss, pickle
index = faiss.read_index("artifacts/faiss_index.bin")
print(f"Index has {index.ntotal} vectors")  # should be 600
with open("artifacts/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)
print(f"Chunks list has {len(chunks)} entries")  # should match index.ntotal
```

**Test Ollama embedding directly:**

```python
import ollama
response = ollama.embeddings(model="qwen3-embedding", prompt="test query")
print(len(response["embedding"]))  # should print 1024
```

**Test Ollama generation directly:**

```python
import ollama
response = ollama.chat(
    model="granite4.1:8b",
    messages=[{"role": "user", "content": "Say 'hello'."}]
)
print(response["message"]["content"])
```
