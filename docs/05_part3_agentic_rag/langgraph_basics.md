# LangGraph Basics

Before building the RAG agent, we need to understand the framework it runs on.
LangGraph is a library for building stateful, multi-step applications where a language
model controls the execution flow. This page explains the core abstractions with simple
examples, then shows how they map to our RAG code.

---

## What LangGraph is

LangGraph models your application as a **directed graph**. Each node in the graph is a
Python function that reads a shared state, does some work, and returns updates to that
state. Edges connect nodes, and conditional edges let the runtime choose which node to
visit next based on the current state.

This sounds abstract. Here is the simplest possible graph — one that just echoes a
message:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class SimpleState(TypedDict):
    message: str
    processed: bool

def process(state: SimpleState) -> dict:
    return {"processed": True}

builder = StateGraph(SimpleState)
builder.add_node("process", process)
builder.set_entry_point("process")
builder.add_edge("process", END)

graph = builder.compile()
result = graph.invoke({"message": "hello", "processed": False})
# result = {"message": "hello", "processed": True}
```

Four steps every graph needs:

1. **Define state** — a `TypedDict` that all nodes share.
2. **Add nodes** — Python functions that accept and return state dictionaries.
3. **Set entry point** — which node runs first.
4. **Add edges** — how control flows between nodes.

---

## StateGraph vs a regular pipeline

A regular pipeline is a list of function calls in a fixed order:

```python
docs = retrieve(question)
ranked = rerank(docs)
answer = generate(question, ranked)
```

A `StateGraph` stores all intermediate values in a shared state object and routes
between nodes dynamically. The same question might take a different path through the
graph depending on what `grade_documents` returns.

The key difference: **the path through the graph is not fixed at write time**.

---

## TypedDict state

Every LangGraph graph is parameterised by a state type. Use a `TypedDict` (not a
dataclass or Pydantic model) — LangGraph merges partial dict updates from each node
into the shared state automatically.

```python
from typing import TypedDict, Optional

class GraphState(TypedDict):
    question: str            # never changes after initialisation
    documents: list[dict]    # set by retrieve
    answer: str              # set by generate
    grade: str               # set by grade — "relevant" or "irrelevant"
```

Each node receives the full current state and returns only the keys it wants to update:

```python
def retrieve(state: GraphState) -> dict:
    docs = retriever.search(state["question"])
    return {"documents": docs}   # only updates "documents"
```

LangGraph merges this return value into the existing state. Keys not returned are
unchanged. This makes nodes composable — each one only knows about the fields it needs.

---

## add_node, add_edge, add_conditional_edges

**`add_node(name, function)`** registers a node. The function signature must be
`(state: YourState) -> dict`.

**`add_edge(from_node, to_node)`** adds a fixed edge — after `from_node` always run
`to_node`. Use `END` as the target to terminate the graph.

**`add_conditional_edges(from_node, routing_function, edge_map)`** is the key to
agentic behaviour. The routing function receives the current state and returns a
string key. The edge map translates that key to the next node:

```python
def route_after_grading(state: GraphState) -> str:
    return state["grade"]   # "relevant" or "irrelevant"

builder.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {
        "relevant": "generate_answer",
        "irrelevant": "web_search",
    }
)
```

After `grade_documents` runs, LangGraph calls `route_after_grading`, reads the returned
string, looks it up in the edge map, and runs the corresponding node next.

---

## compile() and invoke()

`builder.compile()` validates the graph, checks for unreachable nodes and missing
edges, and returns a compiled `CompiledGraph` object.

`graph.invoke(initial_state)` runs the graph synchronously from the entry point to
`END`, returning the final state dictionary.

```python
rag_agent = graph_builder.compile()

result = rag_agent.invoke({
    "question": "What is RLHF?",
    "documents": [],
    "filtered_documents": [],
    "retrieval_grade": "",
    "generation": "",
    "generation_attempts": 0,
    "faithfulness_grade": "",
    "execution_trace": [],
})

print(result["generation"])
```

!!! warning "The .graph attribute gotcha"
    In LangGraph 0.4.x the compiled graph does **not** have a `.graph` attribute.
    If you try `rag_agent.graph.nodes.keys()` you will get an `AttributeError`.
    To inspect the graph structure, use `graph_builder.nodes.keys()` on the
    **builder** object before compiling, not on the compiled result.

    ```python
    # Correct — inspect the builder
    print(list(graph_builder.nodes.keys()))
    # ['retrieve', 'grade_documents', 'web_search', 'generate_answer', 'grade_hallucination']

    # Wrong — compiled graph has no .graph attribute in 0.4.x
    # rag_agent.graph.nodes.keys()  # AttributeError
    ```

---

## Putting it together

Here is the complete graph construction for our RAG agent, without the node
implementations (those are covered in subsequent pages):

```python
from langgraph.graph import StateGraph, END

graph_builder = StateGraph(GraphState)

# Register the five nodes
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("grade_documents", grade_documents)
graph_builder.add_node("web_search", web_search)
graph_builder.add_node("generate_answer", generate_answer)
graph_builder.add_node("grade_hallucination", grade_hallucination)

# Fixed edges
graph_builder.set_entry_point("retrieve")
graph_builder.add_edge("retrieve", "grade_documents")
graph_builder.add_edge("web_search", "generate_answer")
graph_builder.add_edge("generate_answer", "grade_hallucination")

# Conditional edges
graph_builder.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {"relevant": "generate_answer", "irrelevant": "web_search"}
)
graph_builder.add_conditional_edges(
    "grade_hallucination",
    route_after_faithfulness,
    {"faithful": END, "hallucinated": "generate_answer"}
)

rag_agent = graph_builder.compile()
```

Twenty-five lines to define a self-correcting, web-augmented research agent.
The next page explains the state type and the architectural paper behind it.
