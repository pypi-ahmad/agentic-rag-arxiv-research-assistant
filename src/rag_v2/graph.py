"""GraphRAG utilities with lightweight entity extraction."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

import networkx as nx


DEFAULT_TERMS = [
    "transformer",
    "attention",
    "diffusion",
    "retrieval",
    "rag",
    "hallucination",
    "benchmark",
    "alignment",
    "instruction tuning",
    "rlhf",
    "lora",
    "knowledge distillation",
    "speculative decoding",
    "mixture of experts",
    "multimodal",
    "vision-language",
    "reasoning",
    "embedding",
    "fine-tuning",
    "agent",
]


def _extract_entities(text: str, max_entities: int = 8) -> list[str]:
    """
    Lightweight entity extraction for large corpora.

    This intentionally avoids per-paper LLM calls so full 4k-paper runs stay fast.
    """
    lowered = text.lower()
    found = [term for term in DEFAULT_TERMS if term in lowered]

    # Add uppercase acronym candidates (LLM, RAG, MoE, etc.)
    acronyms = re.findall(r"\b[A-Z]{2,8}\b", text)
    for token in acronyms:
        if token.lower() not in found:
            found.append(token.lower())

    # Add hyphenated technical phrases
    hyphenated = re.findall(r"\b[a-z]+-[a-z]+\b", lowered)
    for token in hyphenated:
        if token not in found and len(token) > 5:
            found.append(token)

    deduped = []
    seen = set()
    for term in found:
        if term in seen:
            continue
        seen.add(term)
        deduped.append(term)
    return deduped[:max_entities]


def build_paper_entity_graph(papers: list[dict], max_entities_per_paper: int = 8) -> tuple[nx.Graph, dict[str, list[str]]]:
    graph = nx.Graph()
    paper_to_entities: dict[str, list[str]] = {}

    for paper in papers:
        paper_id = paper["id"]
        graph.add_node(paper_id, node_type="paper", title=paper.get("title", ""))
        entities = _extract_entities(f"{paper.get('title', '')}\n{paper.get('text', '')}", max_entities=max_entities_per_paper)
        paper_to_entities[paper_id] = entities

        for ent in entities:
            entity_id = f"entity::{ent}"
            if entity_id not in graph:
                graph.add_node(entity_id, node_type="entity", name=ent, paper_count=0)
            graph.nodes[entity_id]["paper_count"] += 1
            graph.add_edge(paper_id, entity_id, relation="mentions", weight=1)

    # Build entity co-occurrence edges
    for entities in paper_to_entities.values():
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                a = f"entity::{entities[i]}"
                b = f"entity::{entities[j]}"
                if graph.has_edge(a, b):
                    graph[a][b]["weight"] += 1
                else:
                    graph.add_edge(a, b, relation="co_occurs", weight=1)

    return graph, paper_to_entities


def detect_entity_communities(graph: nx.Graph) -> list[frozenset[str]]:
    entity_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "entity"]
    if not entity_nodes:
        return []
    sub = graph.subgraph(entity_nodes).copy()
    if sub.number_of_nodes() == 0:
        return []
    if not nx.is_connected(sub):
        largest = max(nx.connected_components(sub), key=len)
        sub = sub.subgraph(largest).copy()
    communities = list(nx.community.greedy_modularity_communities(sub, weight="weight"))
    communities.sort(key=len, reverse=True)
    return communities


def build_community_summaries(graph: nx.Graph, communities: list[frozenset[str]], top_n: int = 20) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for idx, comm in enumerate(communities[:top_n]):
        names = [graph.nodes[n]["name"] for n in comm if n in graph.nodes]
        names_sorted = sorted(names, key=lambda x: x.lower())
        summary = (
            "This community clusters research around "
            + ", ".join(names_sorted[:6])
            + ". It represents a recurring topic family in the corpus and is used for global synthesis in GraphRAG."
        )
        out[idx] = {"size": len(comm), "entities": names_sorted, "summary": summary}
    return out


def expand_with_graph(
    seed_paper_ids: list[str],
    paper_to_entities: dict[str, list[str]],
    entity_to_papers: dict[str, list[str]],
    max_extra_papers: int = 12,
) -> list[str]:
    counts: Counter[str] = Counter()
    for pid in seed_paper_ids:
        for entity in paper_to_entities.get(pid, []):
            for linked in entity_to_papers.get(entity, []):
                if linked != pid:
                    counts[linked] += 1
    ranked = [pid for pid, _ in counts.most_common(max_extra_papers)]
    return ranked


def build_entity_to_papers(paper_to_entities: dict[str, list[str]]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = defaultdict(list)
    for pid, entities in paper_to_entities.items():
        for ent in entities:
            mapping[ent].append(pid)
    return dict(mapping)

