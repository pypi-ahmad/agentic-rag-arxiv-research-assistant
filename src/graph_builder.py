"""
src/graph_builder.py — Knowledge graph construction for Graph RAG

Pipeline:
  papers
    → extract_all_entities()    LLM extracts method/concept/model entities per paper
    → build_knowledge_graph()   NetworkX graph: paper nodes + entity nodes + edges
    → detect_communities()      Greedy modularity clustering → topic clusters
    → summarise_all_communities()  LLM writes a 3-5 sentence summary per cluster

The entity extraction and community summarisation steps call granite4.1:8b
once per paper / once per community. Both functions support progress-save/resume:
results are cached to JSON after every batch so a crash never loses work.

Key design decisions:
  - Entities are deduplicated by lowercased name — "RLHF" and "rlhf" collapse to one node.
  - Co-occurrence edges are weighted by the number of papers where both entities appear.
  - Community detection runs on the entity subgraph (not paper nodes) so the algorithm
    finds concept clusters, not paper similarity clusters.
  - Global search uses community summaries, not individual paper embeddings.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import networkx as nx
import numpy as np
import ollama
from loguru import logger
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────────

ENTITY_EXTRACTION_PROMPT = """\
You are an expert in machine learning and AI research.
Extract the most important technical entities from the paper abstract below.

Entity types to extract:
  - method     : algorithms, techniques, architectures (e.g. "LoRA", "RLHF", "BM25")
  - model      : specific trained models (e.g. "GPT-4", "BERT", "Qwen3")
  - dataset    : datasets or benchmarks (e.g. "SQuAD", "MMLU", "MS MARCO")
  - concept    : core ideas or research themes (e.g. "attention mechanism", "RAG", "fine-tuning")
  - metric     : evaluation measures (e.g. "BLEU", "MRR", "perplexity")

Rules:
  - Extract 3 to 8 entities. Do not over-extract.
  - Use the exact name used in the abstract (preserve capitalisation).
  - Write a one-sentence description of the entity IN THE CONTEXT OF THIS PAPER.
  - Return ONLY valid JSON. No explanation, no markdown fences.

Abstract:
{abstract}

Required JSON format:
{{"entities": [{{"name": "...", "type": "method|model|dataset|concept|metric", "description": "..."}}]}}"""

COMMUNITY_SUMMARY_PROMPT = """\
You are summarising a cluster of related machine learning research topics.

The following technical concepts and methods frequently appear together across many papers,
forming a coherent research community:

{entity_list}

Sample paper titles from this community:
{paper_titles}

Write a 3 to 5 sentence technical summary of what research themes this community represents.
Focus on: what core problem it addresses, what methods or ideas it encompasses, and why
these concepts cluster together. Be specific and technical.
Do not list the entities — synthesise them into a coherent narrative."""

CACHE_SAVE_INTERVAL = 10   # save entity cache every N papers


# ── Entity extraction ─────────────────────────────────────────────────────────

def _extract_entities_single(
    paper: dict,
    model: str = "granite4.1:8b",
) -> list[dict]:
    """
    Call granite4.1:8b to extract entities from one paper abstract.

    Returns a list of entity dicts: [{"name": ..., "type": ..., "description": ...}]
    Returns [] on any failure (logged as WARNING — does not propagate).
    """
    prompt = ENTITY_EXTRACTION_PROMPT.format(abstract=paper["abstract"][:2000])
    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0},
        )
        raw = response["message"]["content"]
        parsed = json.loads(raw)
        entities = parsed.get("entities", [])
        # Validate structure
        valid = [
            e for e in entities
            if isinstance(e, dict) and "name" in e and "type" in e
        ]
        return valid
    except (json.JSONDecodeError, KeyError, Exception) as exc:
        logger.warning(f"Entity extraction failed for {paper['id']}: {exc}")
        return []


def extract_all_entities(
    papers: list[dict],
    model: str = "granite4.1:8b",
    cache_path: Path = Path("artifacts/graph/entities_cache.json"),
) -> dict[str, list[dict]]:
    """
    Extract entities from all papers with progress-save and resume support.

    How resume works:
        Results are written to `cache_path` every CACHE_SAVE_INTERVAL papers.
        On next run, already-processed paper IDs are loaded from the cache and
        skipped — the function only processes papers that are not yet cached.
        A paper that returned [] (LLM failure) is also cached so it is not retried.

    Args:
        papers:     List of paper dicts (must include "id" and "abstract" keys).
        model:      Ollama model to use for extraction.
        cache_path: JSON file for saving / loading intermediate results.

    Returns:
        Dict mapping paper_id → list of entity dicts.
        Keys exist for every input paper; value is [] for failed extractions.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing cache
    if cache_path.exists():
        with open(cache_path) as f:
            cache: dict[str, list[dict]] = json.load(f)
        logger.info(f"Entity cache loaded: {len(cache)}/{len(papers)} papers already done")
    else:
        cache = {}

    to_process = [p for p in papers if p["id"] not in cache]
    if not to_process:
        logger.info("All entities already cached — skipping extraction")
        return cache

    logger.info(f"Extracting entities for {len(to_process)} papers (model: {model})")

    for i, paper in enumerate(tqdm(to_process, desc="Extracting entities")):
        entities = _extract_entities_single(paper, model)
        cache[paper["id"]] = entities

        # Progress save every CACHE_SAVE_INTERVAL papers
        if (i + 1) % CACHE_SAVE_INTERVAL == 0:
            with open(cache_path, "w") as f:
                json.dump(cache, f)

        # Small delay to avoid hammering Ollama
        time.sleep(0.05)

    # Final save
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)

    total_entities = sum(len(v) for v in cache.values())
    logger.info(
        f"Entity extraction complete: {len(cache)} papers, "
        f"{total_entities} total entities"
    )
    return cache


# ── Graph construction ────────────────────────────────────────────────────────

def build_knowledge_graph(
    papers: list[dict],
    entities_cache: dict[str, list[dict]],
) -> nx.Graph:
    """
    Build an undirected knowledge graph from papers and their extracted entities.

    Graph structure:
        Node types:
          - Paper nodes:  id = paper["id"], attrs: type="paper", title, category
          - Entity nodes: id = "entity::{name_lower}", attrs: type="entity",
                          entity_type, name, description, paper_count

        Edge types:
          - Paper → Entity:  weight = 1, relation = "contains"
          - Entity → Entity: weight = co-occurrence count, relation = "co_occurs"

    Why include paper → entity edges?
        Local search uses them to hop from a matched chunk's paper to all its
        entities, then to other papers sharing those entities — this is the
        core "graph expansion" that enriches context beyond pure vector search.

    Why include entity → entity edges?
        Community detection runs on entity nodes only. Co-occurrence edges let
        the algorithm find concept clusters (e.g. "LoRA + QLoRA + fine-tuning"
        cluster together because they appear in the same papers).

    Args:
        papers:         List of paper dicts.
        entities_cache: Output of extract_all_entities().

    Returns:
        nx.Graph with paper nodes, entity nodes, and the two edge types above.
    """
    G = nx.Graph()

    # Build a lookup: paper_id → paper dict
    paper_lookup = {p["id"]: p for p in papers}

    # Add paper nodes
    for paper in papers:
        G.add_node(
            paper["id"],
            node_type="paper",
            title=paper["title"],
            category=paper.get("category", ""),
        )

    # Add entity nodes and edges
    # Track co-occurrence: (entity_a, entity_b) → count
    co_occurrence: dict[tuple[str, str], int] = {}

    for paper_id, entities in entities_cache.items():
        if not entities or paper_id not in paper_lookup:
            continue

        entity_ids_this_paper: list[str] = []

        for entity in entities:
            name = entity.get("name", "").strip()
            if not name:
                continue

            entity_id = f"entity::{name.lower()}"

            # Add or update entity node
            if entity_id not in G:
                G.add_node(
                    entity_id,
                    node_type="entity",
                    entity_type=entity.get("type", "concept"),
                    name=name,
                    description=entity.get("description", ""),
                    paper_count=0,
                )
            G.nodes[entity_id]["paper_count"] = G.nodes[entity_id]["paper_count"] + 1

            # Paper → Entity edge
            G.add_edge(paper_id, entity_id, relation="contains", weight=1)
            entity_ids_this_paper.append(entity_id)

        # Entity → Entity co-occurrence edges (within this paper)
        for i, eid_a in enumerate(entity_ids_this_paper):
            for eid_b in entity_ids_this_paper[i + 1 :]:
                key = (min(eid_a, eid_b), max(eid_a, eid_b))
                co_occurrence[key] = co_occurrence.get(key, 0) + 1

    # Add co-occurrence edges
    for (eid_a, eid_b), count in co_occurrence.items():
        if G.has_edge(eid_a, eid_b):
            G[eid_a][eid_b]["weight"] += count
        else:
            G.add_edge(eid_a, eid_b, relation="co_occurs", weight=count)

    n_papers   = sum(1 for _, d in G.nodes(data=True) if d["node_type"] == "paper")
    n_entities = sum(1 for _, d in G.nodes(data=True) if d["node_type"] == "entity")
    logger.info(
        f"Knowledge graph: {n_papers} paper nodes, {n_entities} entity nodes, "
        f"{G.number_of_edges()} edges"
    )
    return G


# ── Community detection ────────────────────────────────────────────────────────

def detect_communities(G: nx.Graph) -> list[frozenset]:
    """
    Detect communities in the entity subgraph using greedy modularity optimisation.

    Why entity-only subgraph?
        Paper nodes connect to many entities and would dominate the modularity
        calculation, producing paper-centric clusters instead of concept clusters.
        Running community detection on entity nodes only produces topic clusters
        (e.g. "fine-tuning methods", "evaluation benchmarks").

    Returns:
        List of frozensets, each containing entity node IDs belonging to one community.
        Sorted descending by community size.
    """
    entity_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "entity"]
    entity_subgraph = G.subgraph(entity_nodes)

    # Keep only the largest connected component (isolated entities don't cluster)
    if not nx.is_connected(entity_subgraph):
        largest_cc = max(nx.connected_components(entity_subgraph), key=len)
        entity_subgraph = G.subgraph(largest_cc)

    communities = list(nx.community.greedy_modularity_communities(entity_subgraph, weight="weight"))
    communities.sort(key=len, reverse=True)

    logger.info(
        f"Community detection: {len(communities)} communities, "
        f"largest has {len(communities[0]) if communities else 0} entities"
    )
    return communities


# ── Community summarisation ────────────────────────────────────────────────────

def _summarise_community_single(
    community: frozenset,
    G: nx.Graph,
    paper_lookup: dict[str, dict],
    model: str = "granite4.1:8b",
) -> str:
    """Generate a text summary for one community using the LLM."""
    # Collect entity names and types
    entity_lines = []
    for eid in list(community)[:25]:   # cap at 25 to keep prompt manageable
        node = G.nodes[eid]
        entity_lines.append(f"  - {node['name']} ({node['entity_type']}): {node.get('description', '')[:80]}")

    # Collect paper titles that contain entities from this community
    paper_ids_in_community: set[str] = set()
    for eid in community:
        for neighbour in G.neighbors(eid):
            if G.nodes[neighbour]["node_type"] == "paper":
                paper_ids_in_community.add(neighbour)

    paper_titles = [
        paper_lookup[pid]["title"]
        for pid in list(paper_ids_in_community)[:8]
        if pid in paper_lookup
    ]

    prompt = COMMUNITY_SUMMARY_PROMPT.format(
        entity_list="\n".join(entity_lines),
        paper_titles="\n".join(f"  - {t}" for t in paper_titles),
    )

    try:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3},
        )
        return response["message"]["content"].strip()
    except Exception as exc:
        logger.warning(f"Community summary failed: {exc}")
        names = [G.nodes[eid]["name"] for eid in list(community)[:5]]
        return f"Community covering: {', '.join(names)}"


def summarise_all_communities(
    G: nx.Graph,
    communities: list[frozenset],
    papers: list[dict],
    model: str = "granite4.1:8b",
    cache_path: Path = Path("artifacts/graph/community_summaries.json"),
    min_community_size: int = 3,
) -> dict[int, dict]:
    """
    Generate summaries for all communities with progress-save and resume.

    How resume works:
        Community summaries are indexed by community index (0, 1, 2, ...).
        On restart, already-summarised indices are skipped.

    Args:
        G:                    The knowledge graph.
        communities:          Output of detect_communities().
        papers:               List of paper dicts (for title lookup).
        model:                Ollama model for summarisation.
        cache_path:           JSON cache file.
        min_community_size:   Skip communities smaller than this threshold
                              (tiny communities produce uninformative summaries).

    Returns:
        Dict[int, dict] mapping community index → {
            "summary":      str,
            "size":         int,
            "entity_names": list[str],
            "paper_ids":    list[str],
        }
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        with open(cache_path) as f:
            raw = json.load(f)
        cache: dict[int, dict] = {int(k): v for k, v in raw.items()}
        logger.info(f"Community summary cache: {len(cache)} summaries already done")
    else:
        cache = {}

    paper_lookup = {p["id"]: p for p in papers}

    for idx, community in enumerate(tqdm(communities, desc="Summarising communities")):
        if idx in cache:
            continue
        if len(community) < min_community_size:
            continue

        summary = _summarise_community_single(community, G, paper_lookup, model)

        # Collect paper IDs that contribute to this community
        paper_ids: set[str] = set()
        for eid in community:
            for neighbour in G.neighbors(eid):
                if G.nodes[neighbour]["node_type"] == "paper":
                    paper_ids.add(neighbour)

        cache[idx] = {
            "summary":      summary,
            "size":         len(community),
            "entity_names": [G.nodes[eid]["name"] for eid in list(community)[:20]],
            "paper_ids":    list(paper_ids),
        }

        # Save after every community
        with open(cache_path, "w") as f:
            json.dump({str(k): v for k, v in cache.items()}, f, indent=2)

    logger.info(f"Community summarisation complete: {len(cache)} summaries")
    return cache


# ── Graph search utilities ────────────────────────────────────────────────────

def get_papers_for_entities(
    entity_ids: list[str],
    G: nx.Graph,
    max_papers: int = 10,
) -> list[str]:
    """
    Given a list of entity node IDs, return the paper IDs that contain them.

    Used in local search to expand from matched entities back to papers,
    then retrieve those papers' chunks for richer context.
    """
    paper_ids: set[str] = set()
    for eid in entity_ids:
        if eid not in G:
            continue
        for neighbour in G.neighbors(eid):
            if G.nodes[neighbour]["node_type"] == "paper":
                paper_ids.add(neighbour)
            if len(paper_ids) >= max_papers:
                break
    return list(paper_ids)[:max_papers]


def get_entity_ids_for_papers(
    paper_ids: list[str],
    G: nx.Graph,
) -> list[str]:
    """
    Given a list of paper node IDs, return all entity node IDs linked to them.

    Used in local search: after vector retrieval returns top-k paper chunks,
    find the entities those papers mention, then expand to their neighbours.
    """
    entity_ids: set[str] = set()
    for pid in paper_ids:
        if pid not in G:
            continue
        for neighbour in G.neighbors(pid):
            if G.nodes[neighbour]["node_type"] == "entity":
                entity_ids.add(neighbour)
    return list(entity_ids)
