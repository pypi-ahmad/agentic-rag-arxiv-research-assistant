"""Evaluation helpers for new RAG notebooks."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def recall_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / len(set(relevant))


def precision_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> float:
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & set(relevant)) / k


def mrr(retrieved: list[str], relevant: list[str]) -> float:
    rel = set(relevant)
    for i, rid in enumerate(retrieved, start=1):
        if rid in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> float:
    rel = set(relevant)
    dcg = sum(1.0 / np.log2(i + 2) for i, rid in enumerate(retrieved[:k]) if rid in rel)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(rel))))
    return float(dcg / ideal) if ideal > 0 else 0.0


def f1_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> float:
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def build_keyword_eval_set(papers: list[dict]) -> list[dict]:
    """
    Build a weakly supervised eval set.

    Ground truth is derived from keyword-match anchors over paper text.
    """
    tasks = [
        ("How does RLHF work?", ["rlhf", "reward model", "human feedback"]),
        ("What is LoRA and why is it useful?", ["lora", "low-rank adaptation", "parameter-efficient"]),
        ("How does retrieval-augmented generation work?", ["retrieval-augmented", "rag", "retrieval generation"]),
        ("What are mixture-of-experts models?", ["mixture of experts", "moe", "expert routing"]),
        ("What is flash attention?", ["flash attention", "io-aware attention"]),
        ("How are diffusion models trained?", ["diffusion model", "denoising diffusion", "ddpm"]),
        ("How does chain-of-thought prompting help?", ["chain-of-thought", "step-by-step reasoning"]),
        ("How does knowledge distillation compress models?", ["knowledge distillation", "teacher student"]),
        ("How do vision-language models learn?", ["vision-language", "multimodal model", "vlm"]),
        ("How is hallucination evaluated in LLMs?", ["hallucination", "factuality", "faithfulness"]),
        ("How does instruction tuning work?", ["instruction tuning", "instruction following", "flan"]),
        ("How does speculative decoding speed inference?", ["speculative decoding", "draft model"]),
    ]

    eval_set: list[dict] = []
    for question, keywords in tasks:
        matched: list[str] = []
        for paper in papers:
            text = f"{paper.get('title', '')} {paper.get('text', '')}".lower()
            if any(k in text for k in keywords):
                matched.append(paper["id"])
            if len(matched) >= 5:
                break
        eval_set.append({"question": question, "relevant_ids": matched})
    return eval_set


def run_retrieval_eval(eval_set: list[dict], retriever, k: int = 5) -> dict:
    recalls: list[float] = []
    precisions: list[float] = []
    mrrs: list[float] = []
    f1s: list[float] = []
    ndcgs: list[float] = []
    latencies_ms: list[float] = []

    for row in eval_set:
        t0 = time.perf_counter()
        results = retriever.retrieve(row["question"], k=max(k, 8))
        dt = (time.perf_counter() - t0) * 1000
        latencies_ms.append(dt)

        retrieved_papers = [r.get("paper_id", "") for r in results if r.get("paper_id")]
        rel = row["relevant_ids"]
        recalls.append(recall_at_k(retrieved_papers, rel, k=k))
        precisions.append(precision_at_k(retrieved_papers, rel, k=k))
        mrrs.append(mrr(retrieved_papers, rel))
        f1s.append(f1_at_k(retrieved_papers, rel, k=k))
        ndcgs.append(ndcg_at_k(retrieved_papers, rel, k=k))

    return {
        f"recall@{k}": round(float(np.mean(recalls)), 4),
        f"precision@{k}": round(float(np.mean(precisions)), 4),
        "mrr": round(float(np.mean(mrrs)), 4),
        f"f1@{k}": round(float(np.mean(f1s)), 4),
        f"ndcg@{k}": round(float(np.mean(ndcgs)), 4),
        "latency_p50_ms": round(float(np.percentile(latencies_ms, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "n_queries": len(eval_set),
    }


def compute_retrieval_metrics(eval_set: list[dict], retrievers: dict[str, object], k: int = 5) -> dict[str, dict]:
    return {name: run_retrieval_eval(eval_set, r, k=k) for name, r in retrievers.items()}


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def plot_retrieval_comparison(results: dict[str, dict], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(results.keys())
    recalls = [results[n].get("recall@5", 0.0) for n in names]
    precisions = [results[n].get("precision@5", 0.0) for n in names]
    mrrs = [results[n].get("mrr", 0.0) for n in names]

    x = np.arange(len(names))
    w = 0.25

    plt.figure(figsize=(11, 5))
    plt.bar(x - w, recalls, width=w, label="Recall@5")
    plt.bar(x, precisions, width=w, label="Precision@5")
    plt.bar(x + w, mrrs, width=w, label="MRR")
    plt.xticks(x, names, rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Retrieval Comparison Across RAG Variants")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


@dataclass
class GenerationEvalRow:
    question: str
    answer: str
    faithfulness: float
    relevance: float
    latency_ms: float

