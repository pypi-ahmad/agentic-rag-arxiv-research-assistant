"""Agentic and CRAG-style orchestration helpers for notebooks 07-08."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import ollama


def _safe_json(text: str, default: dict) -> dict:
    try:
        return json.loads(text)
    except Exception:
        return default


def llm_relevance_grade(
    question: str,
    contexts: list[str],
    judge_model: str = "granite4.1:8b",
) -> tuple[str, float]:
    joined = "\n\n".join(f"[{i+1}] {c[:500]}" for i, c in enumerate(contexts[:3]))
    prompt = f"""You are grading retrieval quality.

Question:
{question}

Retrieved contexts:
{joined}

Respond with strict JSON:
{{"grade":"relevant|partially_relevant|irrelevant","confidence":0.0-1.0}}"""

    resp = ollama.chat(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0, "num_gpu": 0, "num_ctx": 3072, "num_predict": 96},
    )
    data = _safe_json(resp["message"]["content"], {"grade": "irrelevant", "confidence": 0.0})
    grade = str(data.get("grade", "irrelevant"))
    conf = float(data.get("confidence", 0.0))
    return grade, max(0.0, min(1.0, conf))


def llm_answer(
    question: str,
    contexts: list[str],
    model: str = "granite4.1:8b",
    temperature: float = 0.2,
) -> str:
    context_block = "\n\n".join(f"[{i+1}] {c[:700]}" for i, c in enumerate(contexts[:3]))
    prompt = f"""Answer the question using ONLY the provided context.
If the context is insufficient, say that explicitly.

Context:
{context_block}

Question: {question}

Answer:"""
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature, "num_gpu": 0, "num_ctx": 3072, "num_predict": 220},
    )
    return resp["message"]["content"].strip()


def llm_faithfulness(
    question: str,
    answer: str,
    contexts: list[str],
    judge_model: str = "granite4.1:8b",
) -> tuple[float, str]:
    joined = "\n\n".join(f"[{i+1}] {c[:500]}" for i, c in enumerate(contexts[:3]))
    prompt = f"""You are a strict faithfulness grader.
Check if the answer is supported by the provided context.

Question: {question}
Answer: {answer}

Context:
{joined}

Return JSON only:
{{"faithful":true|false,"score":0.0-1.0,"reason":"one short sentence"}}"""
    resp = ollama.chat(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        format="json",
        options={"temperature": 0, "num_gpu": 0, "num_ctx": 3072, "num_predict": 110},
    )
    data = _safe_json(resp["message"]["content"], {"faithful": False, "score": 0.0, "reason": "parse_failure"})
    score = float(data.get("score", 0.0))
    reason = str(data.get("reason", ""))
    if bool(data.get("faithful", False)):
        score = max(score, 0.75)
    return max(0.0, min(1.0, score)), reason


def rewrite_query(question: str, model: str = "granite4.1:8b") -> str:
    prompt = f"""Rewrite this query to improve document retrieval while preserving intent.
Return only the rewritten query.

Query: {question}"""
    resp = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_gpu": 0, "num_ctx": 2048, "num_predict": 48},
    )
    return resp["message"]["content"].strip().replace("\n", " ")


@dataclass
class AgenticResult:
    question: str
    answer: str
    route: str
    retrieval_grade: str
    retrieval_confidence: float
    faithfulness: float
    faithfulness_reason: str
    latency_ms: float
    attempts: int
    trace: list[str]


def route_query(question: str) -> str:
    text = question.lower()
    if any(k in text for k in ["compare", "survey", "landscape", "trends"]):
        return "graph"
    if any(k in text for k in ["exact", "term", "keyword", "acronym"]):
        return "bm25"
    return "hybrid"
