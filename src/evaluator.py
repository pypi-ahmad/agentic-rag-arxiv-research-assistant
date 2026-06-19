"""
src/evaluator.py — RAG evaluation metrics (shared by all three notebooks)

Why evaluate retrieval and generation separately?
    A RAG system has two failure modes:
      1. Retrieval failure: the right documents are not retrieved → the LLM
         cannot produce a correct answer even if it tries.
      2. Generation failure: the right documents are retrieved but the LLM
         ignores them and hallucinates.

    Evaluating end-to-end accuracy alone doesn't tell you *which* component
    broke. These metrics isolate each step.

Metrics implemented:
    Retrieval:
      - Recall@k        : did the relevant doc appear in the top-k results?
      - Precision@k     : what fraction of the top-k were relevant?
      - MRR             : how high up was the first relevant result?

    Generation (LLM-as-judge via Ollama):
      - Faithfulness    : is the answer supported by the retrieved context?
      - Answer Relevance: does the answer address the question asked?
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import ollama
from loguru import logger


# ── Retrieval metrics ─────────────────────────────────────────────────────────

def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Fraction of relevant documents that appeared in the top-k retrieved results.

    Recall@k = |relevant ∩ top-k| / |relevant|

    Interpretation:
        1.0 → all relevant documents were retrieved in the top-k
        0.0 → none of the relevant documents were retrieved
        0.5 → half of the relevant documents appeared in top-k

    Args:
        retrieved_ids: Ordered list of retrieved chunk or paper IDs (top-k first).
        relevant_ids:  Ground-truth set of IDs considered relevant for this query.
        k:             Cut-off rank (only the first k retrieved IDs are considered).

    Returns:
        Float in [0.0, 1.0].
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(top_k & relevant) / len(relevant)


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Fraction of the top-k retrieved results that were actually relevant.

    Precision@k = |relevant ∩ top-k| / k

    Interpretation:
        1.0 → every result in the top-k is relevant (high precision, maybe low recall)
        0.0 → none of the top-k results are relevant

    Note: Unlike Recall@k, Precision@k penalises returning too many results.
    A system that retrieves k=100 and only 2 are relevant has P@100 = 0.02.

    Args:
        retrieved_ids: Ordered list of retrieved IDs (top-k first).
        relevant_ids:  Ground-truth relevant IDs.
        k:             Cut-off rank.

    Returns:
        Float in [0.0, 1.0].
    """
    if k == 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(top_k & relevant) / k


def mean_reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """
    Reciprocal of the rank of the first relevant result, averaged across queries.

    MRR = 1 / rank_of_first_relevant

    Examples:
        First relevant result at rank 1 → MRR = 1.00 (perfect)
        First relevant result at rank 2 → MRR = 0.50
        First relevant result at rank 5 → MRR = 0.20
        No relevant result found        → MRR = 0.00

    This metric captures whether the most relevant document is at the top
    of the list — useful when users only read the first result.

    Args:
        retrieved_ids: Ordered list of retrieved IDs (best first).
        relevant_ids:  Ground-truth relevant IDs.

    Returns:
        Float in [0.0, 1.0].
    """
    relevant = set(relevant_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant:
            return 1.0 / rank
    return 0.0


def compute_retrieval_metrics(
    queries: list[dict],
    retriever,
    k: int = 5,
) -> dict:
    """
    Run a retriever over a list of evaluation queries and compute average metrics.

    Args:
        queries:   List of dicts with keys:
                     "question"     — query string
                     "relevant_ids" — list of paper_id strings that are relevant
        retriever: Any retriever with a .retrieve(query, k) method.
        k:         Cut-off rank for Recall@k and Precision@k.

    Returns:
        Dict with keys "recall@k", "precision@k", "mrr", "k", "n_queries".
    """
    recalls, precisions, mrrs = [], [], []

    for q in queries:
        results = retriever.retrieve(q["question"], k=k)
        retrieved_ids = [r["paper_id"] for r in results]
        relevant_ids = q["relevant_ids"]

        recalls.append(recall_at_k(retrieved_ids, relevant_ids, k))
        precisions.append(precision_at_k(retrieved_ids, relevant_ids, k))
        mrrs.append(mean_reciprocal_rank(retrieved_ids, relevant_ids))

    return {
        f"recall@{k}": round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        f"precision@{k}": round(sum(precisions) / len(precisions), 4) if precisions else 0.0,
        "mrr": round(sum(mrrs) / len(mrrs), 4) if mrrs else 0.0,
        "k": k,
        "n_queries": len(queries),
    }


# ── Generation metrics (LLM-as-judge) ────────────────────────────────────────

def score_faithfulness(
    answer: str,
    contexts: list[str],
    judge_model: str = "granite4.1:8b",
) -> float:
    """
    Ask an LLM to judge whether the answer is grounded in the retrieved contexts.

    Why LLM-as-judge?
        Traditional NLP metrics (BLEU, ROUGE) compare against a reference answer.
        For open-domain RAG there is often no single correct reference. An LLM
        judge can assess semantic grounding without needing a gold-standard answer.

    Faithfulness question asked to the judge:
        "Based ONLY on the provided context passages, is the following answer
         factually supported? Answer JSON: {"faithful": true/false, "reason": "..."}"

    Args:
        answer:      The generated answer to evaluate.
        contexts:    List of retrieved passage strings used to generate the answer.
        judge_model: Ollama model used as the judge (should be a strong instruct model).

    Returns:
        1.0 if the judge says faithful, 0.0 otherwise.
    """
    context_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts[:5]))
    prompt = f"""You are a strict factual evaluator. Read the context passages and the answer.
Determine if the answer is FULLY supported by the context — no claims go beyond what the context states.

Context:
{context_text}

Answer: {answer}

Respond with JSON only: {{"faithful": true or false, "reason": "one sentence"}}"""

    try:
        response = ollama.chat(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        result = json.loads(response["message"]["content"])
        return 1.0 if result.get("faithful", False) else 0.0
    except Exception as e:
        logger.warning(f"Faithfulness scoring failed: {e}")
        return 0.0


def score_answer_relevance(
    question: str,
    answer: str,
    judge_model: str = "granite4.1:8b",
) -> float:
    """
    Ask an LLM to judge whether the answer actually addresses the question.

    A faithful answer might still be irrelevant — e.g., it correctly quotes
    the context but the context doesn't address what was asked. This metric
    catches that failure mode.

    Args:
        question:    The original user question.
        answer:      The generated answer.
        judge_model: Ollama model used as the judge.

    Returns:
        Float in {0.0, 0.5, 1.0} based on judge rating (no/partial/yes).
    """
    prompt = f"""Rate how well the answer addresses the question.

Question: {question}
Answer: {answer}

Scoring:
  1.0 = answer fully and directly answers the question
  0.5 = answer is partially relevant but misses key aspects
  0.0 = answer does not address the question at all

Respond with JSON only: {{"score": 0.0 or 0.5 or 1.0, "reason": "one sentence"}}"""

    try:
        response = ollama.chat(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        result = json.loads(response["message"]["content"])
        score = float(result.get("score", 0.0))
        return min(max(score, 0.0), 1.0)  # clamp to [0, 1]
    except Exception as e:
        logger.warning(f"Answer relevance scoring failed: {e}")
        return 0.0


# ── Results storage ───────────────────────────────────────────────────────────

@dataclass
class EvalResults:
    """
    Container for evaluation results from one notebook run.
    Serialises to JSON for comparison across notebooks.
    """
    experiment_name: str = ""
    retriever_type: str = ""
    embed_model: str = ""
    llm_model: str = ""
    retrieval_metrics: dict = field(default_factory=dict)
    generation_metrics: dict = field(default_factory=dict)
    notes: str = ""

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2)
        logger.info(f"Saved eval results → {path}")

    @classmethod
    def load(cls, path: Path) -> "EvalResults":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def summary(self) -> str:
        lines = [
            f"Experiment : {self.experiment_name}",
            f"Retriever  : {self.retriever_type}",
            f"Embed model: {self.embed_model}",
            f"LLM        : {self.llm_model}",
            "─" * 40,
        ]
        for k, v in self.retrieval_metrics.items():
            lines.append(f"  {k:20s}: {v}")
        for k, v in self.generation_metrics.items():
            lines.append(f"  {k:20s}: {v}")
        if self.notes:
            lines.append(f"\nNotes: {self.notes}")
        return "\n".join(lines)
