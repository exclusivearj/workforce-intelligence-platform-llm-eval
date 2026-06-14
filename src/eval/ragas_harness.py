"""RAGAS-based evaluation harness.

Wraps the qa_generator output into a RAGAS Dataset, runs the metrics, and returns
a structured EvalRun. RAGAS/datasets are imported lazily so this module loads in
lightweight environments; tests exercise the assembly logic with a fake evaluator.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from src.eval.qa_generator import QAPair

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


@dataclass
class EvalRun:
    run_id: str
    scores: dict[str, float]
    row_scores: list[dict]
    model_name: str
    evaluated_at: datetime = field(default_factory=datetime.utcnow)


def default_generation_fn(question: str, contexts: list[str]) -> str:
    """Naive retrieval-augmented answer: echo the most relevant context."""
    if contexts:
        return contexts[0]
    return "No relevant context found."


def build_dataset(
    qa_pairs: list[QAPair],
    retrieval_fn: Callable[[str], list[str]],
    generation_fn: Callable[[str, list[str]], str],
) -> dict:
    """Assemble the columns RAGAS expects from the QA pairs + functions."""
    questions, answers, contexts, ground_truths = [], [], [], []
    for pair in qa_pairs:
        ctx = retrieval_fn(pair.question)
        questions.append(pair.question)
        contexts.append(ctx)
        answers.append(generation_fn(pair.question, ctx))
        ground_truths.append(pair.ground_truth)
    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def run_eval(
    qa_pairs: list[QAPair],
    retrieval_fn: Callable[[str], list[str]],
    generation_fn: Callable[[str, list[str]], str] | None = None,
    model_name: str = "all-MiniLM-L6-v2",
    evaluator: Callable[[dict], dict] | None = None,
) -> EvalRun:
    """Run a RAGAS evaluation. ``evaluator`` is injectable for testing.

    The default evaluator calls ``ragas.evaluate``; tests pass a fake to avoid the
    heavy dependency and any network/LLM calls.
    """
    generation_fn = generation_fn or default_generation_fn
    data = build_dataset(qa_pairs, retrieval_fn, generation_fn)
    evaluator = evaluator or _ragas_evaluator
    result = evaluator(data)

    scores = {m: float(result.get(m, 0.0)) for m in METRIC_NAMES if m in result}
    row_scores = result.get("_rows", [])
    return EvalRun(
        run_id=str(uuid.uuid4()),
        scores=scores,
        row_scores=row_scores,
        model_name=model_name,
    )


def _ragas_evaluator(data: dict) -> dict:  # pragma: no cover - requires heavy deps
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = Dataset.from_dict(data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    return dict(result)
