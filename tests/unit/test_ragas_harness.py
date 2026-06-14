"""Unit tests for the RAGAS harness assembly logic (evaluator injected)."""

from __future__ import annotations

from src.eval.qa_generator import generate_qa_pairs
from src.eval.ragas_harness import (
    METRIC_NAMES,
    build_dataset,
    default_generation_fn,
    run_eval,
)


def _retrieval(question: str) -> list[str]:
    return [f"context for: {question}"]


def test_build_dataset_shapes(sample_employees):
    pairs = generate_qa_pairs(sample_employees)[:10]
    data = build_dataset(pairs, _retrieval, default_generation_fn)
    assert set(data.keys()) == {"question", "answer", "contexts", "ground_truth"}
    assert all(len(v) == 10 for v in data.values())


def test_default_generation_uses_first_context():
    assert default_generation_fn("q", ["ctx1", "ctx2"]) == "ctx1"
    assert "No relevant" in default_generation_fn("q", [])


def test_run_eval_with_injected_evaluator(sample_employees):
    pairs = generate_qa_pairs(sample_employees)[:5]

    def fake_evaluator(data: dict) -> dict:
        return {m: 0.9 for m in METRIC_NAMES}

    run = run_eval(pairs, _retrieval, evaluator=fake_evaluator)
    assert set(run.scores.keys()) == set(METRIC_NAMES)
    assert run.scores["faithfulness"] == 0.9
    assert run.run_id
