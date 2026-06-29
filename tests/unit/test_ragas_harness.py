"""Unit tests for the RAGAS harness assembly logic (evaluator injected)."""

from __future__ import annotations

import pytest

from src.eval import ragas_harness
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


def test_run_eval_backfills_question_id_by_order(sample_employees):
    pairs = generate_qa_pairs(sample_employees)[:3]

    def fake_evaluator(data: dict) -> dict:
        # rows in dataset order, without question_id (as real RAGAS returns)
        return {
            m: 0.8 for m in METRIC_NAMES
        } | {"_rows": [{"question": q} for q in data["question"]]}

    run = run_eval(pairs, _retrieval, evaluator=fake_evaluator)
    assert [r["question_id"] for r in run.row_scores] == [p.question_id for p in pairs]


def test_build_claude_judge_requires_api_key(monkeypatch):
    # The guard must fire before importing langchain-anthropic, so this stays hermetic.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        ragas_harness._build_claude_judge()


def test_judge_model_name_honors_env(monkeypatch):
    monkeypatch.setenv("RAGAS_LLM_MODEL", "claude-haiku-4-5")
    assert ragas_harness._judge_model_name() == "claude-haiku-4-5"
    monkeypatch.delenv("RAGAS_LLM_MODEL", raising=False)
    assert ragas_harness._judge_model_name() == ragas_harness.DEFAULT_JUDGE_MODEL


def test_run_eval_records_judge_model(sample_employees):
    # The evaluator reports the judge model it used → EvalRun.model_name reflects it
    # (so eval_results / cost_log attribute the Claude judge, not the embedding model).
    pairs = generate_qa_pairs(sample_employees)[:2]

    def fake_evaluator(data: dict) -> dict:
        return {m: 0.8 for m in METRIC_NAMES} | {"_model_name": "claude-haiku-4-5"}

    run = run_eval(pairs, _retrieval, evaluator=fake_evaluator, model_name="all-MiniLM-L6-v2")
    assert run.model_name == "claude-haiku-4-5"
