"""Unit tests for metrics_writer, feedback store, masking apply, alerts, db."""

from __future__ import annotations

import sys
import types

import pytest

from tests.conftest import FakeConn, FakeCursor

from src.eval.metrics_writer import (
    get_latest_mean_scores,
    write_cost_log,
    write_eval_results,
)
from src.eval.qa_generator import generate_qa_pairs
from src.eval.ragas_harness import EvalRun
from src.feedback.store import get_feedback_summary, record_feedback
from src.masking.views import apply_masking_views


def test_write_eval_results(sample_employees):
    pairs = generate_qa_pairs(sample_employees)[:2]
    rows = [
        {"question_id": pairs[0].question_id, "question": pairs[0].question,
         "answer": "a", "contexts": ["c"], "faithfulness": 0.9,
         "answer_relevancy": 0.8, "context_precision": 0.7, "context_recall": 0.6},
    ]
    run = EvalRun("run-1", {"faithfulness": 0.9}, rows, "m")
    conn = FakeConn()
    assert write_eval_results(conn, run, pairs) == 1
    assert conn.committed


def test_write_cost_log():
    conn = FakeConn()
    write_cost_log(conn, "embedding", "m", embedding_count=5, cost_usd=0.0)
    assert conn.committed
    assert conn.cursor().executed


def test_get_latest_mean_scores():
    cur = FakeCursor(fetchone_queue=[(0.9, 0.8, 0.7, 0.6)])
    scores = get_latest_mean_scores(FakeConn(cur))
    assert scores["faithfulness"] == 0.9
    assert scores["context_recall"] == 0.6


def test_record_feedback_returns_id():
    cur = FakeCursor(fetchone_queue=[("fb-1",)])
    new_id = record_feedback(FakeConn(cur), "eval-1", "hr_partner", 1)
    assert new_id == "fb-1"


def test_record_feedback_rejects_bad_rating():
    with pytest.raises(ValueError, match="rating must be"):
        record_feedback(FakeConn(), "eval-1", "hr_partner", 5)


def test_get_feedback_summary():
    cur = FakeCursor(
        fetchone_queue=[(10, 7, 3, 2)],
        fetchall_queue=[[("hr_partner", 6), ("legal", 4)], [("bad question",)]],
    )
    summary = get_feedback_summary(FakeConn(cur))
    assert summary.total_ratings == 10
    assert summary.positive_count == 7
    assert summary.correction_rate_pct == 20.0
    assert summary.by_role["hr_partner"] == 6
    assert summary.lowest_rated_questions == ["bad question"]


def test_apply_masking_views_executes():
    conn = FakeConn()
    apply_masking_views(conn)
    assert len(conn.cursor().executed) == 1
    assert conn.committed


def test_db_get_connection(monkeypatch):
    captured = {}

    def fake_connect(**kwargs):
        captured.update(kwargs)
        return "CONN"

    pkg = types.ModuleType("psycopg2")
    pkg.connect = fake_connect
    monkeypatch.setitem(sys.modules, "psycopg2", pkg)
    monkeypatch.setenv("POSTGRES_DB", "workforce")

    from src.utils.db import get_connection

    assert get_connection() == "CONN"
    assert captured["dbname"] == "workforce"


def test_alerts_logs_without_webhook(monkeypatch, caplog):
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    from src.utils.alerts import send_slack_alert

    with caplog.at_level("WARNING"):
        assert send_slack_alert("hi") is False
