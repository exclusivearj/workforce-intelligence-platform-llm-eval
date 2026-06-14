"""Unit tests for the embedding pipeline + setup/eval orchestration."""

from __future__ import annotations

from tests.conftest import FakeConn, FakeCursor

from src.embeddings import pipeline as pipeline_module
from src.embeddings.pipeline import _row_to_text, fetch_safe_context, run_embedding_pipeline


def test_row_to_text_excludes_sensitive():
    text = _row_to_text(
        {
            "department": "Data",
            "job_title": "Engineer",
            "level": "IC3",
            "location": "Remote - US",
            "hire_date": "2020-01-01",
            "is_active": True,
            "employment_type": "full_time",
        }
    )
    assert "Data" in text and "salary" not in text.lower()


def test_fetch_safe_context_maps_columns():
    cur = FakeCursor(
        fetchall_queue=[[("emp-1", "Data", "Eng", "IC3", "2020-01-01", True, "full_time", "Remote")]],
        description=[
            ("employee_id",), ("department",), ("job_title",), ("level",),
            ("hire_date",), ("is_active",), ("employment_type",), ("location",),
        ],
    )
    rows = fetch_safe_context(FakeConn(cur))
    assert rows[0]["employee_id"] == "emp-1"
    assert rows[0]["department"] == "Data"


def test_run_embedding_pipeline(monkeypatch):
    rows = [
        {
            "employee_id": "emp-1",
            "department": "Data",
            "job_title": "Eng",
            "level": "IC3",
            "location": "Remote",
            "hire_date": "2020-01-01",
            "is_active": True,
            "employment_type": "full_time",
        }
    ]
    monkeypatch.setattr(pipeline_module, "fetch_safe_context", lambda conn: rows)

    class _Enc:
        model_name = "all-MiniLM-L6-v2"

        def encode(self, texts):
            return [[0.1] * 384 for _ in texts]

    monkeypatch.setattr(pipeline_module, "get_encoder", lambda: _Enc())
    captured = {}
    monkeypatch.setattr(
        pipeline_module, "upsert_embeddings",
        lambda conn, records, model: captured.setdefault("n", len(records)) or len(records),
    )

    stats = run_embedding_pipeline(conn=FakeConn())
    assert stats["rows_embedded"] == 1
    assert stats["model_name"] == "all-MiniLM-L6-v2"


def test_setup_main(monkeypatch):
    from src.pipeline import setup as setup_module

    monkeypatch.setattr(setup_module, "get_connection", lambda: FakeConn())
    monkeypatch.setattr(setup_module, "apply_masking_views", lambda conn: None)
    monkeypatch.setattr(setup_module, "verify_masking", lambda conn: [])
    monkeypatch.setattr(
        setup_module, "run_embedding_pipeline",
        lambda conn: {"rows_embedded": 3, "model_name": "m"},
    )
    assert setup_module.main()["rows_embedded"] == 3


def test_eval_main(monkeypatch):
    from src.pipeline import eval as eval_module
    from src.eval.ragas_harness import EvalRun

    monkeypatch.setattr(eval_module, "get_connection", lambda: FakeConn())
    monkeypatch.setattr(
        eval_module, "fetch_safe_context",
        lambda conn: [
            {"employee_id": "e1", "department": "Data", "level": "IC3",
             "location": "Remote", "is_active": True}
        ],
    )
    monkeypatch.setattr(
        eval_module, "run_eval",
        lambda pairs, retrieval_fn: EvalRun("run-1", {"faithfulness": 0.9}, [], "m"),
    )
    monkeypatch.setattr(eval_module, "write_eval_results", lambda *a, **k: 0)
    monkeypatch.setattr(eval_module, "write_cost_log", lambda *a, **k: None)
    scores = eval_module.main()
    assert scores["faithfulness"] == 0.9
