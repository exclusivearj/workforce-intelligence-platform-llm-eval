"""Unit tests for the db helpers (teardown logic, fully mocked)."""

from __future__ import annotations

from tests.conftest import FakeConn, FakeCursor

from src.utils import db


def test_teardown_data_drops_views_and_truncates_present_tables():
    # to_regclass returns a non-None row for each of the 4 llm tables → all present.
    cur = FakeCursor(fetchone_queue=[("llm.embeddings",)] * 4)
    conn = FakeConn(cur)

    actions = db.teardown_data(conn)

    sql = " ".join(s for s, _ in cur.executed)
    assert "DROP VIEW IF EXISTS llm.safe_employee_context" in sql
    assert "DROP VIEW IF EXISTS analytics.v_employees_safe" in sql
    assert "TRUNCATE" in sql and "RESTART IDENTITY CASCADE" in sql
    assert conn.committed and conn.closed
    assert any("truncated" in a for a in actions)
    assert any("dropped view" in a for a in actions)


def test_teardown_data_skips_truncate_when_no_tables():
    # to_regclass returns None for every table → nothing to truncate.
    cur = FakeCursor(fetchone_queue=[(None,)] * 4)
    conn = FakeConn(cur)

    actions = db.teardown_data(conn)

    sql = " ".join(s for s, _ in cur.executed)
    assert "TRUNCATE" not in sql
    assert any("nothing to truncate" in a for a in actions)
    assert conn.committed and conn.closed
