"""Shared fixtures for the llm-eval test suite."""

from __future__ import annotations

import random
import sys
import types

import pytest


class FakeCursor:
    """Minimal psycopg2-style cursor that records SQL and returns queued rows."""

    def __init__(self, fetchone_queue=None, fetchall_queue=None, description=None):
        self.executed: list[tuple] = []
        self._fetchone = list(fetchone_queue or [])
        self._fetchall = list(fetchall_queue or [])
        self.description = description

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))

    def fetchone(self):
        return self._fetchone.pop(0) if self._fetchone else None

    def fetchall(self):
        return self._fetchall.pop(0) if self._fetchall else []


class FakeConn:
    def __init__(self, cursor: FakeCursor | None = None):
        self._cursor = cursor or FakeCursor()
        self.committed = False
        self.closed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def close(self):
        self.closed = True


@pytest.fixture
def fake_psycopg2(monkeypatch):
    """Inject a fake psycopg2.extras.execute_values + fake pgvector adapter."""
    captured = {"rows": None}

    def execute_values(cur, sql, rows, template=None, page_size=100):
        captured["rows"] = list(rows)
        cur.execute(sql)

    extras = types.ModuleType("psycopg2.extras")
    extras.execute_values = execute_values
    pkg = types.ModuleType("psycopg2")
    pkg.extras = extras
    monkeypatch.setitem(sys.modules, "psycopg2", pkg)
    monkeypatch.setitem(sys.modules, "psycopg2.extras", extras)

    pgvector = types.ModuleType("pgvector")
    pgvector_psycopg2 = types.ModuleType("pgvector.psycopg2")
    pgvector_psycopg2.register_vector = lambda conn: None
    pgvector.psycopg2 = pgvector_psycopg2
    monkeypatch.setitem(sys.modules, "pgvector", pgvector)
    monkeypatch.setitem(sys.modules, "pgvector.psycopg2", pgvector_psycopg2)
    return captured

_DEPTS = ["Engineering", "Data", "Product", "Design"]
_LEVELS = ["IC1", "IC2", "IC3", "M1", "M2"]
_LOCS = ["San Francisco, CA", "Remote - US", "London, UK"]


@pytest.fixture
def sample_employees() -> list[dict]:
    rng = random.Random(5)
    employees = []
    for i in range(120):
        terminated = rng.random() < 0.15
        employees.append(
            {
                "employee_id": f"emp-{i:04d}",
                "department": rng.choice(_DEPTS),
                "level": rng.choice(_LEVELS),
                "location": rng.choice(_LOCS),
                "is_active": not terminated,
                "termination_date": "2024-01-01" if terminated else None,
            }
        )
    return employees
