"""Unit tests for PII masking verification."""

from __future__ import annotations

import pytest

from src.masking import views as views_module
from src.masking.views import MaskingViolation, verify_masking

SAFE_COLUMNS = [
    "employee_id",
    "department",
    "job_title",
    "level",
    "hire_date",
    "is_active",
    "employment_type",
    "location",
    "manager_id_hashed",
]


def test_verify_masking_passes_when_safe(monkeypatch):
    monkeypatch.setattr(views_module, "_view_columns", lambda conn, view: list(SAFE_COLUMNS))
    results = verify_masking(conn=object())
    assert len(results) == 1
    assert all(r.ok for r in results)


def test_verify_masking_raises_on_salary(monkeypatch):
    monkeypatch.setattr(
        views_module, "_view_columns", lambda conn, view: [*SAFE_COLUMNS, "salary"]
    )
    with pytest.raises(MaskingViolation, match="forbidden columns"):
        verify_masking(conn=object())


def test_verify_masking_raises_on_performance(monkeypatch):
    monkeypatch.setattr(
        views_module,
        "_view_columns",
        lambda conn, view: [*SAFE_COLUMNS, "performance_rating"],
    )
    with pytest.raises(MaskingViolation):
        verify_masking(conn=object())


def test_verify_masking_raises_on_raw_manager_id(monkeypatch):
    cols = [c for c in SAFE_COLUMNS if c != "manager_id_hashed"] + ["manager_id"]
    monkeypatch.setattr(views_module, "_view_columns", lambda conn, view: cols)
    with pytest.raises(MaskingViolation, match="manager_id"):
        verify_masking(conn=object())
