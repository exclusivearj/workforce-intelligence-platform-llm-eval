"""PII masking views for LLM-safe access to employee data.

These views guarantee that salary and performance_rating never appear in any
result set that could flow into an LLM context window, and that manager_id is
always hashed rather than exposed as a raw identifier.
"""

from __future__ import annotations

from dataclasses import dataclass

FORBIDDEN_COLUMNS = {"salary", "performance_rating"}

SAFE_EMPLOYEE_CONTEXT_SQL = """
CREATE OR REPLACE VIEW llm.safe_employee_context AS
SELECT
    employee_id,
    department,
    job_title,
    level,
    hire_date,
    is_active,
    employment_type,
    location,
    MD5(COALESCE(manager_id::text, '')) AS manager_id_hashed
FROM analytics.dim_employees;
"""

# NOTE: analytics.v_employees_safe is owned by the 3-governance module, which generates it
# from policies/data_classification.yml (it masks confidential PII such as full_name/email
# rather than dropping it). This module owns only the LLM-context view above. Defining
# v_employees_safe here too caused a last-writer-wins conflict on the shared database.


class MaskingViolation(Exception):
    """Raised when a masked view leaks a forbidden column."""


@dataclass
class MaskingVerificationResult:
    view_name: str
    ok: bool
    detail: str


def apply_masking_views(conn) -> None:
    """(Re)create the LLM-safe employee-context view. Idempotent."""
    with conn.cursor() as cur:
        cur.execute(SAFE_EMPLOYEE_CONTEXT_SQL)
    conn.commit()


def _view_columns(conn, view_name: str) -> list[str]:
    schema, _, name = view_name.partition(".")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema, name),
        )
        return [r[0] for r in cur.fetchall()]


def verify_masking(conn) -> list[MaskingVerificationResult]:
    """Assert each masked view excludes forbidden columns and hashes manager_id.

    Raises MaskingViolation on the first leak so callers fail loudly.
    """
    results: list[MaskingVerificationResult] = []
    for view in ("llm.safe_employee_context",):
        cols = set(_view_columns(conn, view))
        leaked = cols & FORBIDDEN_COLUMNS
        if leaked:
            raise MaskingViolation(f"{view} exposes forbidden columns: {sorted(leaked)}")
        if "manager_id" in cols:
            raise MaskingViolation(f"{view} exposes raw manager_id (must be hashed).")
        results.append(
            MaskingVerificationResult(view, True, "no forbidden columns present")
        )
    return results
