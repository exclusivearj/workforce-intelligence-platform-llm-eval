"""Postgres connection helper for the llm-eval module.

Each module carries its own small db helper so it can be extracted into a
standalone repo. pgvector's type adapter is registered on demand by the store.
"""

from __future__ import annotations

import os


def get_connection():
    """Return a psycopg2 connection built from environment variables."""
    import psycopg2

    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        dbname=os.getenv("POSTGRES_DB", "workforce"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "changeme"),
    )


# Tables created by docker/init_llm_schema.sql (truncated on teardown, not dropped —
# init only runs on a fresh volume, so dropping them would break re-`setup`).
_LLM_TABLES = (
    "llm.embeddings",
    "llm.eval_results",
    "llm.feedback",
    "llm.cost_log",
)
# Views created at setup time by src.masking.views.apply_masking_views (safe to drop).
_MASKING_VIEWS = ("llm.safe_employee_context", "analytics.v_employees_safe")


def teardown_data(conn) -> list[str]:
    """Drop the PII masking views and truncate the ``llm`` tables this module owns.

    The inverse of ``make setup``: removes the setup-created masking views and clears
    the init-created ``llm`` tables, while keeping the tables themselves so
    ``make setup`` rebuilds cleanly. Missing objects are skipped (idempotent). Closes
    ``conn`` before returning. Returns a list of human-readable actions taken.
    """
    actions: list[str] = []
    with conn.cursor() as cur:
        for view in _MASKING_VIEWS:
            cur.execute(f"DROP VIEW IF EXISTS {view};")
            actions.append(f"dropped view {view}")
        present = []
        for table in _LLM_TABLES:
            cur.execute("SELECT to_regclass(%s)", (table,))
            if cur.fetchone()[0] is not None:
                present.append(table)
        if present:
            cur.execute(f"TRUNCATE {', '.join(present)} RESTART IDENTITY CASCADE;")
            actions.append("truncated " + ", ".join(present))
        else:
            actions.append("no llm tables present — nothing to truncate")
    conn.commit()
    conn.close()
    return actions
