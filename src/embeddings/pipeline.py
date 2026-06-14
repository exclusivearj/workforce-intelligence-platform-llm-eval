"""Orchestrates encode -> store for the safe employee context view."""

from __future__ import annotations

from src.embeddings.encoder import get_encoder
from src.embeddings.store import EmbeddingRecord, upsert_embeddings


def _row_to_text(row: dict) -> str:
    """Render a safe_employee_context row into a sentence for embedding."""
    status = "active" if row.get("is_active") else "former"
    return (
        f"{status} {row.get('employment_type', '')} employee in "
        f"{row.get('department', '')} as {row.get('job_title', '')} "
        f"at level {row.get('level', '')}, based in {row.get('location', '')}, "
        f"hired {row.get('hire_date', '')}."
    )


def fetch_safe_context(conn) -> list[dict]:
    """Read the PII-safe employee context view (never salary/performance)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT employee_id, department, job_title, level, hire_date,
                   is_active, employment_type, location
            FROM llm.safe_employee_context
            """
        )
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def run_embedding_pipeline(conn=None) -> dict:
    """Embed every safe_employee_context row and upsert into llm.embeddings."""
    from src.utils.db import get_connection

    owns_conn = conn is None
    conn = conn or get_connection()
    try:
        rows = fetch_safe_context(conn)
        encoder = get_encoder()
        texts = [_row_to_text(r) for r in rows]
        vectors = encoder.encode(texts) if texts else []
        records = [
            EmbeddingRecord(
                source_table="analytics.dim_employees",
                source_row_id=str(row["employee_id"]),
                content_text=text,
                embedding=vec,
            )
            for row, text, vec in zip(rows, texts, vectors)
        ]
        count = upsert_embeddings(conn, records, encoder.model_name)
    finally:
        if owns_conn:
            conn.close()
    return {"rows_embedded": count, "model_name": encoder.model_name}
