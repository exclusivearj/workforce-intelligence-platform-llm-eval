"""pgvector read/write operations for llm.embeddings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EmbeddingRecord:
    source_table: str
    source_row_id: str
    content_text: str
    embedding: list[float]


@dataclass
class SimilarityResult:
    source_row_id: str
    content_text: str
    score: float


def _register_vector(conn) -> None:
    from pgvector.psycopg2 import register_vector

    register_vector(conn)


def upsert_embeddings(conn, records: list[EmbeddingRecord], model_name: str) -> int:
    """Upsert embedding vectors keyed on (source_table, source_row_id, model_name)."""
    if not records:
        return 0
    from psycopg2.extras import execute_values

    _register_vector(conn)
    rows = [
        (r.source_table, r.source_row_id, r.content_text, r.embedding, model_name)
        for r in records
    ]
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO llm.embeddings
                (source_table, source_row_id, content_text, embedding, model_name)
            VALUES %s
            ON CONFLICT (source_table, source_row_id, model_name) DO UPDATE
                SET content_text = EXCLUDED.content_text,
                    embedding    = EXCLUDED.embedding,
                    refreshed_at = NOW()
            """,
            rows,
            template="(%s, %s::uuid, %s, %s::vector, %s)",
        )
    conn.commit()
    return len(rows)


def similarity_search(
    conn,
    query_vector: list[float],
    top_k: int = 5,
    source_table: str = "analytics.dim_employees",
) -> list[SimilarityResult]:
    """Return the ``top_k`` nearest neighbours by cosine distance."""
    _register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT source_row_id, content_text,
                   1 - (embedding <=> %(q)s::vector) AS score
            FROM llm.embeddings
            WHERE source_table = %(t)s
            ORDER BY embedding <=> %(q)s::vector
            LIMIT %(k)s
            """,
            {"q": query_vector, "t": source_table, "k": top_k},
        )
        return [
            SimilarityResult(str(row[0]), row[1], float(row[2]))
            for row in cur.fetchall()
        ]
