"""Integration test: pgvector upsert + similarity search against real Postgres.

Run with: ``pytest tests/integration/ -m integration``
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

_DDL = """
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS llm;
CREATE TABLE IF NOT EXISTS llm.embeddings (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    source_table VARCHAR(100) NOT NULL,
    source_row_id UUID NOT NULL,
    content_text TEXT NOT NULL,
    embedding vector(384),
    model_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    refreshed_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (source_table, source_row_id, model_name)
);
"""


@pytest.fixture(scope="module")
def pg_conn():
    psycopg2 = pytest.importorskip("psycopg2")
    pytest.importorskip("pgvector")
    testcontainers = pytest.importorskip("testcontainers.postgres")

    with testcontainers.PostgresContainer("pgvector/pgvector:pg16") as pg:
        conn = psycopg2.connect(pg.get_connection_url().replace("+psycopg2", ""))
        with conn.cursor() as cur:
            cur.execute(_DDL)
        conn.commit()
        yield conn
        conn.close()


def test_upsert_and_search_idempotent(pg_conn):
    import uuid

    from src.embeddings.store import EmbeddingRecord, similarity_search, upsert_embeddings

    base = [0.0] * 384
    records = []
    for i in range(5):
        vec = list(base)
        vec[i] = 1.0
        records.append(
            EmbeddingRecord(
                source_table="analytics.dim_employees",
                source_row_id=str(uuid.uuid4()),
                content_text=f"emp {i}",
                embedding=vec,
            )
        )
    upsert_embeddings(pg_conn, records, "all-MiniLM-L6-v2")
    upsert_embeddings(pg_conn, records, "all-MiniLM-L6-v2")  # idempotent

    with pg_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM llm.embeddings")
        assert cur.fetchone()[0] == 5

    query = list(base)
    query[0] = 1.0
    top = similarity_search(pg_conn, query, top_k=1)
    assert top[0].content_text == "emp 0"
