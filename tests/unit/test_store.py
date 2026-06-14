"""Unit tests for the pgvector store (fake psycopg2 + pgvector)."""

from __future__ import annotations

from tests.conftest import FakeConn, FakeCursor

from src.embeddings.store import (
    EmbeddingRecord,
    SimilarityResult,
    similarity_search,
    upsert_embeddings,
)


def _record(i: int) -> EmbeddingRecord:
    return EmbeddingRecord(
        source_table="analytics.dim_employees",
        source_row_id=f"00000000-0000-0000-0000-00000000000{i}",
        content_text=f"text {i}",
        embedding=[0.1] * 384,
    )


def test_upsert_embeddings_returns_count(fake_psycopg2):
    conn = FakeConn()
    n = upsert_embeddings(conn, [_record(1), _record(2)], "all-MiniLM-L6-v2")
    assert n == 2
    assert len(fake_psycopg2["rows"]) == 2
    assert conn.committed


def test_upsert_embeddings_empty(fake_psycopg2):
    assert upsert_embeddings(FakeConn(), [], "m") == 0
    assert fake_psycopg2["rows"] is None


def test_similarity_search_maps_rows(fake_psycopg2):
    cur = FakeCursor(fetchall_queue=[[("row-1", "hello", 0.92), ("row-2", "world", 0.81)]])
    results = similarity_search(FakeConn(cur), [0.1] * 384, top_k=2)
    assert all(isinstance(r, SimilarityResult) for r in results)
    assert results[0].source_row_id == "row-1"
    assert results[0].score == 0.92
