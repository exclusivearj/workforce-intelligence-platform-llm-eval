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
