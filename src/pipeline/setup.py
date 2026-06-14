"""One-shot local setup for the llm-eval module.

Applies the masking views, then embeds the safe employee context into pgvector.
Run with: ``python -m src.pipeline.setup``.
"""

from __future__ import annotations

from src.embeddings.pipeline import run_embedding_pipeline
from src.masking.views import apply_masking_views, verify_masking
from src.utils.db import get_connection


def main() -> dict:
    conn = get_connection()
    try:
        apply_masking_views(conn)
        verify_masking(conn)
        stats = run_embedding_pipeline(conn)
    finally:
        conn.close()
    print(f"Embedded {stats['rows_embedded']} rows with {stats['model_name']}.")
    return stats


if __name__ == "__main__":
    main()
