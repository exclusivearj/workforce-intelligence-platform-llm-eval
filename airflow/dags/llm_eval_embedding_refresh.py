"""llm_eval_embedding_refresh — triggered by the hr_ingestion DAG on data change."""

from __future__ import annotations

from airflow.decorators import dag, task


@dag(
    dag_id="llm_eval_embedding_refresh",
    schedule=None,  # triggered by hr_ingestion only
    catchup=False,
    tags=["llm-eval", "people-analytics"],
)
def llm_eval_embedding_refresh():
    @task
    def check_source_freshness() -> bool:
        from src.utils.db import get_connection

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT MAX(updated_at) > COALESCE(
                        (SELECT MAX(refreshed_at) FROM llm.embeddings), 'epoch'
                    )
                    FROM analytics.dim_employees
                    """
                )
                row = cur.fetchone()
        finally:
            conn.close()
        return bool(row[0]) if row else True

    @task
    def refresh_embeddings(has_changes: bool) -> dict:
        if not has_changes:
            return {"rows_embedded": 0, "skipped": True}
        from src.embeddings.pipeline import run_embedding_pipeline

        return run_embedding_pipeline()

    @task
    def write_cost_log(stats: dict) -> None:
        from src.eval.metrics_writer import write_cost_log
        from src.utils.db import get_connection

        conn = get_connection()
        try:
            write_cost_log(
                conn,
                run_type="embedding",
                model_name=stats.get("model_name", "all-MiniLM-L6-v2"),
                embedding_count=stats.get("rows_embedded", 0),
                cost_usd=0.0,
            )
        finally:
            conn.close()

    write_cost_log(refresh_embeddings(check_source_freshness()))


llm_eval_embedding_refresh()
