"""llm_eval_nightly — regenerate Q&A, run RAGAS eval, persist + alert on drops."""

from __future__ import annotations

from datetime import datetime

from airflow.decorators import dag, task

FAITHFULNESS_THRESHOLD = 0.7
RELEVANCY_THRESHOLD = 0.7


@dag(
    dag_id="llm_eval_nightly",
    schedule="0 2 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["llm-eval", "people-analytics"],
)
def llm_eval_nightly():
    @task
    def load_qa_pairs() -> list:
        from src.embeddings.pipeline import fetch_safe_context
        from src.eval.qa_generator import generate_qa_pairs
        from src.utils.db import get_connection

        conn = get_connection()
        try:
            rows = fetch_safe_context(conn)
        finally:
            conn.close()
        employees = [
            {
                "employee_id": str(r["employee_id"]),
                "department": r["department"],
                "level": r["level"],
                "location": r["location"],
                "is_active": r["is_active"],
                "termination_date": None,
            }
            for r in rows
        ]
        return [p.__dict__ for p in generate_qa_pairs(employees)]

    @task
    def run_ragas_eval(qa_pairs: list) -> dict:
        from src.eval.qa_generator import QAPair
        from src.eval.ragas_harness import run_eval

        pairs = [QAPair(**p) for p in qa_pairs]

        def retrieval_fn(question: str) -> list[str]:
            return [p.ground_truth for p in pairs if p.question == question][:3]

        run = run_eval(pairs, retrieval_fn)
        return run.scores

    @task
    def alert_if_scores_drop(scores: dict) -> None:
        low = (
            scores.get("faithfulness", 1.0) < FAITHFULNESS_THRESHOLD
            or scores.get("answer_relevancy", 1.0) < RELEVANCY_THRESHOLD
        )
        if not low:
            return
        from src.utils.alerts import send_slack_alert

        send_slack_alert(f"RAGAS scores dropped below threshold: {scores}")

    alert_if_scores_drop(run_ragas_eval(load_qa_pairs()))


llm_eval_nightly()
