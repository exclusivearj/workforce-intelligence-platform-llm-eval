"""Persist eval runs and cost rows to Postgres."""

from __future__ import annotations

from src.eval.qa_generator import QAPair
from src.eval.ragas_harness import EvalRun


def write_eval_results(conn, eval_run: EvalRun, qa_pairs: list[QAPair]) -> int:
    """Write per-question scores to llm.eval_results. Returns rows written."""
    rows = eval_run.row_scores or []
    by_id = {p.question_id: p for p in qa_pairs}
    written = 0
    with conn.cursor() as cur:
        for row in rows:
            pair = by_id.get(row.get("question_id"))
            cur.execute(
                """
                INSERT INTO llm.eval_results
                    (run_id, question, ground_truth, generated_answer,
                     retrieved_contexts, faithfulness, answer_relevancy,
                     context_precision, context_recall, model_name)
                VALUES (%(run_id)s, %(q)s, %(gt)s, %(ans)s, %(ctx)s,
                        %(f)s, %(ar)s, %(cp)s, %(cr)s, %(model)s)
                """,
                {
                    "run_id": eval_run.run_id,
                    "q": row.get("question", pair.question if pair else ""),
                    "gt": pair.ground_truth if pair else "",
                    "ans": row.get("answer", ""),
                    "ctx": row.get("contexts", []),
                    "f": row.get("faithfulness"),
                    "ar": row.get("answer_relevancy"),
                    "cp": row.get("context_precision"),
                    "cr": row.get("context_recall"),
                    "model": eval_run.model_name,
                },
            )
            written += 1
    conn.commit()
    return written


def write_cost_log(conn, run_type: str, model_name: str, **kwargs) -> None:
    """Write a cost-tracking row to llm.cost_log."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO llm.cost_log
                (run_type, model_name, input_tokens, output_tokens,
                 embedding_count, cost_usd)
            VALUES (%(rt)s, %(model)s, %(it)s, %(ot)s, %(ec)s, %(cost)s)
            """,
            {
                "rt": run_type,
                "model": model_name,
                "it": kwargs.get("input_tokens"),
                "ot": kwargs.get("output_tokens"),
                "ec": kwargs.get("embedding_count"),
                "cost": kwargs.get("cost_usd", 0.0),
            },
        )
    conn.commit()


def get_latest_mean_scores(conn) -> dict[str, float]:
    """Return the most recent run's mean metric scores (for alerting)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT AVG(faithfulness), AVG(answer_relevancy),
                   AVG(context_precision), AVG(context_recall)
            FROM llm.eval_results
            WHERE run_id = (
                SELECT run_id FROM llm.eval_results
                ORDER BY evaluated_at DESC LIMIT 1
            )
            """
        )
        row = cur.fetchone() or (None, None, None, None)
    keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    return {k: float(v) for k, v in zip(keys, row) if v is not None}
