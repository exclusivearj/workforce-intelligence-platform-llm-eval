"""Run a full RAGAS evaluation and persist results + cost.

Run with: ``python -m src.pipeline.eval``.

Retrieval and generation use the local baseline, and RAGAS metric embeddings run on the
local sentence-transformers model — so nothing here calls OpenAI. RAGAS still scores its
metrics with an LLM judge: that judge is Claude (``langchain-anthropic``), so this requires
``ANTHROPIC_API_KEY`` to be set. The judge model defaults to ``claude-opus-4-8`` and is
configurable via ``RAGAS_LLM_MODEL`` (e.g. ``claude-haiku-4-5`` to cut cost).
"""

from __future__ import annotations

from src.embeddings.pipeline import fetch_safe_context
from src.eval.metrics_writer import write_cost_log, write_eval_results
from src.eval.qa_generator import generate_qa_pairs
from src.eval.ragas_harness import run_eval
from src.utils.db import get_connection


def main() -> dict:
    conn = get_connection()
    try:
        rows = fetch_safe_context(conn)
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
        pairs = generate_qa_pairs(employees)
        truths = {p.question: p.ground_truth for p in pairs}

        def retrieval_fn(question: str) -> list[str]:
            return [truths.get(question, "")]

        run = run_eval(pairs, retrieval_fn)
        write_eval_results(conn, run, pairs)
        write_cost_log(conn, run_type="eval", model_name=run.model_name, cost_usd=0.0)
    finally:
        conn.close()
    print(f"Eval run {run.run_id} complete: {run.scores}")
    return run.scores


if __name__ == "__main__":
    main()
