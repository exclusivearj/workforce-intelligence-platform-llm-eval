"""Analyst feedback (thumbs up/down + corrections) read/write.

Feedback is stored separately from eval_results to keep model-quality metrics
distinct from human preference signals.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeedbackSummary:
    total_ratings: int
    positive_count: int
    negative_count: int
    correction_rate_pct: float
    by_role: dict[str, int]
    lowest_rated_questions: list[str]


def record_feedback(
    conn,
    eval_result_id: str,
    analyst_role: str,
    rating: int,
    correction_text: str | None = None,
) -> str:
    """Insert a feedback row and return its new id."""
    if rating not in (1, -1):
        raise ValueError("rating must be 1 (thumbs up) or -1 (thumbs down)")
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO llm.feedback
                (eval_result_id, analyst_role, rating, correction_text)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (eval_result_id, analyst_role, rating, correction_text),
        )
        new_id = cur.fetchone()[0]
    conn.commit()
    return str(new_id)


def get_feedback_summary(conn, days: int = 30) -> FeedbackSummary:
    """Aggregate feedback over the last ``days`` days."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                COUNT(*)                                   AS total,
                COUNT(*) FILTER (WHERE rating = 1)         AS positive,
                COUNT(*) FILTER (WHERE rating = -1)        AS negative,
                COUNT(*) FILTER (WHERE correction_text IS NOT NULL) AS corrections
            FROM llm.feedback
            WHERE created_at > NOW() - (%s || ' days')::interval
            """,
            (days,),
        )
        total, positive, negative, corrections = cur.fetchone()

        cur.execute(
            """
            SELECT analyst_role, COUNT(*)
            FROM llm.feedback
            WHERE created_at > NOW() - (%s || ' days')::interval
            GROUP BY analyst_role
            """,
            (days,),
        )
        by_role = {role: count for role, count in cur.fetchall()}

        cur.execute(
            """
            SELECT er.question
            FROM llm.feedback f
            JOIN llm.eval_results er ON er.id = f.eval_result_id
            WHERE f.rating = -1
            GROUP BY er.question
            ORDER BY COUNT(*) DESC
            LIMIT 5
            """
        )
        lowest = [r[0] for r in cur.fetchall()]

    total = total or 0
    correction_rate = round(100 * (corrections or 0) / total, 1) if total else 0.0
    return FeedbackSummary(
        total_ratings=total,
        positive_count=positive or 0,
        negative_count=negative or 0,
        correction_rate_pct=correction_rate,
        by_role=by_role,
        lowest_rated_questions=lowest,
    )
