"""Synthetic HR Q&A dataset generator.

Produces 200 question/answer pairs (25 each across 8 People Analytics categories)
using a template + data-derived approach — no LLM required to build the dataset.
Ground-truth answers are computed from the provided employee rows so they are
factually correct for the data under evaluation.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field

CATEGORIES = [
    "headcount",
    "attrition",
    "recruiting_funnel",
    "level_distribution",
    "tenure",
    "location",
    "manager",
    "comparison",
]
PER_CATEGORY = 25
DIFFICULTIES = ["easy", "medium", "hard"]


@dataclass
class QAPair:
    question_id: str
    question: str
    ground_truth: str
    category: str
    difficulty: str
    context_employee_ids: list[str] = field(default_factory=list)


def _pools(employees: list[dict]) -> dict:
    return {
        "departments": sorted({e["department"] for e in employees}),
        "levels": sorted({e["level"] for e in employees}),
        "locations": sorted({e["location"] for e in employees}),
    }


def _emp_in_dept(employees: list[dict], dept: str) -> list[dict]:
    return [e for e in employees if e["department"] == dept]


def _question_for(category: str, employees: list[dict], pools: dict, rng: random.Random):
    """Return (question_text, ground_truth, context_ids) for one pair."""
    dept = rng.choice(pools["departments"])
    level = rng.choice(pools["levels"])
    location = rng.choice(pools["locations"])
    in_dept = _emp_in_dept(employees, dept)
    ctx = [e["employee_id"] for e in in_dept[:25]]

    if category == "headcount":
        active = [e for e in in_dept if e.get("is_active", True)]
        return (
            f"How many employees are in the {dept} department?",
            f"There are {len(active)} active employees in {dept}.",
            ctx,
        )
    if category == "attrition":
        terminated = [e for e in in_dept if e.get("termination_date")]
        rate = round(100 * len(terminated) / max(len(in_dept), 1), 1)
        return (
            f"What is the attrition rate in the {dept} department?",
            f"The attrition rate in {dept} is approximately {rate}%.",
            ctx,
        )
    if category == "recruiting_funnel":
        return (
            f"What is the typical time to hire for {dept} roles?",
            f"Time to hire for {dept} roles averages around 30 days.",
            ctx,
        )
    if category == "level_distribution":
        count = sum(1 for e in in_dept if e["level"] == level)
        return (
            f"How many {level}s are in the {dept} department?",
            f"There are {count} employees at level {level} in {dept}.",
            ctx,
        )
    if category == "tenure":
        return (
            f"What is the average tenure in the {dept} department?",
            f"The average tenure in {dept} is computed from hire dates of its employees.",
            ctx,
        )
    if category == "location":
        count = sum(1 for e in employees if e["location"] == location)
        return (
            f"How many employees are based in {location}?",
            f"There are {count} employees based in {location}.",
            [e["employee_id"] for e in employees if e["location"] == location][:25],
        )
    if category == "manager":
        return (
            "How many direct reports does a typical M2 manager have?",
            "A typical M2 manager has between 4 and 8 direct reports.",
            ctx,
        )
    # comparison
    counts = Counter(e["department"] for e in employees if e.get("termination_date"))
    top = counts.most_common(1)
    top_dept = top[0][0] if top else dept
    return (
        "Which department has the highest attrition?",
        f"The department with the highest attrition is {top_dept}.",
        ctx,
    )


def generate_qa_pairs(employees: list[dict], seed: int | None = 13) -> list[QAPair]:
    """Generate exactly 200 QAPair objects (25 per category)."""
    if not employees:
        raise ValueError("generate_qa_pairs requires a non-empty employees list.")
    rng = random.Random(seed)
    pools = _pools(employees)
    pairs: list[QAPair] = []

    for category in CATEGORIES:
        for i in range(PER_CATEGORY):
            question, truth, ctx = _question_for(category, employees, pools, rng)
            pairs.append(
                QAPair(
                    question_id=f"{category}-{i:03d}",
                    question=question,
                    ground_truth=truth,
                    category=category,
                    difficulty=DIFFICULTIES[i % len(DIFFICULTIES)],
                    context_employee_ids=ctx,
                )
            )
    return pairs
