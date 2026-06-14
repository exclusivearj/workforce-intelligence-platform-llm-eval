"""Unit tests for the synthetic Q&A generator."""

from __future__ import annotations

import pytest

from src.eval.qa_generator import CATEGORIES, PER_CATEGORY, generate_qa_pairs


def test_generates_200_pairs(sample_employees):
    pairs = generate_qa_pairs(sample_employees)
    assert len(pairs) == 200


def test_25_per_category(sample_employees):
    pairs = generate_qa_pairs(sample_employees)
    counts = {c: 0 for c in CATEGORIES}
    for p in pairs:
        counts[p.category] += 1
    assert all(v == PER_CATEGORY for v in counts.values())


def test_ground_truth_non_empty(sample_employees):
    pairs = generate_qa_pairs(sample_employees)
    assert all(p.ground_truth.strip() for p in pairs)


def test_question_ids_unique(sample_employees):
    pairs = generate_qa_pairs(sample_employees)
    ids = [p.question_id for p in pairs]
    assert len(ids) == len(set(ids))


def test_empty_employees_raises():
    with pytest.raises(ValueError, match="non-empty"):
        generate_qa_pairs([])
