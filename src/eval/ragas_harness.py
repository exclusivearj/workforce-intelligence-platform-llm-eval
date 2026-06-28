"""RAGAS-based evaluation harness.

Wraps the qa_generator output into a RAGAS Dataset, runs the metrics, and returns
a structured EvalRun. RAGAS/datasets are imported lazily so this module loads in
lightweight environments; tests exercise the assembly logic with a fake evaluator.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime

from src.eval.qa_generator import QAPair

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


@dataclass
class EvalRun:
    run_id: str
    scores: dict[str, float]
    row_scores: list[dict]
    model_name: str
    evaluated_at: datetime = field(default_factory=datetime.utcnow)


def default_generation_fn(question: str, contexts: list[str]) -> str:
    """Naive retrieval-augmented answer: echo the most relevant context."""
    if contexts:
        return contexts[0]
    return "No relevant context found."


def build_dataset(
    qa_pairs: list[QAPair],
    retrieval_fn: Callable[[str], list[str]],
    generation_fn: Callable[[str, list[str]], str],
) -> dict:
    """Assemble the columns RAGAS expects from the QA pairs + functions."""
    questions, answers, contexts, ground_truths = [], [], [], []
    for pair in qa_pairs:
        ctx = retrieval_fn(pair.question)
        questions.append(pair.question)
        contexts.append(ctx)
        answers.append(generation_fn(pair.question, ctx))
        ground_truths.append(pair.ground_truth)
    return {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }


def run_eval(
    qa_pairs: list[QAPair],
    retrieval_fn: Callable[[str], list[str]],
    generation_fn: Callable[[str, list[str]], str] | None = None,
    model_name: str = "all-MiniLM-L6-v2",
    evaluator: Callable[[dict], dict] | None = None,
) -> EvalRun:
    """Run a RAGAS evaluation. ``evaluator`` is injectable for testing.

    The default evaluator calls ``ragas.evaluate``; tests pass a fake to avoid the
    heavy dependency and any network/LLM calls.
    """
    generation_fn = generation_fn or default_generation_fn
    data = build_dataset(qa_pairs, retrieval_fn, generation_fn)
    evaluator = evaluator or _ragas_evaluator
    result = evaluator(data)

    scores = {m: float(result.get(m, 0.0)) for m in METRIC_NAMES if m in result}
    row_scores = result.get("_rows", [])
    # RAGAS returns per-row scores in dataset order (same order as qa_pairs); backfill
    # question_id so metrics_writer can link each row to its ground-truth pair.
    for i, row in enumerate(row_scores):
        if not row.get("question_id") and i < len(qa_pairs):
            row["question_id"] = qa_pairs[i].question_id
    return EvalRun(
        run_id=str(uuid.uuid4()),
        scores=scores,
        row_scores=row_scores,
        model_name=model_name,
    )


def _build_claude_judge():  # pragma: no cover - requires langchain-anthropic + API key
    """Return a RAGAS LLM backed by Claude.

    RAGAS scores its metrics with an LLM judge. We use Anthropic's Claude through the
    ``langchain-anthropic`` integration (RAGAS 0.1.x is built on LangChain, so this is
    the idiomatic way to plug in a non-OpenAI judge). The model is configurable via
    ``RAGAS_LLM_MODEL`` and defaults to ``claude-opus-4-8``.

    Claude Opus 4.8/4.7 reject sampling parameters (``temperature``/``top_p``/``top_k``)
    with a 400, but RAGAS forwards a ``temperature`` on every call. We therefore wrap the
    model in a RAGAS LLM that never passes sampling parameters, so the judge works on any
    Claude model.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set — the RAGAS eval judge requires a Claude API key. "
            "Set it in the environment or the repo-root .env, then re-run. "
            "(make setup / make embed do not need a key.)"
        )

    from langchain_anthropic import ChatAnthropic
    from ragas.llms import LangchainLLMWrapper

    chat = ChatAnthropic(
        model=os.getenv("RAGAS_LLM_MODEL", "claude-opus-4-8"),
        max_tokens=int(os.getenv("RAGAS_LLM_MAX_TOKENS", "1024")),
    )

    class _NoSamplingClaude(LangchainLLMWrapper):
        def generate_text(self, prompt, n=1, temperature=None, stop=None, callbacks=None):
            return self.langchain_llm.generate_prompt(
                [prompt], stop=stop, callbacks=callbacks or []
            )

        async def agenerate_text(
            self, prompt, n=1, temperature=None, stop=None, callbacks=None
        ):
            return await self.langchain_llm.agenerate_prompt(
                [prompt], stop=stop, callbacks=callbacks or []
            )

    return _NoSamplingClaude(chat)


def _build_local_embeddings():  # pragma: no cover - requires sentence-transformers
    """Local sentence-transformers embeddings for RAGAS — keeps the eval fully offline
    on the embeddings side so no OpenAI key is ever required."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    model = os.getenv("RAGAS_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=model))


def _maybe_float(value) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _ragas_evaluator(data: dict) -> dict:  # pragma: no cover - requires heavy deps + API key
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    dataset = Dataset.from_dict(data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=_build_claude_judge(),
        embeddings=_build_local_embeddings(),
    )

    scores: dict = {}
    for metric, value in dict(result).items():
        as_float = _maybe_float(value)
        if metric in METRIC_NAMES and as_float is not None:
            scores[metric] = as_float

    rows: list[dict] = []
    try:
        records = result.to_pandas().to_dict(orient="records")
    except Exception:  # to_pandas shape varies across ragas versions; means still returned
        records = []
    for rec in records:
        rows.append(
            {
                "question": rec.get("question", ""),
                "answer": rec.get("answer", ""),
                "contexts": list(rec.get("contexts", []) or []),
                **{m: _maybe_float(rec.get(m)) for m in METRIC_NAMES},
            }
        )
    scores["_rows"] = rows
    return scores
