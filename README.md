# llm-eval — LLM evaluation data infrastructure

Part 2 of 4 in the [workforce-intelligence-platform](../README.md).

Builds the data foundation for AI-powered People Analytics tooling:
embedding store, eval harness, PII masking views, feedback loop, and cost tracking.

---

## Architecture

```
 analytics.dim_employees (from ingestion)
         │
         ▼
  llm.safe_employee_context   ← PII masking view (salary, perf excluded)
         │
         ▼
  LocalEncoder (sentence-transformers)
  OR OpenAIEncoder (text-embedding-3-small)
         │
         ▼
  llm.embeddings (pgvector, 384-dim, ivfflat index)
         │
         ├──────────────────────────────────────┐
         ▼                                      ▼
  Q&A eval dataset (200 pairs)         Similarity search
         │                              (top-k retrieval)
         ▼                                      │
  RAGAS eval harness ◄───────────────────────────┘
         │
         ├── llm.eval_results (per-question scores)
         ├── llm.cost_log     (token + cost tracking)
         └── llm.feedback     (analyst thumbs up/down)
                  │
                  ▼
         Airflow DAGs: embedding_refresh (triggered) + nightly_eval (2am)
```

---

## Tech stack

| Concern | Technology |
|---|---|
| Vector DB | pgvector 0.7 on Postgres 16 |
| Embeddings | sentence-transformers (local) / OpenAI text-embedding-3-small |
| Eval framework | RAGAS 0.1.9 |
| Data validation | Pydantic v2 |
| Orchestration | Apache Airflow 2.9 |
| Testing | pytest + testcontainers |

---

## Setup

### Prerequisites
- `ingestion/` fully set up and `analytics.dim_employees` populated

```bash
cd llm-eval
pip install -e ".[dev]"
make setup         # enables pgvector, creates llm schema, generates Q&A pairs, embeds
make eval          # runs RAGAS eval once and writes results
```

---

## Data readiness for AI

The `llm.safe_employee_context` view is the only approved data source for LLM context windows.
It explicitly excludes `salary`, `performance_rating`, and raw `manager_id` before data
reaches any embedding or completion pipeline. This implements the principle of least privilege
for AI systems: the model sees only what it needs to answer workforce questions.

The `llm.feedback` table captures analyst corrections and thumbs-down ratings, creating a
human-in-the-loop feedback loop that can drive future fine-tuning or prompt improvement.

---

## Eval metrics

| Metric | What it measures |
|---|---|
| `faithfulness` | Is the generated answer grounded in the retrieved context? |
| `answer_relevancy` | Does the answer actually address the question? |
| `context_precision` | Are the retrieved chunks relevant to the question? |
| `context_recall` | Did retrieval find all necessary information? |

Target thresholds: all metrics ≥ 0.70. Airflow alerts fire below this threshold.

---

## Design decisions

**sentence-transformers over OpenAI by default.** The `all-MiniLM-L6-v2` model runs locally,
costs nothing, has no API dependency, and produces 384-dimensional embeddings that fit comfortably
in a pgvector ivfflat index. OpenAI is available as an opt-in via `EMBEDDING_BACKEND=openai`
for production scenarios where quality matters more than cost.

**RAGAS over custom eval.** RAGAS provides standardised, reproducible metrics with published
benchmarks. Rolling your own eval framework is a maintenance liability — RAGAS scores are
comparable across model versions and meaningful to non-engineers reviewing the eval dashboard.

**Feedback separate from eval_results.** Eval results measure model quality against ground truth.
Feedback measures human preference. These are different signals and should be queryable
independently — a model can score high on faithfulness but still get thumbs-down from analysts
who find the phrasing unhelpful.
