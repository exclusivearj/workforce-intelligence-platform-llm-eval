![workforce-intelligence-platform-llm-eval banner](assets/02-llm-eval-banner.png)

# workforce-intelligence-platform-llm-eval — safe, evaluated LLM infrastructure for People Analytics

Part 2 of 4 in the [workforce-intelligence-platform](../README.md).

Builds the data foundation for AI-powered People Analytics tooling:
embedding store, eval harness, PII masking views, feedback loop, and cost tracking.

---

## What this part accomplishes

Everyone wants to put a chatbot in front of their HR data — *"how many L5+ engineers
do we have in Dublin, and how does that compare to last quarter?"* — answered in plain
English instead of a SQL query. The hard part isn't wiring up an LLM. It's making that
answer **safe to expose** and **trustworthy enough to act on**. This module builds the
data infrastructure that has to exist *before* any such feature ships:

- a **vector store** (`pgvector`) so the model can retrieve the right employee context for a question;
- a **PII-masking layer** so the model only ever sees data it's allowed to see;
- a **reproducible eval harness** (RAGAS) that scores answer quality against a ground-truth dataset, on a schedule, with alerting;
- a **cost log** and a **human feedback loop** so spend and quality stay observable over time.

In short: it treats *"can we trust this AI, and is it safe?"* as a data-engineering
problem — tables, tests, thresholds, and schedules — rather than a one-off prompt.

### Why it matters

HR data is among the most sensitive data a company holds (salary, performance ratings,
reporting chains), and LLMs hallucinate confidently. Those two facts make a naïve
"point an LLM at the warehouse" approach a liability:

- **Privacy.** Feeding raw HR rows into an embedding or completion pipeline leaks compensation and performance data into prompts, logs, and third-party APIs. The `llm.safe_employee_context` view is the single approved context source — it strips `salary`, `performance_rating`, and raw `manager_id` *before* data reaches any model. This is least privilege applied to AI: the model sees only what it needs to answer the question.
- **Trust.** An HR leader making a headcount or attrition decision on a plausible-but-wrong answer is a real harm. The RAGAS harness measures faithfulness, relevancy, and retrieval quality against 200 known-answer questions and gates them at a ≥ 0.70 threshold. The nightly Airflow DAG re-runs that eval at 2am and alerts on regressions — so quality drift from a model upgrade or a data change is caught by a pipeline, not by a user getting a bad answer.

This is the difference between an impressive demo and something a People Analytics team
would actually let HR business partners use.

### Use case, end to end

1. An HR partner asks a natural-language workforce question in the dashboard.
2. The question is embedded and used to retrieve the most relevant rows from `llm.embeddings` — which, by construction, only ever contains **masked, salary-free** context.
3. An LLM composes an answer grounded in that retrieved context.
4. The nightly eval has *already* proven, on 200 ground-truth questions, that this retrieval-and-answer pipeline scores above threshold — so the answer ships with a measured quality bar behind it, not a hope.
5. A thumbs-up/down on the answer lands in `llm.feedback`, building a human-in-the-loop signal for future prompt or model improvement.

### Impact

This is the layer that lets the downstream [dashboard](../4-dashboard/) — and any future
conversational HR feature — ship AI with **guardrails, a quality SLA, and a cost audit
trail** instead of an unmeasured black box. The eval scores are standardised (RAGAS) and
comparable across model versions, so swapping or upgrading the judge or embedding model is
a measurable change, not a leap of faith.

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
  RAGAS eval harness ◄──────────────────────────┘
         │
         ├── llm.eval_results (per-question scores)
         ├── llm.cost_log     (token + cost tracking)
         └── llm.feedback     (analyst thumbs up/down)
                  │
                  ▼
         Airflow DAGs: embedding_refresh (triggered) + nightly_eval (2am)
```

---

## ⭐ Evaluating the RAG pipeline — `make eval`

This module powers a **RAG pipeline** (Retrieval-Augmented Generation). Rather than letting an
LLM answer HR questions from its own training memory, the question is first used to *retrieve* the
relevant (masked) employee records, and the model then answers **grounded in that retrieved
context**. Grounding is what keeps answers specific and current — but it does **not** make them
automatically correct.

**That's why running an evaluation isn't optional here — it's the point of this module.** A RAG
pipeline can fail quietly in two ways: retrieval can pull the *wrong* records, or the model can
*hallucinate* an answer the context doesn't actually support. Neither is visible from eyeballing a
couple of demo questions. Before anyone trusts these answers for a headcount or attrition
decision, you need an **objective, repeatable score** — and a way to keep checking it as models
and data change.

**`make eval` is that check, and it's the heart of this module.** It runs the pipeline against
~200 questions that have known-correct answers, has a **Claude** model act as an impartial judge,
and grades the results on four quality metrics, each gated at **≥ 0.70**:

- *Did the answer stick to the retrieved facts, or make something up?* — **faithfulness**
- *Did it actually answer the question asked?* — **answer_relevancy**
- *Did retrieval surface the right context, and all of it?* — **context_precision / context_recall**

The per-question scores land in `llm.eval_results` and a cost row in `llm.cost_log`. Crucially, the
**nightly Airflow DAG runs this same eval at 2am** and alerts if any metric drops below the gate —
so a regression from a model upgrade or a data change is caught by a pipeline, not by a user
getting a wrong answer. That scheduled gate is what makes this a quality *SLA*, not a one-time
spot check.

```bash
# Needs ANTHROPIC_API_KEY (in ../.env or 2-llm-eval/.env).
# Claude Haiku keeps the run cheap:
RAGAS_LLM_MODEL=claude-haiku-4-5 make eval
```

A representative run scored faithfulness **0.97**, answer_relevancy **0.92**, context_precision
**0.77**, context_recall **0.90** — all above the gate. See [Eval metrics](#eval-metrics) for the
full metric definitions and [Cost tracking](#cost-tracking) for the spend audit trail.

---

## Tech stack

| Concern | Technology |
|---|---|
| Vector DB | pgvector on Postgres 16 (`pgvector/pgvector:pg16`) |
| Embeddings | sentence-transformers (local) / OpenAI text-embedding-3-small |
| Eval framework | RAGAS 0.1.9 |
| Eval LLM judge | Claude via `langchain-anthropic` (`claude-opus-4-8` default) |
| Data validation | Pydantic v2 |
| Orchestration | Apache Airflow 2.9 |
| Testing | pytest + testcontainers |

---

## Setup

### Prerequisites

The shared Postgres (with pgvector) and the `analytics.dim_employees` table must exist
first — the masking view reads from it. From the repo root:

```bash
make infra-up         # Postgres (pgvector/pgvector:pg16) + Trino + Airflow
make ingestion-setup  # seed synthetic HR data
make ingestion-dbt    # build analytics.dim_employees
```

The `llm` schema and its tables are created on first DB init by
`docker/init_llm_schema.sql` (mounted as `02_llm.sql` in the shared compose).

### This module

```bash
cd 2-llm-eval
make setup         # install deps → create masking views → embed safe context into pgvector
make eval          # generate Q&A pairs → run RAGAS eval → write results + cost log
```

`make setup` runs `src.pipeline.setup`: it applies the PII masking views, verifies they
expose no forbidden columns, then embeds every `safe_employee_context` row into
`llm.embeddings`. The default `local` embedding backend runs fully offline.

> **`make eval` uses a Claude LLM judge.** RAGAS scores its metrics with an LLM judge; this
> module uses **Claude** via `langchain-anthropic`, with RAGAS's metric embeddings running on
> the local sentence-transformers model — so the eval path never touches OpenAI. `make eval`
> therefore requires `ANTHROPIC_API_KEY`. The judge model defaults to `claude-opus-4-8` and is
> configurable with `RAGAS_LLM_MODEL` (e.g. `RAGAS_LLM_MODEL=claude-haiku-4-5` to cut cost on
> the 200-question dataset). The embedding and similarity-search paths (`make setup` /
> `make embed`) need no API key.

### Make targets

| Target | What it does |
|---|---|
| `make install` | Install the package + dev dependencies |
| `make setup` | Install, apply masking views, embed safe context |
| `make embed` | Re-embed `safe_employee_context` into pgvector |
| `make eval` | Run the RAGAS eval — Claude judge (needs `ANTHROPIC_API_KEY`) |
| `make test-unit` | Unit tests + coverage (no infra required) |
| `make test-integration` | Integration tests (requires Docker / testcontainers) |
| `make test` | Unit + integration tests |
| `make lint` | Ruff lint over `src/` + `tests/` |
| `make clean` | Remove caches + coverage artifacts |
| `make teardown` | Drop masking views + truncate `llm` tables, then `infra-down` |

The `setup`, `embed`, `eval`, and `teardown` targets source the repo-root `../.env` (Postgres
creds + `ANTHROPIC_API_KEY`) automatically, so you don't have to export it; the test targets
stay hermetic and ignore it. Unit tests mock the heavy ML/DB dependencies and run offline; the
integration tests spin up a real `pgvector/pgvector:pg16` container via testcontainers (Docker
required), and the local-encoder integration test is skipped unless `sentence-transformers` is
installed.

`make teardown` is the inverse of `make setup`: it drops the PII masking views and truncates the
`llm` tables (keeping the tables so a later `make setup` rebuilds cleanly), clears caches, then
shuts the shared Docker stack down via the repo-root `infra-down` (volumes preserved — use
repo-root `make infra-reset` to also wipe them). DB cleanup is skipped if Postgres is already down.

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

## Cost tracking

Every embedding and eval run writes a row to `llm.cost_log` (the local backend records
`cost_usd = 0`, so the offline default is free to run while still producing an audit trail).
Spend by run type over the last 30 days:

```sql
SELECT run_type,
       COUNT(*)              AS runs,
       SUM(embedding_count)  AS embeddings,
       SUM(input_tokens)     AS input_tokens,
       SUM(cost_usd)         AS total_cost_usd
FROM llm.cost_log
WHERE run_at > NOW() - INTERVAL '30 days'
GROUP BY run_type
ORDER BY total_cost_usd DESC;
```

`make eval` calls the Claude judge, a metered API cost. Eval rows in `llm.eval_results` and
`llm.cost_log` record the **judge model** (`RAGAS_LLM_MODEL`, e.g. `claude-haiku-4-5`) so spend
is attributable per model — but token counts aren't wired yet (eval rows log `cost_usd = 0`),
so track actual spend in the Anthropic console, and set `RAGAS_LLM_MODEL=claude-haiku-4-5` to
keep the 200-question run cheap. Switching `EMBEDDING_BACKEND=openai` likewise starts populating
real embedding token counts.

---

## Design decisions

**sentence-transformers over OpenAI by default.** The `all-MiniLM-L6-v2` model runs locally,
costs nothing, has no API dependency, and produces 384-dimensional embeddings that fit comfortably
in a pgvector ivfflat index. OpenAI is available as an opt-in via `EMBEDDING_BACKEND=openai`
for production scenarios where quality matters more than cost.

**RAGAS over custom eval.** RAGAS (Retrieval-Augmented Generation Assessment) is an open-source
framework for evaluating RAG pipelines, using an LLM judge to score generated answers against the
retrieved context on metrics like faithfulness and answer relevancy. It provides standardised, reproducible metrics with published
benchmarks. Rolling your own eval framework is a maintenance liability — RAGAS scores are
comparable across model versions and meaningful to non-engineers reviewing the eval dashboard.

**Claude as the RAGAS judge, local embeddings underneath.** RAGAS needs both an LLM (to grade
faithfulness/relevancy) and an embedding model (for the similarity-based metrics). Out of the
box RAGAS reaches for OpenAI for both. We instead inject a **Claude** judge via
`langchain-anthropic` and keep the embedding side on the local `all-MiniLM-L6-v2` model, so the
eval depends on a single Anthropic key and no OpenAI account. The judge model is configurable
(`RAGAS_LLM_MODEL`), and because Claude Opus 4.8/4.7 reject sampling parameters that RAGAS
forwards by default, the harness wraps the model to drop them — so any Claude model works.

**Feedback separate from eval_results.** Eval results measure model quality against ground truth.
Feedback measures human preference. These are different signals and should be queryable
independently — a model can score high on faithfulness but still get thumbs-down from analysts
who find the phrasing unhelpful.
