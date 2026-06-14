# TASKS.md — llm-eval/

> Read `../TASKS.md` first for platform-wide rules.
> Requires `ingestion/` to be fully set up and running before starting this module.

---

## What this project builds

The data infrastructure layer for LLM-powered People Analytics tooling:

1. A pgvector embedding store for synthetic HR Q&A knowledge
2. A RAGAS-based evaluation harness that scores LLM response quality
3. A PII masking view layer so sensitive employee data never enters LLM context windows
4. A feedback loop table for analyst thumbs-up/down corrections
5. Automated cost tracking for all LLM/embedding API calls
6. An Airflow DAG group that refreshes embeddings nightly and runs eval on a schedule

This directly addresses the Airbnb JD requirement:
> "Build, update, and maintain a production-grade data foundation that supports AI initiatives —
> including pipelines that feed LLM-powered tools, evaluation and feedback datasets, and the
> access controls and data models required to responsibly scale AI products from prototype to production."

---

## Directory structure

```
llm-eval/
├── TASKS.md                        ← this file
├── README.md
├── Makefile
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── encoder.py              ← sentence-transformers (local) or OpenAI encoder
│   │   ├── store.py                ← pgvector read/write operations
│   │   └── pipeline.py             ← orchestrates encode → store
│   ├── eval/
│   │   ├── __init__.py
│   │   ├── qa_generator.py         ← synthetic HR Q&A dataset generator
│   │   ├── ragas_harness.py        ← RAGAS eval runner
│   │   └── metrics_writer.py       ← writes eval results to Postgres
│   ├── masking/
│   │   ├── __init__.py
│   │   └── views.py                ← creates/refreshes PII masking views
│   └── feedback/
│       ├── __init__.py
│       └── store.py                ← feedback table read/write
├── airflow/
│   ├── dags/
│   │   ├── llm_eval_embedding_refresh.py
│   │   └── llm_eval_nightly.py
│   └── plugins/
│       └── llm_eval_plugin.py
├── docker/
│   └── init_llm_schema.sql         ← extends shared Postgres with llm schema
├── notebooks/
│   └── eval_exploration.ipynb      ← interactive eval result analysis
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_encoder.py
│   │   ├── test_qa_generator.py
│   │   ├── test_masking.py
│   │   └── test_ragas_harness.py
│   └── integration/
│       ├── test_pgvector_store.py
│       └── test_eval_pipeline.py
└── .github/
    └── workflows/
        └── llm-eval-ci.yml
```

---

## Implementation tasks

### Task 2.0 — llm schema DDL (`docker/init_llm_schema.sql`)

Run this against the shared Postgres instance (not a new DB — same `workforce` DB from ingestion).

```sql
-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Embeddings table
CREATE TABLE IF NOT EXISTS llm.embeddings (
    id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    source_table    VARCHAR(100) NOT NULL,    -- e.g. 'analytics.dim_employees'
    source_row_id   UUID NOT NULL,
    content_text    TEXT NOT NULL,            -- the text that was embedded
    embedding       vector(384),              -- 384-dim for sentence-transformers/all-MiniLM-L6-v2
    model_name      VARCHAR(100) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    refreshed_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (source_table, source_row_id, model_name)
);
CREATE INDEX idx_embeddings_vector ON llm.embeddings USING ivfflat (embedding vector_cosine_ops);

-- 2. Eval results table
CREATE TABLE IF NOT EXISTS llm.eval_results (
    id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    run_id          UUID NOT NULL,
    question        TEXT NOT NULL,
    ground_truth    TEXT NOT NULL,
    generated_answer TEXT NOT NULL,
    retrieved_contexts TEXT[] NOT NULL,
    faithfulness    NUMERIC(5,4),
    answer_relevancy NUMERIC(5,4),
    context_precision NUMERIC(5,4),
    context_recall  NUMERIC(5,4),
    model_name      VARCHAR(100) NOT NULL,
    evaluated_at    TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Feedback table
CREATE TABLE IF NOT EXISTS llm.feedback (
    id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    eval_result_id  UUID REFERENCES llm.eval_results(id),
    analyst_role    VARCHAR(100) NOT NULL,    -- 'hr_partner' | 'recruiter' | 'legal'
    rating          SMALLINT NOT NULL         -- 1 (thumbs up) | -1 (thumbs down)
        CHECK (rating IN (1, -1)),
    correction_text TEXT,                     -- optional correction if thumbs down
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Cost tracking table
CREATE TABLE IF NOT EXISTS llm.cost_log (
    id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    run_type        VARCHAR(50) NOT NULL,     -- 'embedding' | 'completion' | 'eval'
    model_name      VARCHAR(100) NOT NULL,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    embedding_count INTEGER,
    cost_usd        NUMERIC(10,6),
    run_at          TIMESTAMPTZ DEFAULT NOW()
);

-- 5. PII-safe context view (used by LLM pipelines — never exposes salary/perf)
CREATE OR REPLACE VIEW llm.safe_employee_context AS
SELECT
    employee_id,
    department,
    job_title,
    level,
    hire_date,
    is_active,
    employment_type,
    location,
    -- salary: EXCLUDED
    -- performance_rating: EXCLUDED
    -- manager_id: hashed
    MD5(COALESCE(manager_id::text, '')) AS manager_id_hashed
FROM analytics.dim_employees;
-- Grant read to the LLM pipeline role only
GRANT SELECT ON llm.safe_employee_context TO dbt_transformer;
```

**Critical**: Document the `safe_employee_context` view as the only approved data source
for LLM context windows. Any query that bypasses this view to access salary/performance
fields must be rejected by the governance audit trigger (implemented in Project 3).

---

### Task 2.1 — Embedding encoder (`src/embeddings/encoder.py`)

Support two backends, selected by environment variable `EMBEDDING_BACKEND`:

**Local (default, zero cost)**: `sentence-transformers/all-MiniLM-L6-v2`
```python
from sentence_transformers import SentenceTransformer

class LocalEncoder:
    model_name = "all-MiniLM-L6-v2"
    dimensions = 384

    def encode(self, texts: list[str]) -> list[list[float]]:
        model = SentenceTransformer(self.model_name)
        return model.encode(texts, batch_size=32).tolist()
```

**OpenAI (optional)**: `text-embedding-3-small`
```python
class OpenAIEncoder:
    model_name = "text-embedding-3-small"
    dimensions = 384    # truncated via dimensions parameter

    def encode(self, texts: list[str]) -> list[list[float]]:
        # Use openai.embeddings.create with dimensions=384
        # Return list of embedding vectors
        # Track token count for cost_log
```

Factory function:
```python
def get_encoder() -> LocalEncoder | OpenAIEncoder:
    backend = os.getenv("EMBEDDING_BACKEND", "local")
    if backend == "openai":
        return OpenAIEncoder()
    return LocalEncoder()
```

---

### Task 2.2 — pgvector store (`src/embeddings/store.py`)

```python
def upsert_embeddings(
    conn,
    records: list[EmbeddingRecord],
    model_name: str,
) -> int:
    """
    Upsert embedding vectors into llm.embeddings.
    Use INSERT ... ON CONFLICT (source_table, source_row_id, model_name) DO UPDATE.
    Use psycopg2 with the pgvector register_vector() call.
    Return count of rows upserted.
    """

def similarity_search(
    conn,
    query_vector: list[float],
    top_k: int = 5,
    source_table: str = "analytics.dim_employees",
) -> list[SimilarityResult]:
    """
    SELECT ... ORDER BY embedding <=> %(query)s::vector LIMIT %(k)s
    Return list of SimilarityResult(source_row_id, content_text, score).
    """
```

Register the pgvector type adapter:
```python
from pgvector.psycopg2 import register_vector
register_vector(conn)
```

---

### Task 2.3 — Synthetic HR Q&A generator (`src/eval/qa_generator.py`)

Generate 200 question/answer pairs covering realistic People Analytics queries.
Use a template + faker approach — no LLM needed to generate Q&A pairs.

Question categories (25 per category, 8 categories = 200 total):
1. **Headcount queries** — "How many engineers are in the [dept] department?"
2. **Attrition queries** — "What was the attrition rate in [dept] last quarter?"
3. **Recruiting funnel** — "What is the average time to hire for [job_title] roles?"
4. **Level distribution** — "How many IC3s are in [dept]?"
5. **Tenure queries** — "What is the average tenure in [dept]?"
6. **Location queries** — "How many employees are based in [location]?"
7. **Manager queries** — "How many direct reports does a typical M2 manager have?"
8. **Comparison queries** — "Which department has the highest attrition rate?"

For each Q&A pair, store:
```python
@dataclass
class QAPair:
    question_id: str
    question: str
    ground_truth: str          # the correct answer (derived from synthetic data)
    context_employee_ids: list[str]  # which dim_employees rows are relevant
    category: str
    difficulty: str            # 'easy' | 'medium' | 'hard'
```

The ground truth answers must be derivable from the actual data in `analytics.dim_employees`.
The generator should query Postgres to produce factually correct answers.

---

### Task 2.4 — RAGAS eval harness (`src/eval/ragas_harness.py`)

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

def run_eval(
    qa_pairs: list[QAPair],
    retrieval_fn: Callable[[str], list[str]],  # returns context strings for a question
    generation_fn: Callable[[str, list[str]], str],  # generates answer given question + contexts
) -> EvalRun:
    """
    Build a RAGAS Dataset from qa_pairs, run evaluate(), return EvalRun with all scores.
    """

@dataclass
class EvalRun:
    run_id: UUID
    scores: dict[str, float]     # metric_name -> mean score
    row_scores: list[dict]       # per-question scores
    model_name: str
    evaluated_at: datetime
```

Default `generation_fn`: a simple retrieval-augmented prompt using the local Postgres
query (no OpenAI needed for basic eval). For OpenAI eval, swap the `generation_fn`.

The `retrieval_fn` calls `similarity_search()` from `store.py` and returns the top-3
`content_text` values.

---

### Task 2.5 — Metrics writer (`src/eval/metrics_writer.py`)

```python
def write_eval_results(conn, eval_run: EvalRun, qa_pairs: list[QAPair]) -> None:
    """Write per-question scores to llm.eval_results."""

def write_cost_log(conn, run_type: str, model_name: str, **kwargs) -> None:
    """Write a cost tracking row to llm.cost_log."""

def get_latest_mean_scores(conn) -> dict[str, float]:
    """Return the most recent run's mean scores for alerting."""
```

---

### Task 2.6 — PII masking views (`src/masking/views.py`)

```python
def apply_masking_views(conn) -> None:
    """
    Execute the CREATE OR REPLACE VIEW statements for all PII-sensitive tables.
    Called during setup and whenever the data classification config changes.
    Views created:
    - llm.safe_employee_context (already in init SQL — idempotent here)
    - analytics.v_employees_safe (same content, accessible to analyst_reader)
    """

def verify_masking(conn) -> list[MaskingVerificationResult]:
    """
    Query each masked view and assert that:
    - salary column is not present
    - performance_rating column is not present
    - manager_id is hashed (not raw UUID)
    Return list of verification results.
    """
```

---

### Task 2.7 — Feedback store (`src/feedback/store.py`)

```python
def record_feedback(
    conn,
    eval_result_id: UUID,
    analyst_role: str,
    rating: int,              # 1 or -1
    correction_text: str | None = None,
) -> UUID:
    """Insert a feedback row. Return the new feedback id."""

def get_feedback_summary(conn, days: int = 30) -> FeedbackSummary:
    """
    Return aggregate feedback stats:
    - total_ratings, positive_count, negative_count, correction_rate_pct
    - breakdown by analyst_role
    - top 5 questions with lowest ratings (candidates for retraining)
    """
```

---

### Task 2.8 — Airflow DAGs

**`airflow/dags/llm_eval_embedding_refresh.py`**:
```python
@dag(
    dag_id="llm_eval_embedding_refresh",
    schedule=None,              # triggered by hr_ingestion DAG only
    tags=["llm-eval", "people-analytics"],
)
def llm_eval_embedding_refresh():
    @task
    def check_source_freshness() -> bool:
        """Query analytics.dim_employees updated_at. Skip if no changes since last run."""

    @task
    def refresh_embeddings(has_changes: bool) -> dict:
        """
        If has_changes: re-embed all dim_employees rows using safe_employee_context view.
        Write to llm.embeddings via upsert.
        Return stats: rows_embedded, cost_usd.
        """

    @task
    def write_cost_log(stats: dict) -> None:
        """Write embedding cost to llm.cost_log."""

    check = check_source_freshness()
    embed = refresh_embeddings(check)
    write_cost_log(embed)
```

**`airflow/dags/llm_eval_nightly.py`**:
```python
@dag(
    dag_id="llm_eval_nightly",
    schedule="0 2 * * *",
    tags=["llm-eval", "people-analytics"],
)
def llm_eval_nightly():
    @task
    def load_qa_pairs() -> list:
        """Load Q&A pairs from Postgres or regenerate if stale."""

    @task
    def run_ragas_eval(qa_pairs: list) -> dict:
        """Run eval harness. Return scores dict."""

    @task
    def write_results(scores: dict) -> None:
        """Write to llm.eval_results via metrics_writer."""

    @task
    def alert_if_scores_drop(scores: dict) -> None:
        """
        Compare with previous run's scores.
        If faithfulness < 0.7 OR answer_relevancy < 0.7: send Slack alert.
        """

    qa = load_qa_pairs()
    scores = run_ragas_eval(qa)
    write_results(scores)
    alert_if_scores_drop(scores)
```

---

### Task 2.9 — Tests

**`tests/unit/test_qa_generator.py`**:
- Assert 200 Q&A pairs are generated
- Assert each category has 25 pairs
- Assert ground_truth is non-empty for all pairs
- Assert question_id is unique across all pairs

**`tests/unit/test_masking.py`**:
- Mock a DB connection
- Assert `verify_masking()` raises `MaskingViolation` if salary column is present in view output
- Assert `verify_masking()` passes when salary is absent

**`tests/unit/test_encoder.py`**:
- Assert `LocalEncoder.encode(["test"])` returns a list of length 1
- Assert each embedding has 384 dimensions
- Assert all values are floats between -1 and 1

**`tests/integration/test_pgvector_store.py`**:
- Requires Postgres with pgvector extension
- Insert 5 embeddings via `upsert_embeddings()`
- Run `similarity_search()` and assert top result is the most similar embedding
- Run upsert again — assert row count unchanged (idempotency)

---

### Task 2.10 — README.md

Include:
1. Purpose statement (one paragraph)
2. ASCII diagram showing: Q&A dataset → encoder → pgvector → RAGAS harness → feedback table
3. Tech stack table
4. Setup instructions
5. "Data readiness for AI" section explaining the PII masking design
6. Eval metrics glossary (faithfulness, answer_relevancy, context_precision, context_recall — one line each)
7. Cost tracking section with example cost_log query
8. Design decisions:
   - Why sentence-transformers over OpenAI by default (zero cost, no API dependency, runs offline)
   - Why RAGAS over custom eval (standardised metrics, reproducible, dataset-agnostic)
   - Why feedback is stored separately from eval_results (separation of model quality vs human preference)

---

### Task 2.11 — pyproject.toml

```toml
[project]
name = "wip-llm-eval"
dependencies = [
    "psycopg2-binary>=2.9",
    "pgvector>=0.3",
    "sentence-transformers>=3.0",
    "ragas>=0.1.9",
    "datasets>=2.19",
    "openai>=1.30",          # optional — only used if EMBEDDING_BACKEND=openai
    "pydantic>=2.7",
    "faker>=25.0",
    "python-dotenv>=1.0",
    "apache-airflow>=2.9.1",
    "langchain-community>=0.2",  # RAGAS dependency
]
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "pytest-mock>=3.14",
    "testcontainers[postgres]>=4.5",
    "ruff>=0.4",
    "jupyter>=1.0",
]
```

---

## Acceptance criteria

- [ ] `pgvector` extension enabled on shared Postgres instance
- [ ] `llm.embeddings` contains ≥ 500 rows after `make setup`
- [ ] `similarity_search("how many engineers are in the data department")` returns relevant rows
- [ ] `llm.safe_employee_context` view exists and contains no salary/performance columns
- [ ] RAGAS eval run completes and writes to `llm.eval_results`
- [ ] `make test-unit` passes with ≥ 80% coverage
- [ ] Both Airflow DAGs visible in UI and manually triggerable without errors
- [ ] `llm.cost_log` has at least one row after a full pipeline run
