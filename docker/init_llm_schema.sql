-- ============================================================================
-- llm-eval :: extends the shared `workforce` database with the `llm` schema.
-- Runs against the SAME Postgres instance created by 1-ingestion (not a new DB).
-- Mounted as docker-entrypoint-initdb.d/02_llm.sql and idempotent.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS llm;

-- 1. Embeddings ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS llm.embeddings (
    id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    source_table    VARCHAR(100) NOT NULL,        -- e.g. 'analytics.dim_employees'
    source_row_id   UUID NOT NULL,
    content_text    TEXT NOT NULL,
    embedding       vector(384),                  -- all-MiniLM-L6-v2
    model_name      VARCHAR(100) NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    refreshed_at    TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (source_table, source_row_id, model_name)
);
CREATE INDEX IF NOT EXISTS idx_embeddings_vector
    ON llm.embeddings USING ivfflat (embedding vector_cosine_ops);

-- 2. Eval results -------------------------------------------------------------
CREATE TABLE IF NOT EXISTS llm.eval_results (
    id                 UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    run_id             UUID NOT NULL,
    question           TEXT NOT NULL,
    ground_truth       TEXT NOT NULL,
    generated_answer   TEXT NOT NULL,
    retrieved_contexts TEXT[] NOT NULL,
    faithfulness       NUMERIC(5, 4),
    answer_relevancy   NUMERIC(5, 4),
    context_precision  NUMERIC(5, 4),
    context_recall     NUMERIC(5, 4),
    model_name         VARCHAR(100) NOT NULL,
    evaluated_at       TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Feedback -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS llm.feedback (
    id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    eval_result_id  UUID REFERENCES llm.eval_results(id),
    analyst_role    VARCHAR(100) NOT NULL,        -- 'hr_partner' | 'recruiter' | 'legal'
    rating          SMALLINT NOT NULL CHECK (rating IN (1, -1)),
    correction_text TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Cost log -----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS llm.cost_log (
    id              UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    run_type        VARCHAR(50) NOT NULL,         -- 'embedding' | 'completion' | 'eval'
    model_name      VARCHAR(100) NOT NULL,
    input_tokens    INTEGER,
    output_tokens   INTEGER,
    embedding_count INTEGER,
    cost_usd        NUMERIC(10, 6),
    run_at          TIMESTAMPTZ DEFAULT NOW()
);

-- 5. PII-safe context view.
-- NOTE: the llm.safe_employee_context view depends on analytics.dim_employees,
-- which is built by dbt AFTER this file runs at container init. Creating it here
-- would fail on a fresh database. It is instead created (idempotently) by
-- src.masking.views.apply_masking_views(), invoked during `make llm-eval-setup`
-- once the dbt models exist. This is the only approved LLM context data source.
