"""Airflow plugin placeholder for the llm_eval DAG group."""

from __future__ import annotations

from airflow.plugins_manager import AirflowPlugin


class LLMEvalPlugin(AirflowPlugin):
    name = "llm_eval_plugin"
    macros = []
    operators = []
