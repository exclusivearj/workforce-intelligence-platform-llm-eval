"""Embedding encoders.

Two backends, chosen by EMBEDDING_BACKEND:
- 'local'  (default): sentence-transformers/all-MiniLM-L6-v2, 384-dim, zero cost.
- 'openai': text-embedding-3-small truncated to 384 dims (tracks token usage).

Heavy libraries (sentence_transformers, openai) are imported lazily so this module
is importable in lightweight/offline environments and easy to unit test with mocks.
"""

from __future__ import annotations

import os

DIMENSIONS = 384


class LocalEncoder:
    model_name = "all-MiniLM-L6-v2"
    dimensions = DIMENSIONS

    def __init__(self) -> None:
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(texts, batch_size=32)
        return [list(map(float, v)) for v in vectors]


class OpenAIEncoder:
    model_name = "text-embedding-3-small"
    dimensions = DIMENSIONS

    def __init__(self) -> None:
        self.last_token_count = 0

    def encode(self, texts: list[str]) -> list[list[float]]:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.embeddings.create(
            model=self.model_name, input=texts, dimensions=self.dimensions
        )
        self.last_token_count = getattr(resp.usage, "total_tokens", 0)
        return [list(item.embedding) for item in resp.data]


def get_encoder() -> LocalEncoder | OpenAIEncoder:
    backend = os.getenv("EMBEDDING_BACKEND", "local").lower()
    if backend == "openai":
        return OpenAIEncoder()
    return LocalEncoder()
