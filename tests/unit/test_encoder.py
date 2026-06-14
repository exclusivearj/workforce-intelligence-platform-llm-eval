"""Unit tests for the embedding encoder (heavy model mocked)."""

from __future__ import annotations

import numpy as np

from src.embeddings import encoder as enc_module
from src.embeddings.encoder import LocalEncoder, OpenAIEncoder, get_encoder


class _FakeModel:
    def encode(self, texts, batch_size=32):
        # Return deterministic 384-dim vectors in [-1, 1].
        return np.array([[0.01 * (i % 100) - 0.5 for i in range(384)] for _ in texts])


def test_local_encoder_returns_384_dims(monkeypatch):
    encoder = LocalEncoder()
    monkeypatch.setattr(encoder, "_load", lambda: _FakeModel())
    out = encoder.encode(["hello"])
    assert len(out) == 1
    assert len(out[0]) == 384
    assert all(-1.0 <= v <= 1.0 for v in out[0])


def test_local_encoder_batch(monkeypatch):
    encoder = LocalEncoder()
    monkeypatch.setattr(encoder, "_load", lambda: _FakeModel())
    out = encoder.encode(["a", "b", "c"])
    assert len(out) == 3


def test_get_encoder_defaults_to_local(monkeypatch):
    monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)
    assert isinstance(get_encoder(), LocalEncoder)


def test_get_encoder_openai(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "openai")
    assert isinstance(get_encoder(), OpenAIEncoder)


def test_dimensions_constant():
    assert enc_module.DIMENSIONS == 384
