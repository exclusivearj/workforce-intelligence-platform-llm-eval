"""Integration test: end-to-end eval assembly with a local encoder.

Skipped unless sentence-transformers is installed (heavy, optional).
Run with: ``pytest tests/integration/ -m integration``
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


def test_local_encoder_produces_384_dims():
    pytest.importorskip("sentence_transformers")
    from src.embeddings.encoder import LocalEncoder

    out = LocalEncoder().encode(["how many engineers are in the data department"])
    assert len(out) == 1
    assert len(out[0]) == 384
