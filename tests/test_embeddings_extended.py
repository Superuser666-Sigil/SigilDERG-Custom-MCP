# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import pytest

import sigil_mcp.embeddings as embeddings


def test_create_embedding_unknown_provider():
    with pytest.raises(ValueError):
        embeddings.create_embedding_provider("bad", "model", 128)


def test_sentence_transformers_missing_dep(monkeypatch):
    monkeypatch.setattr(embeddings, "SENTENCE_TRANSFORMERS_AVAILABLE", False)
    monkeypatch.setattr(embeddings, "SentenceTransformer", None)
    with pytest.raises(ImportError):
        embeddings.create_embedding_provider("sentence-transformers", "x", 128)


def test_openai_missing_key(monkeypatch):
    monkeypatch.setattr(embeddings, "OPENAI_AVAILABLE", True)
    class DummyOpenAI:
        def __init__(self, api_key=None): ...
    monkeypatch.setattr(embeddings, "OpenAI", DummyOpenAI)
    with pytest.raises(ValueError):
        embeddings.create_embedding_provider("openai", "text-embedding", 128)


def test_llamacpp_missing_model(monkeypatch, tmp_path):
    fake_path = tmp_path / "missing.gguf"
    with pytest.raises(FileNotFoundError):
        embeddings.create_embedding_provider("llamacpp", str(fake_path), 256)


def test_sentence_transformers_success(monkeypatch):
    class DummyST:
        def __init__(self, model, cache_folder=None): ...
        def encode(self, texts, convert_to_numpy=True):
            import numpy as np
            if isinstance(texts, list):
                return np.array([[0.1] * 3 for _ in texts])
            return np.array([[0.1] * 3])

    monkeypatch.setattr(embeddings, "SENTENCE_TRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(embeddings, "SentenceTransformer", DummyST)
    provider = embeddings.create_embedding_provider("sentence-transformers", "m", 3)
    assert len(provider.embed_documents(["a"])[0]) == 3
