# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

import pytest

import sigil_mcp.llamacpp_provider as lcpp


def test_llamacpp_import_error(monkeypatch):
    monkeypatch.setattr(lcpp, "LLAMACPP_AVAILABLE", False)
    monkeypatch.setattr(lcpp, "Llama", None)
    with pytest.raises(ImportError):
        lcpp.LlamaCppEmbeddingProvider("/tmp/missing.gguf")


def test_llamacpp_success(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.write_text("fake")

    class DummyLlama:
        def __init__(self, *a, **k): ...
        def embed(self, text):
            return [0.1] * 4
        def n_ctx(self):
            return 4

    monkeypatch.setattr(lcpp, "LLAMACPP_AVAILABLE", True)
    monkeypatch.setattr(lcpp, "Llama", DummyLlama)

    provider = lcpp.LlamaCppEmbeddingProvider(str(model), dimension=4, context_size=4, n_gpu_layers=0)
    docs = provider.embed_documents(["abc"])
    assert len(docs[0]) == 4
    q = provider.embed_query("abc")
    assert len(q) == 4
