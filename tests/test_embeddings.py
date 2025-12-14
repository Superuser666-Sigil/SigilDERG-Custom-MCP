# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
Unit tests for embedding providers and configuration.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest


class TestEmbeddingProviderCreation:
    """Test embedding provider factory."""

    def test_sentence_transformers_provider_creation(self):
        """Test creating sentence-transformers provider."""
        with patch('sigil_mcp.embeddings.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([[0.1] * 384])
            mock_st.return_value = mock_model

            from sigil_mcp.embeddings import create_embedding_provider

            provider = create_embedding_provider(
                provider="sentence-transformers",
                model="all-MiniLM-L6-v2",
                dimension=384
            )

            # Test embedding generation
            result = provider.embed_documents(["test text"])
            assert len(result) == 1
            assert len(result[0]) == 384

            # Test dimension
            assert provider.get_dimension() == 384

    def test_openai_provider_creation(self):
        """Test creating OpenAI provider."""
        with patch('sigil_mcp.embeddings.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1536)]
            mock_client.embeddings.create.return_value = mock_response
            mock_openai.return_value = mock_client

            from sigil_mcp.embeddings import create_embedding_provider

            provider = create_embedding_provider(
                provider="openai",
                model="text-embedding-3-small",
                dimension=1536,
                api_key="sk-test123"
            )

            # Test embedding generation
            result = provider.embed_documents(["test text"])
            assert len(result) == 1
            assert len(result[0]) == 1536

    def test_llamacpp_provider_creation(self, temp_dir):
        """Test creating llama.cpp provider."""
        # Create a fake GGUF file
        fake_model = temp_dir / "test.gguf"
        fake_model.write_text("fake model data")

        with patch('sigil_mcp.llamacpp_provider.Llama') as mock_llama:
            mock_instance = Mock()
            mock_instance.embed.return_value = [0.1] * 4096
            mock_instance.n_ctx.return_value = 2048
            mock_llama.return_value = mock_instance

            from sigil_mcp.embeddings import create_embedding_provider

            provider = create_embedding_provider(
                provider="llamacpp",
                model=str(fake_model),
                dimension=4096,
                n_gpu_layers=0
            )

            # Test embedding generation
            result = provider.embed_documents(["test text"])
            assert len(result) == 1
            assert len(result[0]) == 4096

    def test_unknown_provider_raises_error(self):
        """Test that unknown provider raises ValueError."""
        from sigil_mcp.embeddings import create_embedding_provider

        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedding_provider(
                provider="unknown_provider",
                model="some-model",
                dimension=768
            )

    def test_missing_dependencies_raises_import_error(self):
        """Test that missing dependencies raise ImportError."""
        with patch('sigil_mcp.embeddings.SENTENCE_TRANSFORMERS_AVAILABLE', False):
            from sigil_mcp.embeddings import create_embedding_provider

            with pytest.raises(ImportError, match="sentence-transformers is required"):
                create_embedding_provider(
                    provider="sentence-transformers",
                    model="all-MiniLM-L6-v2",
                    dimension=384
                )


class TestEmbeddingConfiguration:
    """Test embedding configuration properties."""

    def test_embeddings_disabled_by_default(self):
        """Test that embeddings are disabled by default."""
        from pathlib import Path

        from sigil_mcp.config import Config

        # Use non-existent path to force env fallback (default behavior)
        config = Config(Path("/nonexistent/config.json"))
        assert config.embeddings_enabled is False

    def test_embeddings_config_properties(self, temp_dir):
        """Test all embedding config properties."""
        import json

        config_path = temp_dir / "config.json"
        config_data = {
            "embeddings": {
                "enabled": True,
                "provider": "sentence-transformers",
                "model": "all-MiniLM-L6-v2",
                "dimension": 384,
                "cache_dir": "/tmp/embeddings",
                "api_key": "test-key"
            }
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        from sigil_mcp.config import Config
        config = Config(config_path)

        assert config.embeddings_enabled is True
        assert config.embeddings_provider == "sentence-transformers"
        assert config.embeddings_model == "all-MiniLM-L6-v2"
        assert config.embeddings_dimension == 384
        assert config.embeddings_cache_dir == "/tmp/embeddings"
        assert config.embeddings_api_key == "test-key"

    def test_embeddings_api_key_fallback_to_env(self, temp_dir, monkeypatch):
        """Test API key falls back to environment variable."""
        import json

        config_path = temp_dir / "config.json"
        config_data = {
            "embeddings": {
                "enabled": True,
                "provider": "openai"
            }
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")

        from sigil_mcp.config import Config
        config = Config(config_path)

        assert config.embeddings_api_key == "sk-env-key"

    def test_embeddings_kwargs_extraction(self, temp_dir):
        """Test that extra kwargs are extracted correctly."""
        import json

        config_path = temp_dir / "config.json"
        config_data = {
            "embeddings": {
                "enabled": True,
                "provider": "llamacpp",
                "model": "/path/to/model.gguf",
                "dimension": 4096,
                "n_gpu_layers": 35,
                "n_ctx": 2048,
                "use_mlock": True,
                "custom_param": "value"
            }
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        from sigil_mcp.config import Config
        config = Config(config_path)

        kwargs = config.embeddings_kwargs
        assert "n_gpu_layers" in kwargs
        assert kwargs["n_gpu_layers"] == 35
        assert kwargs["context_size"] == 2048
        assert "n_ctx" not in kwargs
        assert kwargs["use_mlock"] is True
        assert kwargs["custom_param"] == "value"

        # Known keys should not be in kwargs
        assert "enabled" not in kwargs
        assert "provider" not in kwargs
        assert "model" not in kwargs

    def test_llamacpp_context_aliases(self, temp_dir):
        """Context size should map from llamacpp_context_size and n_ctx aliases."""
        import json
        cfg_path = temp_dir / "config.json"
        cfg_data = {
            "embeddings": {
                "enabled": True,
                "provider": "llamacpp",
                "model": "/path/to/model.gguf",
                "llamacpp_context_size": 4096,
                "n_ctx": 1234,
            }
        }
        cfg_path.write_text(json.dumps(cfg_data))
        from sigil_mcp.config import Config
        cfg = Config(cfg_path)
        kwargs = cfg.embeddings_kwargs
        # llamacpp_context_size wins, n_ctx becomes a fallback
        assert kwargs["context_size"] == 4096
        assert "n_ctx" not in kwargs


class TestServerEmbeddingIntegration:
    """Test server embedding function creation."""

    def test_embeddings_disabled_returns_none(self, temp_dir):
        """Test that disabled embeddings return None."""
        import importlib
        import json

        from sigil_mcp.config import load_config

        config_path = temp_dir / "config.json"
        config_data = {
            "embeddings": {
                "enabled": False
            }
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        load_config(config_path)

        # Reload server module to pick up new config
        import sigil_mcp.server
        importlib.reload(sigil_mcp.server)

        # Import after reload
        from sigil_mcp.server import _create_embedding_function

        embed_fn, model_name = _create_embedding_function()

        assert embed_fn is None
        assert model_name is None

    def test_missing_provider_returns_none(self, temp_dir):
        """Test that missing provider config returns None."""
        import json

        from sigil_mcp.config import load_config

        config_path = temp_dir / "config.json"
        config_data = {
            "embeddings": {
                "enabled": True
            }
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        load_config(config_path)

        from sigil_mcp.server import _create_embedding_function

        embed_fn, model_name = _create_embedding_function()

        assert embed_fn is None
        assert model_name is None

    def test_missing_model_returns_none(self, temp_dir):
        """Test that missing model config returns None."""
        import json

        from sigil_mcp.config import load_config

        config_path = temp_dir / "config.json"
        config_data = {
            "embeddings": {
                "enabled": True,
                "provider": "sentence-transformers"
            }
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        load_config(config_path)

        from sigil_mcp.server import _create_embedding_function

        embed_fn, model_name = _create_embedding_function()

        assert embed_fn is None
        assert model_name is None

    def test_successful_embedding_creation(self, temp_dir):
        """Test successful embedding function creation."""
        import json

        from sigil_mcp.config import load_config

        config_path = temp_dir / "config.json"
        config_data = {
            "embeddings": {
                "enabled": True,
                "provider": "sentence-transformers",
                "model": "all-MiniLM-L6-v2",
                "dimension": 384
            }
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        test_config = load_config(config_path)

        with patch('sigil_mcp.embeddings.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])
            mock_st.return_value = mock_model

            with patch('sigil_mcp.server.config', test_config):
                from sigil_mcp.server import _create_embedding_function

                embed_fn, model_name = _create_embedding_function()

                assert embed_fn is not None
                assert model_name == "sentence-transformers:all-MiniLM-L6-v2"

                # Test the function works
                result = embed_fn(["text1", "text2"])
                assert isinstance(result, np.ndarray)
                assert result.shape == (2, 384)
                assert result.dtype == np.float32

    def test_import_error_returns_none(self, temp_dir):
        """Test that import errors are handled gracefully."""
        import json

        from sigil_mcp.config import load_config

        config_path = temp_dir / "config.json"
        config_data = {
            "embeddings": {
                "enabled": True,
                "provider": "sentence-transformers",
                "model": "all-MiniLM-L6-v2",
                "dimension": 384
            }
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        load_config(config_path)

        with patch('sigil_mcp.embeddings.create_embedding_provider',
                   side_effect=ImportError("Module not found")):
            from sigil_mcp.server import _create_embedding_function

            embed_fn, model_name = _create_embedding_function()

            assert embed_fn is None
            assert model_name is None

    def test_generic_error_returns_none(self, temp_dir):
        """Test that generic errors are handled gracefully."""
        import json

        from sigil_mcp.config import load_config

        config_path = temp_dir / "config.json"
        config_data = {
            "embeddings": {
                "enabled": True,
                "provider": "sentence-transformers",
                "model": "all-MiniLM-L6-v2",
                "dimension": 384
            }
        }

        with open(config_path, 'w') as f:
            json.dump(config_data, f)

        load_config(config_path)

        with patch('sigil_mcp.embeddings.create_embedding_provider',
                   side_effect=RuntimeError("Initialization failed")):
            from sigil_mcp.server import _create_embedding_function

            embed_fn, model_name = _create_embedding_function()

            assert embed_fn is None
            assert model_name is None


class TestLlamaCppProvider:
    """Test llama.cpp provider specific functionality."""

    def test_llamacpp_not_available(self):
        """Test error when llama-cpp-python not installed."""
        with patch('sigil_mcp.llamacpp_provider.LLAMACPP_AVAILABLE', False):
            from sigil_mcp.llamacpp_provider import LlamaCppEmbeddingProvider

            with pytest.raises(ImportError, match="llama-cpp-python is required"):
                LlamaCppEmbeddingProvider(
                    model_path="/fake/path.gguf",
                    dimension=4096
                )

    def test_llamacpp_missing_model_file(self, temp_dir):
        """Test error when model file doesn't exist."""
        with patch('sigil_mcp.llamacpp_provider.LLAMACPP_AVAILABLE', True):
            with patch('sigil_mcp.llamacpp_provider.Llama'):
                from sigil_mcp.llamacpp_provider import LlamaCppEmbeddingProvider

                with pytest.raises(FileNotFoundError):
                    LlamaCppEmbeddingProvider(
                        model_path=temp_dir / "nonexistent.gguf",
                        dimension=4096
                    )

    def test_llamacpp_text_truncation(self, temp_dir):
        """Test that long texts are truncated properly."""
        fake_model = temp_dir / "test.gguf"
        fake_model.write_text("fake")

        with patch('sigil_mcp.llamacpp_provider.LLAMACPP_AVAILABLE', True):
            with patch('sigil_mcp.llamacpp_provider.Llama') as mock_llama:
                mock_instance = Mock()
                mock_instance.embed.return_value = [0.1] * 4096
                mock_instance.n_ctx.return_value = 100  # Small context
                mock_llama.return_value = mock_instance

                from sigil_mcp.llamacpp_provider import LlamaCppEmbeddingProvider

                provider = LlamaCppEmbeddingProvider(
                    model_path=fake_model,
                    dimension=4096
                )

                # Create text longer than context
                long_text = "x" * 10000

                # Should not raise, should truncate
                result = provider.embed_query(long_text)
                assert len(result) == 4096
