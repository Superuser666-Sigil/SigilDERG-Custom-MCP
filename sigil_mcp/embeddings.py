# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""Embedding providers for semantic search."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

logger = logging.getLogger(__name__)

# Optional dependencies - import at top level
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None  # type: ignore
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None  # type: ignore
    OPENAI_AVAILABLE = False

if TYPE_CHECKING:
    from openai import OpenAI as OpenAIType
    from sentence_transformers import SentenceTransformer as SentenceTransformerType


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query."""
        ...

    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        ...


def create_embedding_provider(  # noqa: C901
    provider: str,
    model: str,
    dimension: int,
    **kwargs: Any,
) -> EmbeddingProvider:
    """Create an embedding provider based on configuration.

    Args:
        provider: Provider name ('sentence-transformers', 'openai', or 'llamacpp')
        model: Model name or path
        dimension: Expected embedding dimension
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured embedding provider

    Raises:
        ValueError: If provider is unknown or configuration is invalid
        ImportError: If required dependencies are not installed
    """
    try:
        if provider == "sentence-transformers":
            st_available = SENTENCE_TRANSFORMERS_AVAILABLE or SentenceTransformer is not None
            if not st_available:
                raise ImportError(
                    "sentence-transformers is required for this provider. "
                    "Install it with: pip install sentence-transformers"
                )

            cache_dir = kwargs.get("cache_dir")
            logger.info(f"Loading sentence-transformers model: {model}")
            # Pylance/static analysis can't infer that SentenceTransformer is non-None
            # after the ImportError guard. Explicitly assert to help type-checkers.
            assert SentenceTransformer is not None, "sentence-transformers not installed"
            model_obj = SentenceTransformer(model, cache_folder=cache_dir)

            class STProvider:
                def __init__(self, model: "SentenceTransformerType", dim: int) -> None:
                    self.model = model
                    self.dimension = dim

                def embed_documents(self, texts: list[str]) -> list[list[float]]:
                    embeddings = self.model.encode(texts, convert_to_numpy=True)
                    return embeddings.tolist()

                def embed_query(self, text: str) -> list[float]:
                    embedding = self.model.encode([text], convert_to_numpy=True)
                    return embedding[0].tolist()

                def get_dimension(self) -> int:
                    return self.dimension

            return STProvider(model_obj, dimension)

        elif provider == "openai":
            openai_available = OPENAI_AVAILABLE or OpenAI is not None
            if not openai_available:
                raise ImportError(
                    "openai is required for this provider. "
                    "Install it with: pip install openai"
                )

            api_key = kwargs.get("api_key")
            if not api_key:
                raise ValueError("OpenAI API key is required")

            # Pylance/static analysis can't infer OpenAI is non-None after the guard.
            assert OpenAI is not None, "openai package not installed"
            client = OpenAI(api_key=api_key)

            class OpenAIProvider:
                def __init__(
                    self, client: "OpenAIType", model: str, dim: int
                ) -> None:
                    self.client = client
                    self.model = model
                    self.dimension = dim

                def embed_documents(self, texts: list[str]) -> list[list[float]]:
                    response = self.client.embeddings.create(
                        input=texts, model=self.model
                    )
                    return [item.embedding for item in response.data]

                def embed_query(self, text: str) -> list[float]:
                    response = self.client.embeddings.create(
                        input=[text], model=self.model
                    )
                    return response.data[0].embedding

                def get_dimension(self) -> int:
                    return self.dimension

            return OpenAIProvider(client, model, dimension)

        elif provider == "llamacpp":
            from .llamacpp_provider import LlamaCppEmbeddingProvider

            # model is the path to the GGUF file
            model_path = Path(model).expanduser()
            return LlamaCppEmbeddingProvider(
                model_path=model_path,
                dimension=dimension,
                **kwargs,
            )

        else:
            raise ValueError(
                f"Unknown embedding provider: {provider}. "
                "Supported providers: sentence-transformers, openai, llamacpp"
            )
    except ImportError as exc:
        logger.error(
            "Embedding provider '%s' missing dependencies: %s. "
            "Install the appropriate optional extras (e.g., pip install .[embeddings-llamacpp-cpu] "
            "or .[embeddings-sentencetransformers]).",
            provider,
            exc,
        )
        raise
    except FileNotFoundError as exc:
        logger.error(
            "Embedding model file not found: %s. Place the model under ./models or "
            "set embeddings.model to a valid path. Default Jina GGUF instructions are "
            "documented in README.",
            exc,
        )
        raise


__all__ = ["EmbeddingProvider", "create_embedding_provider"]
