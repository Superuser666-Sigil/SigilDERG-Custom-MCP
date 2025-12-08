# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""Llama.cpp embedding provider for local LLM embeddings."""

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False
    Llama = None  # type: ignore


class LlamaCppEmbeddingProvider:
    """Embedding provider using llama.cpp for local LLM embeddings.
    
    This provider uses llama.cpp's Python bindings to generate embeddings
    from a local Llama model (e.g., Meta Llama 3.1 8B Instruct).
    """

    def __init__(
        self,
        model_path: str | Path,
        dimension: int = 4096,
        context_size: int = 2048,
        n_gpu_layers: int = 0,
        use_mlock: bool = False,
        embedding: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize llama.cpp embedding provider.
        
        Args:
            model_path: Path to the GGUF model file
            dimension: Expected embedding dimension (default: 4096 for Llama 3.1 8B)
            context_size: Context window size (default: 2048)
            n_gpu_layers: Number of layers to offload to GPU (0 = CPU only)
            use_mlock: Lock model in RAM to prevent swapping
            embedding: Enable embedding mode
            **kwargs: Additional arguments passed to Llama constructor
        """
        if not LLAMACPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is required for LlamaCppEmbeddingProvider. "
                "Install it with: pip install llama-cpp-python"
            )
        
        if Llama is None:
            raise ImportError("Llama class not available")

        self.model_path = Path(model_path).expanduser()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.dimension = dimension
        logger.info(f"Loading llama.cpp model from {self.model_path}...")
        logger.info(f"GPU layers: {n_gpu_layers}, Context: {context_size}")

        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=context_size,
            n_gpu_layers=n_gpu_layers,
            use_mlock=use_mlock,
            embedding=embedding,
            verbose=False,
            **kwargs,
        )

        logger.info("Llama.cpp model loaded successfully")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for i, text in enumerate(texts):
            # Truncate text to avoid context overflow
            # Reserve some tokens for prompt formatting
            max_chars = self.llm.n_ctx() * 3  # Rough estimate: ~3 chars per token
            if len(text) > max_chars:
                text = text[:max_chars]
                logger.debug(f"Truncated text {i+1}/{len(texts)} to {max_chars} characters")

            # Generate embedding
            result = self.llm.embed(text)
            # Handle different return types from llama.cpp
            if isinstance(result, tuple):
                embeddings.append(result[0])  # type: ignore
            else:
                embeddings.append(result)  # type: ignore
            
            if (i + 1) % 10 == 0:
                logger.info(f"Generated embeddings for {i+1}/{len(texts)} documents")

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        # Truncate if needed
        max_chars = self.llm.n_ctx() * 3
        if len(text) > max_chars:
            text = text[:max_chars]
            logger.debug(f"Truncated query to {max_chars} characters")

        result = self.llm.embed(text)
        # Handle different return types from llama.cpp
        if isinstance(result, tuple):
            return result[0]  # type: ignore
        return result  # type: ignore

    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimension

    def __del__(self) -> None:
        """Cleanup llama.cpp resources."""
        if hasattr(self, 'llm'):
            del self.llm
