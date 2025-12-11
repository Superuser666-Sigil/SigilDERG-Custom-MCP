# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""Llama.cpp embedding provider for local LLM embeddings."""

import logging
from pathlib import Path
from typing import Any, List, Sequence

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    Llama = None  # type: ignore
    LLAMACPP_AVAILABLE = False


class LlamaCppEmbeddingProvider:
    """Embedding provider using llama.cpp for local LLM embeddings.
    
    This provider uses llama.cpp's Python bindings to generate embeddings
    from a local Llama model (e.g., Meta Llama 3.1 8B Instruct).
    """

    def __init__(
        self,
        model_path: str | Path,
        dimension: int = 4096,
        context_size: int = 8192,
        n_gpu_layers: int = 999,
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
        if not LLAMACPP_AVAILABLE or Llama is None:
            logger.error(
                "llama-cpp-python is required for LlamaCppEmbeddingProvider. "
                "Install it with: pip install llama-cpp-python or pip install .[embeddings-llamacpp-cpu]"
            )
            raise ImportError(
                "llama-cpp-python is required for LlamaCppEmbeddingProvider. "
                "Install it with: pip install llama-cpp-python"
            )

        self.model_path = Path(model_path).expanduser()
        if not self.model_path.exists():
            logger.error(
                "Llama.cpp model not found at %s. "
                "Place the Jina GGUF (or configured model) under ./models "
                "or update embeddings.model.",
                self.model_path,
            )
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

    def _combine_embeddings(self, vectors: Sequence[Sequence[float]]) -> list[float]:
        """Average multiple embedding vectors to keep output size consistent."""

        if not vectors:
            return []
        vectors_list = [list(v) for v in vectors]
        length = len(vectors_list[0])
        totals = [0.0] * length
        for vec in vectors_list:
            if len(vec) != length:
                logger.warning(
                    "Embedding length mismatch while combining segments; "
                    "skipping inconsistent vector"
                )
                continue
            for idx, val in enumerate(vec):
                totals[idx] += float(val)

        count = max(1, len(vectors_list))
        return [val / count for val in totals]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        for i, text in enumerate(texts):
            max_chars = self.llm.n_ctx() * 3  # Rough estimate: ~3 chars per token
            segments = [text[j : j + max_chars] for j in range(0, len(text), max_chars)] or [text]
            segment_vectors: List[list[float]] = []
            for segment in segments:
                try:
                    result = self.llm.embed(segment)
                except Exception:
                    logger.exception("Failed to embed segment %s/%s", i + 1, len(texts))
                    continue
                if isinstance(result, tuple):
                    segment_vectors.append(list(result[0]))  # type: ignore
                else:
                    segment_vectors.append(list(result))  # type: ignore

            if not segment_vectors:
                logger.warning("No embeddings produced for document %s", i + 1)
                embeddings.append([])
            elif len(segment_vectors) == 1:
                embeddings.append(segment_vectors[0])
            else:
                logger.debug(
                    "Combined %s embedding segments for document %s",
                    len(segment_vectors),
                    i + 1,
                )
                embeddings.append(self._combine_embeddings(segment_vectors))
            
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
        max_chars = self.llm.n_ctx() * 3
        segments = [text[j : j + max_chars] for j in range(0, len(text), max_chars)] or [text]
        vectors: List[list[float]] = []
        for segment in segments:
            result = self.llm.embed(segment)
            if isinstance(result, tuple):
                vectors.append(list(result[0]))  # type: ignore
            else:
                vectors.append(list(result))  # type: ignore

        if not vectors:
            return []

        if len(vectors) == 1:
            return vectors[0]

        logger.debug("Combined %s query segments into averaged embedding", len(vectors))
        return self._combine_embeddings(vectors)

    def get_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.dimension

    def __del__(self) -> None:
        """Cleanup llama.cpp resources."""
        if hasattr(self, 'llm'):
            del self.llm
