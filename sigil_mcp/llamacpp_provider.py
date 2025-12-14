# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""Llama.cpp embedding provider for local LLM embeddings."""

import logging
from pathlib import Path
from typing import Any, List, Sequence
from time import perf_counter

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    Llama = None  # type: ignore
    LLAMACPP_AVAILABLE = False

# Track whether we've installed a llama.cpp log filter to suppress noisy warnings
_llama_log_filter_installed = False


def _install_llama_log_filter() -> None:
    """Suppress noisy llama.cpp warnings while preserving important logs."""
    global _llama_log_filter_installed
    if _llama_log_filter_installed:
        return
    try:
        # Newer bindings expose llama_cpp as a submodule with llama_log_set
        from llama_cpp import llama_cpp as _llama_cpp  # type: ignore
    except Exception:
        return

    def _log_cb(level, msg, *args):
        try:
            text = msg.decode() if isinstance(msg, (bytes, bytearray)) else str(msg)
        except Exception:
            text = str(msg)
        normalized = text.strip().lower()
        # Drop the frequent embedding output warning noise
        if "embeddings required but some input tokens were not marked as outputs" in normalized:
            return
        if level <= 2:
            logger.error(text.strip())
        elif level == 3:
            logger.warning(text.strip())
        else:
            logger.info(text.strip())

    try:
        _llama_cpp.llama_log_set(_log_cb)
        _llama_log_filter_installed = True
    except Exception:
        logger.debug("Failed to install llama.cpp log filter", exc_info=True)


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
        n_threads: int | None = None,
        batch_size: int | None = None,
        n_threads_batch: int | None = None,
        llama_n_batch: int | None = None,
        llama_n_ubatch: int | None = None,
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
        # Determine thread/batching defaults
        from .config import get_config
        import os as _os

        cfg = get_config()
        cores = _os.cpu_count() or 1
        env_threads = _os.getenv("SIGIL_MCP_LLAMA_THREADS") or _os.getenv("SIGIL_MCP_LLAMACPP_THREADS")
        env_batch = _os.getenv("SIGIL_MCP_LLAMA_BATCH")
        env_threads_batch = _os.getenv("SIGIL_MCP_LLAMA_THREADS_BATCH") or _os.getenv("SIGIL_MCP_LLAMACPP_THREADS_BATCH")
        env_n_batch = _os.getenv("SIGIL_MCP_LLAMA_N_BATCH") or _os.getenv("SIGIL_MCP_LLAMACPP_N_BATCH")
        env_n_ubatch = _os.getenv("SIGIL_MCP_LLAMA_N_UBATCH") or _os.getenv("SIGIL_MCP_LLAMACPP_N_UBATCH")
        if _os.getenv("SIGIL_MCP_LLAMA_NO_LOG_FILTER", "").lower() not in {"1", "true", "yes"}:
            _install_llama_log_filter()

        # Preference: explicit constructor args > config.json values > sane defaults
        if n_threads is not None:
            self.n_threads = int(n_threads)
            threads_source = "ctor"
        elif env_threads:
            try:
                self.n_threads = max(1, int(env_threads))
                threads_source = "env"
            except (TypeError, ValueError):
                logger.warning("Invalid SIGIL_MCP_LLAMA_THREADS '%s', falling back to config/default", env_threads)
                cfg_threads = cfg.embeddings_llamacpp_threads
                self.n_threads = int(cfg_threads) if cfg_threads is not None else max(1, cores - 1)
                threads_source = "config" if cfg.embeddings_llamacpp_threads is not None else "default"
        else:
            cfg_threads = cfg.embeddings_llamacpp_threads
            self.n_threads = int(cfg_threads) if cfg_threads is not None else max(1, cores - 1)
            threads_source = "config" if cfg.embeddings_llamacpp_threads is not None else "default"

        if batch_size is not None:
            self.batch_size = int(batch_size)
            batch_source = "ctor"
        elif env_batch:
            try:
                self.batch_size = max(1, int(env_batch))
                batch_source = "env"
            except (TypeError, ValueError):
                logger.warning("Invalid SIGIL_MCP_LLAMA_BATCH '%s', falling back to config/default", env_batch)
                cfg_batch = cfg.embeddings_llamacpp_batch_size
                self.batch_size = int(cfg_batch) if cfg_batch is not None else 32
                batch_source = "config" if cfg.embeddings_llamacpp_batch_size is not None else "default"
        else:
            cfg_batch = cfg.embeddings_llamacpp_batch_size
            self.batch_size = int(cfg_batch) if cfg_batch is not None else 32
            batch_source = "config" if cfg.embeddings_llamacpp_batch_size is not None else "default"

        # Control llama.cpp's internal batch threading separately to reduce CPU load
        if n_threads_batch is not None:
            self.n_threads_batch = int(n_threads_batch)
            threads_batch_source = "ctor"
        elif env_threads_batch:
            try:
                self.n_threads_batch = max(1, int(env_threads_batch))
                threads_batch_source = "env"
            except (TypeError, ValueError):
                logger.warning("Invalid SIGIL_MCP_LLAMA_THREADS_BATCH '%s', falling back to config/default", env_threads_batch)
                cfg_threads_batch = cfg.embeddings_llamacpp_threads_batch
                self.n_threads_batch = int(cfg_threads_batch) if cfg_threads_batch is not None else None
                threads_batch_source = "config" if cfg.embeddings_llamacpp_threads_batch is not None else "default"
        else:
            cfg_threads_batch = cfg.embeddings_llamacpp_threads_batch
            self.n_threads_batch = int(cfg_threads_batch) if cfg_threads_batch is not None else None
            threads_batch_source = "config" if cfg.embeddings_llamacpp_threads_batch is not None else "default"

        # Control llama.cpp token batching (n_batch/n_ubatch) to reduce native-level spikes
        if llama_n_batch is not None:
            self.llama_n_batch = int(llama_n_batch)
            n_batch_source = "ctor"
        elif env_n_batch:
            try:
                self.llama_n_batch = max(1, int(env_n_batch))
                n_batch_source = "env"
            except (TypeError, ValueError):
                logger.warning("Invalid SIGIL_MCP_LLAMA_N_BATCH '%s', falling back to config/default", env_n_batch)
                cfg_n_batch = cfg.embeddings_llamacpp_n_batch
                self.llama_n_batch = int(cfg_n_batch) if cfg_n_batch is not None else 2048
                n_batch_source = "config" if cfg.embeddings_llamacpp_n_batch is not None else "default"
        else:
            cfg_n_batch = cfg.embeddings_llamacpp_n_batch
            self.llama_n_batch = int(cfg_n_batch) if cfg_n_batch is not None else 2048
            n_batch_source = "config" if cfg.embeddings_llamacpp_n_batch is not None else "default"

        default_ubatch = self.llama_n_batch if self.llama_n_batch is not None else 512
        if llama_n_ubatch is not None:
            self.llama_n_ubatch = int(llama_n_ubatch)
            n_ubatch_source = "ctor"
        elif env_n_ubatch:
            try:
                self.llama_n_ubatch = max(1, int(env_n_ubatch))
                n_ubatch_source = "env"
            except (TypeError, ValueError):
                logger.warning("Invalid SIGIL_MCP_LLAMA_N_UBATCH '%s', falling back to config/default", env_n_ubatch)
                cfg_n_ubatch = cfg.embeddings_llamacpp_n_ubatch
                self.llama_n_ubatch = int(cfg_n_ubatch) if cfg_n_ubatch is not None else default_ubatch
                n_ubatch_source = "config" if cfg.embeddings_llamacpp_n_ubatch is not None else "default"
        else:
            cfg_n_ubatch = cfg.embeddings_llamacpp_n_ubatch
            self.llama_n_ubatch = int(cfg_n_ubatch) if cfg_n_ubatch is not None else default_ubatch
            n_ubatch_source = "config" if cfg.embeddings_llamacpp_n_ubatch is not None else "default"

        # llama.cpp hard-aborts if n_ubatch < tokens_per_embed; keep n_batch in sync
        if self.llama_n_ubatch is not None:
            if self.llama_n_batch is None or self.llama_n_batch < self.llama_n_ubatch:
                logger.info(
                    "Adjusting llama_n_batch from %s to %s to satisfy n_batch >= n_ubatch for embeddings",
                    self.llama_n_batch,
                    self.llama_n_ubatch,
                )
                self.llama_n_batch = self.llama_n_ubatch

        logger.info(
            "GPU layers: %s, Context: %s, threads: %s (%s), doc_batch: %s (%s), threads_batch: %s (%s), llama_n_batch: %s (%s), llama_n_ubatch: %s (%s)",
            n_gpu_layers,
            context_size,
            self.n_threads,
            threads_source,
            self.batch_size,
            batch_source,
            self.n_threads_batch if self.n_threads_batch is not None else "default",
            threads_batch_source,
            self.llama_n_batch,
            n_batch_source,
            self.llama_n_ubatch,
            n_ubatch_source,
        )

        llm_kwargs: dict[str, Any] = dict(
            model_path=str(self.model_path),
            n_ctx=context_size,
            n_gpu_layers=n_gpu_layers,
            use_mlock=use_mlock,
            embedding=embedding,
            verbose=False,
            n_threads=self.n_threads,
        )
        if self.n_threads_batch is not None:
            llm_kwargs["n_threads_batch"] = self.n_threads_batch
        if self.llama_n_batch is not None:
            llm_kwargs["n_batch"] = self.llama_n_batch
        if self.llama_n_ubatch is not None:
            llm_kwargs["n_ubatch"] = self.llama_n_ubatch
        # Allow explicit override via kwargs for advanced parameters
        safe_kwargs = {k: v for k, v in (kwargs or {}).items() if v is not None}
        llm_kwargs.update(safe_kwargs)

        self.llm = Llama(**llm_kwargs)
        self._max_tokens_per_segment = self._infer_token_limit(context_size)
        logger.debug(
            "Effective llama.cpp token limits -> ctx: %s, n_batch: %s, n_ubatch: %s, max tokens per embed: %s",
            self._safe_llm_int("n_ctx") or context_size,
            self._safe_llm_int("n_batch") or self.llama_n_batch,
            self._safe_llm_int("n_ubatch") or self.llama_n_ubatch,
            self._max_tokens_per_segment,
        )

        logger.info("Llama.cpp model loaded successfully")

    def _safe_llm_int(self, attr: str) -> int | None:
        """Safely resolve an integer attribute or callable from the llama.cpp binding."""
        val = getattr(self.llm, attr, None)
        try:
            if callable(val):
                val = val()
            return int(val) if val is not None else None
        except Exception:
            return None

    def _infer_token_limit(self, requested_ctx: int) -> int:
        """Determine the maximum tokens we can feed per embed call without hitting GGML_ASSERT."""
        ctx_tokens = self._safe_llm_int("n_ctx") or requested_ctx
        ubatch_tokens = self._safe_llm_int("n_ubatch") or self.llama_n_ubatch
        batch_tokens = self._safe_llm_int("n_batch") or self.llama_n_batch

        # Respect llama-cpp-python's clamping of n_ubatch <= n_batch
        if ubatch_tokens is not None and batch_tokens is not None:
            ubatch_tokens = min(ubatch_tokens, batch_tokens)
        elif ubatch_tokens is None:
            ubatch_tokens = batch_tokens

        limit = ubatch_tokens or batch_tokens or ctx_tokens
        if ctx_tokens is not None and limit is not None:
            limit = min(limit, ctx_tokens)

        # Safety guard: never drop below 1 token
        return max(1, int(limit or ctx_tokens or 1))

    def _split_text_for_embedding(self, text: str) -> list[str]:
        """Split text so each segment fits within the llama.cpp micro-batch token budget."""
        max_tokens = max(1, int(self._max_tokens_per_segment))
        tokenizer = getattr(self.llm, "tokenize", None)
        detokenizer = getattr(self.llm, "detokenize", None)

        # Cheap heuristic to skip tokenization for typical chunks
        if len(text) <= max_tokens * 4:
            return [text]

        if callable(tokenizer):
            try:
                tokens = tokenizer(text.encode("utf-8"), special=True)
                tokens_list = list(tokens) if not isinstance(tokens, list) else tokens
                if len(tokens_list) <= max_tokens:
                    return [text]

                segments: list[str] = []
                for start in range(0, len(tokens_list), max_tokens):
                    chunk_tokens = tokens_list[start : start + max_tokens]
                    chunk_text: str | None = None
                    if callable(detokenizer):
                        try:
                            chunk_bytes = detokenizer(chunk_tokens)
                            chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
                        except Exception:
                            chunk_text = None
                    if not chunk_text:
                        approx_chars = max_tokens * 3
                        chunk_text = text[start * 3 : start * 3 + approx_chars] or text[start : start + approx_chars]
                    segments.append(chunk_text)

                if len(segments) > 1:
                    logger.debug(
                        "Split text into %s segments to satisfy llama.cpp n_ubatch token cap (%s tokens)",
                        len(segments),
                        max_tokens,
                    )
                return segments or [text]
            except Exception:
                logger.debug("Token-based chunking failed; falling back to character slicing", exc_info=True)

        max_chars = max_tokens * 3
        return [text[j : j + max_chars] for j in range(0, len(text), max_chars)] or [text]

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
        # Split documents into segments that fit within n_ubatch to avoid llama.cpp GGML_ASSERT
        doc_segments: list[list[str]] = []
        for text in texts:
            doc_segments.append(self._split_text_for_embedding(text))

        # Flatten segments into a list with mapping back to document index
        flat_segments: list[tuple[int, str]] = []  # (doc_idx, segment_text)
        for doc_idx, segs in enumerate(doc_segments):
            for s in segs:
                flat_segments.append((doc_idx, s))

        segment_results: list[list[float] | None] = [None] * len(flat_segments)

        # Process in batches for efficiency
        i = 0
        while i < len(flat_segments):
            batch = flat_segments[i : i + self.batch_size]
            texts_batch = [t for (_, t) in batch]
            batch_start = perf_counter()
            # Try batch embed if supported, otherwise fall back per-item
            try:
                res = self.llm.embed(texts_batch)  # type: ignore[arg-type]
                batch_time = perf_counter() - batch_start
                # If the result is a sequence of embeddings, map them
                if isinstance(res, (list, tuple)):
                    for j, out in enumerate(res):
                        idx = i + j
                        if out is None:
                            segment_results[idx] = None
                        elif isinstance(out, tuple):
                            segment_results[idx] = list(out[0])  # type: ignore
                        else:
                            segment_results[idx] = list(out)  # type: ignore
                else:
                    # Single-result returned; fallback to per-segment
                    raise TypeError("embed returned single result for batch")
            except Exception:
                # Fallback: call embed per segment
                for j, (_, seg_text) in enumerate(batch):
                    try:
                        r = self.llm.embed(seg_text)
                        if isinstance(r, tuple):
                            segment_results[i + j] = list(r[0])  # type: ignore
                        else:
                            segment_results[i + j] = list(r)  # type: ignore
                    except Exception:
                        logger.exception("Failed to embed segment batch item at index %s", i + j)
                        segment_results[i + j] = None

            i += self.batch_size

        # Aggregate segment embeddings back into per-document embeddings
        embeddings: list[list[float]] = []
        seg_idx = 0
        for doc_idx, segs in enumerate(doc_segments):
            seg_count = len(segs)
            seg_vecs: list[list[float]] = []
            for _ in range(seg_count):
                vec = segment_results[seg_idx]
                seg_idx += 1
                if vec:
                    seg_vecs.append(vec)

            if not seg_vecs:
                logger.warning("No embeddings produced for document %s", doc_idx + 1)
                embeddings.append([])
            elif len(seg_vecs) == 1:
                embeddings.append(seg_vecs[0])
            else:
                embeddings.append(self._combine_embeddings(seg_vecs))

            if (doc_idx + 1) % 10 == 0:
                logger.info(f"Generated embeddings for {doc_idx+1}/{len(texts)} documents")

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        segments = self._split_text_for_embedding(text)
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
