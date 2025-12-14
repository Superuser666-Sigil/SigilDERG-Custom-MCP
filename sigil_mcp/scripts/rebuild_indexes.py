# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""CLI helpers for wiping and rebuilding Sigil MCP indexes."""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..config import get_config, load_config
from ..embeddings import create_embedding_provider
from ..indexer import SigilIndex

logger = logging.getLogger(__name__)


def _find_config_path() -> Path | None:
    """
    Locate config.json for CLI runs.

    Search order:
    1) Current working directory or any of its parents
    2) Any parent of this file (useful when installed)
    """
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        cand = parent / "config.json"
        if cand.exists():
            return cand

    for parent in Path(__file__).resolve().parents:
        cand = parent / "config.json"
        if cand.exists():
            return cand
    return None


def delete_all_trigrams(index: SigilIndex) -> int:
    """Delete all trigram postings from the trigrams database."""
    logger.info("Deleting all trigrams...")
    # Use backend-agnostic iteration to delete K/V-backed trigram postings.
    count_before = 0
    try:
        count_before = index._trigram_count()
        for gram, _ in index._trigram_iter_items():
            index._trigram_delete(gram)
    except Exception:
        logger.exception("Failed to delete trigrams via backend iteration")
        count_before = 0

    logger.info("Deleted %s trigram entries", count_before)
    return count_before


def rebuild_trigrams_for_repo(index: SigilIndex, repo_name: str, repo_path: Path) -> dict:
    """Rebuild trigrams/symbols for a repository by re-indexing."""
    logger.info("Rebuilding trigrams for %s...", repo_name)
    stats = index.index_repository(repo_name, repo_path, force=True)
    logger.info(
        "  Indexed %s files, %s trigrams",
        stats.get("files_indexed", 0),
        stats.get("trigrams_built", 0),
    )
    return stats


def rebuild_embeddings_for_repo(
    index: SigilIndex,
    repo_name: str,
    embed_fn,
    model: str,
) -> dict:
    """Rebuild embeddings for a repository."""
    logger.info("Rebuilding embeddings for %s...", repo_name)
    stats = index.build_vector_index(
        repo=repo_name,
        embed_fn=embed_fn,
        model=model,
        force=True,
    )
    logger.info(
        "  Processed %s documents, %s chunks",
        stats.get("documents_processed", 0),
        stats.get("chunks_indexed", 0),
    )
    return stats


def _setup_index_for_rebuild(index: SigilIndex | None, wipe_index: bool) -> SigilIndex:
    """Initialize or prepare index for rebuild."""
    config_path = _find_config_path()
    config = load_config(config_path) if config_path else get_config()

    index_dir = config.index_path
    lance_dir = config.lance_dir
    if wipe_index:
        if index_dir.exists():
            logger.info("Removing entire index directory at %s", index_dir)
            shutil.rmtree(index_dir)
        if lance_dir.exists() and lance_dir != index_dir:
            logger.info("Removing LanceDB directory at %s", lance_dir)
            shutil.rmtree(lance_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    if index is None:
        logger.info("Initializing index at %s", config.index_path)
        index = SigilIndex(
            config.index_path,
            embed_fn=None,
            embed_model="none",
        )

    return index


def _resolve_repo_path(repo_value: Any) -> Path:
    """Normalize a repo entry (str or dict with path) into a Path."""
    if isinstance(repo_value, dict):
        repo_path = repo_value.get("path")
    else:
        repo_path = repo_value
    return Path(repo_path)


def _rebuild_trigrams_for_all_repos(index: SigilIndex, repos: dict[str, Any]) -> dict[str, dict]:
    """Rebuild trigrams for all configured repositories."""
    logger.info("Rebuilding trigrams for all repositories...")
    trigram_stats: dict[str, dict] = {}
    for repo_name, repo_value in repos.items():
        repo_path = _resolve_repo_path(repo_value)
        if not repo_path.exists():
            logger.warning("Repository path does not exist: %s", repo_path)
            continue

        stats = rebuild_trigrams_for_repo(index, repo_name, repo_path)
        trigram_stats[repo_name] = stats

    return trigram_stats


def _setup_embedding_function(config) -> tuple:
    """Initialize and return embedding function and model name."""
    provider = config.embeddings_provider
    model_name = config.embeddings_model

    if not provider or not model_name:
        raise ValueError("Embeddings enabled but provider/model not configured.")

    logger.info("Initializing embedding provider: %s", provider)
    logger.info("Model: %s", model_name)

    kwargs = dict(config.embeddings_kwargs)
    if config.embeddings_cache_dir:
        kwargs["cache_dir"] = config.embeddings_cache_dir
    if provider == "openai" and config.embeddings_api_key:
        kwargs["api_key"] = config.embeddings_api_key

    embedding_provider = create_embedding_provider(
        provider=provider,
        model=model_name,
        dimension=config.embeddings_dimension,
        **kwargs,
    )

    def embed_fn(texts):
        embeddings_list = embedding_provider.embed_documents(list(texts))
        return np.array(embeddings_list, dtype="float32")

    return embed_fn, f"{provider}:{model_name}"


def _rebuild_embeddings_for_all_repos(
    index: SigilIndex,
    repos: dict[str, Any],
    embed_fn,
    model: str,
) -> dict[str, dict]:
    """Rebuild embeddings for all repositories."""
    logger.info("Rebuilding embeddings...")
    embedding_stats: dict[str, dict] = {}

    for repo_name, repo_value in repos.items():
        repo_path = _resolve_repo_path(repo_value)
        if not repo_path.exists():
            logger.warning("Repository path does not exist for embeddings: %s", repo_path)
            continue
        stats = rebuild_embeddings_for_repo(index, repo_name, embed_fn, model)
        embedding_stats[repo_name] = stats

    return embedding_stats


def rebuild_all_indexes(
    index: SigilIndex | None = None,
    wipe_index: bool = True,
    rebuild_embeddings: bool = True,
) -> dict[str, Any]:
    """Rebuild all indexes using the same logic as this CLI."""

    config_path = _find_config_path()
    config = load_config(config_path) if config_path else get_config()
    index = _setup_index_for_rebuild(index, wipe_index)

    repos = config.repositories
    if not repos:
        raise ValueError("No repositories configured!")

    logger.info("Found %s configured repositories", len(repos))

    trigram_count = delete_all_trigrams(index)
    trigram_stats = _rebuild_trigrams_for_all_repos(index, repos)

    embedding_stats: dict[str, dict] = {}
    if rebuild_embeddings and config.embeddings_enabled:
        embed_fn, model_name = _setup_embedding_function(config)
        index.embed_fn = embed_fn
        index.embed_model = model_name
        embedding_stats = _rebuild_embeddings_for_all_repos(
            index, repos, embed_fn, model_name
        )
    elif not config.embeddings_enabled:
        logger.info("Embeddings disabled in config - skipping embedding rebuild")

    total_files = sum(s.get("files_indexed", 0) for s in trigram_stats.values())
    total_symbols = sum(s.get("symbols_extracted", 0) for s in trigram_stats.values())

    return {
        "success": True,
        "status": "completed",
        "message": f"Successfully rebuilt indexes for {len(repos)} repositories",
        "deleted_trigrams": trigram_count,
        "stats": {
            "documents": total_files,
            "symbols": total_symbols,
            "files": total_files,
        },
        "trigram_stats": trigram_stats,
        "embedding_stats": embedding_stats,
        "repos": {
            name: {
                "files": stats.get("files_indexed", 0),
                "symbols": stats.get("symbols_extracted", 0),
                "trigrams": stats.get("trigrams_built", 0),
            }
            for name, stats in trigram_stats.items()
        },
    }


def rebuild_single_repo_index(
    index: SigilIndex,
    repo_name: str,
    repo_path: Path,
    rebuild_embeddings: bool = False,
    embed_fn=None,
    model: str = "default",
) -> dict[str, Any]:
    """Rebuild index for a single repository using script logic."""

    if not repo_path.exists():
        raise ValueError(f"Repository path does not exist: {repo_path}")

    trigram_stats = rebuild_trigrams_for_repo(index, repo_name, repo_path)

    embedding_stats = None
    if rebuild_embeddings:
        if embed_fn is None:
            raise ValueError("embed_fn required for rebuilding embeddings")
        embedding_stats = rebuild_embeddings_for_repo(
            index, repo_name, embed_fn, model
        )

    return {
        "success": True,
        "status": "completed",
        "repo": repo_name,
        "message": f"Successfully rebuilt index for {repo_name}",
        "stats": {
            "documents": trigram_stats.get("files_indexed", 0),
            "symbols": trigram_stats.get("symbols_extracted", 0),
            "files": trigram_stats.get("files_indexed", 0),
        },
        "duration_seconds": trigram_stats.get("duration_seconds", 0),
        **trigram_stats,
        "embedding_stats": embedding_stats,
    }


def main() -> int:
    """Main execution (CLI entry point)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    print("=" * 80)
    print("SIGIL MCP SERVER - REBUILD INDEXES")
    print("=" * 80)
    print()

    try:
        result = rebuild_all_indexes(wipe_index=True, rebuild_embeddings=True)

        print()
        print("=" * 80)
        print("REBUILD COMPLETE")
        print("=" * 80)
        print()
        print("Summary:")
        print(f"  Deleted trigrams: {result['deleted_trigrams']}")
        print()
        print("Trigram rebuild summary:")
        for repo_name, stats in result["repos"].items():
            print(
                f"  {repo_name}: {stats['files']} files, "
                f"{stats['trigrams']} trigrams"
            )

        if result.get("embedding_stats"):
            print()
            print("Embedding rebuild summary:")
            for repo_name, stats in result["embedding_stats"].items():
                print(
                    f"  {repo_name}: {stats.get('documents_processed', 0)} docs, "
                    f"{stats.get('chunks_indexed', 0)} chunks"
                )

        return 0
    except Exception as exc:  # pragma: no cover - surface errors at CLI
        logger.error("Rebuild failed: %s", exc)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
