#!/usr/bin/env python3
"""
Script to delete all embeddings, trigrams, and vectors, then rebuild them.

This script:
1. Deletes all entries from the trigrams table
2. Deletes all entries from the embeddings table
3. Rebuilds trigrams by re-indexing all repositories
4. Rebuilds embeddings/vectors for all repositories (if embeddings enabled)
"""

import sys
from pathlib import Path
import sqlite3
import logging
import shutil

from sigil_mcp.indexer import SigilIndex
from sigil_mcp.config import get_config
from sigil_mcp.embeddings import create_embedding_provider
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)


def delete_all_trigrams(index: SigilIndex) -> int:
    """Delete all trigrams from the trigrams database."""
    logger.info("Deleting all trigrams...")
    cursor = index.trigrams_db.cursor()
    cursor.execute("SELECT COUNT(*) FROM trigrams")
    count_before = cursor.fetchone()[0]
    
    cursor.execute("DELETE FROM trigrams")
    index.trigrams_db.commit()
    
    logger.info(f"Deleted {count_before} trigram entries")
    return count_before


def delete_all_embeddings(index: SigilIndex) -> int:
    """Delete all embeddings from the repos database."""
    logger.info("Deleting all embeddings...")
    cursor = index.repos_db.cursor()
    cursor.execute("SELECT COUNT(*) FROM embeddings")
    count_before = cursor.fetchone()[0]
    
    cursor.execute("DELETE FROM embeddings")
    index.repos_db.commit()
    
    logger.info(f"Deleted {count_before} embedding entries")
    return count_before


def rebuild_trigrams_for_repo(index: SigilIndex, repo_name: str, repo_path: Path) -> dict:
    """Rebuild trigrams for a repository by re-indexing."""
    logger.info(f"Rebuilding trigrams for {repo_name}...")
    stats = index.index_repository(repo_name, repo_path, force=True)
    logger.info(
        f"  Indexed {stats.get('files_indexed', 0)} files, "
        f"{stats.get('trigrams_built', 0)} trigrams"
    )
    return stats


def rebuild_embeddings_for_repo(
    index: SigilIndex,
    repo_name: str,
    embed_fn,
    model: str
) -> dict:
    """Rebuild embeddings for a repository."""
    logger.info(f"Rebuilding embeddings for {repo_name}...")
    stats = index.build_vector_index(
        repo=repo_name,
        embed_fn=embed_fn,
        model=model,
        force=True
    )
    logger.info(
        f"  Processed {stats.get('documents_processed', 0)} documents, "
        f"{stats.get('chunks_indexed', 0)} chunks"
    )
    return stats


def main():
    """Main execution."""
    print("=" * 80)
    print("SIGIL MCP SERVER - REBUILD INDEXES")
    print("=" * 80)
    print()
    
    config = get_config()

    # Step 0: delete entire index directory for a truly clean rebuild
    index_dir = config.index_path
    if index_dir.exists():
        logger.info(f"Removing entire index directory at {index_dir}")
        shutil.rmtree(index_dir)
    # Recreate base directory for subsequent operations
    index_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize index
    logger.info("Initializing index...")
    index = SigilIndex(
        config.index_path,
        embed_fn=None,
        embed_model="none"
    )
    logger.info(f"Index path: {config.index_path}")
    print()
    
    # List repositories
    repos = config.repositories
    if not repos:
        logger.error("No repositories configured!")
        return 1
    
    print(f"Found {len(repos)} configured repositories:")
    for name, path in repos.items():
        print(f"  - {name}: {path}")
    print()
    
    # Step 1: Delete all trigrams
    print("=" * 80)
    print("STEP 1: DELETING TRIGRAMS")
    print("=" * 80)
    trigram_count = delete_all_trigrams(index)
    print()
    
    # Step 2: Delete all embeddings
    print("=" * 80)
    print("STEP 2: DELETING EMBEDDINGS")
    print("=" * 80)
    embedding_count = delete_all_embeddings(index)
    print()
    
    # Step 3: Rebuild trigrams for all repos
    print("=" * 80)
    print("STEP 3: REBUILDING TRIGRAMS")
    print("=" * 80)
    trigram_stats = {}
    for repo_name, repo_path_str in repos.items():
        repo_path = Path(repo_path_str)
        if not repo_path.exists():
            logger.warning(f"Repository path does not exist: {repo_path}")
            continue
        
        try:
            stats = rebuild_trigrams_for_repo(index, repo_name, repo_path)
            trigram_stats[repo_name] = stats
        except Exception as e:
            logger.error(f"Error rebuilding trigrams for {repo_name}: {e}")
    print()
    
    # Step 4: Rebuild embeddings if enabled
    if config.embeddings_enabled:
        print("=" * 80)
        print("STEP 4: REBUILDING EMBEDDINGS")
        print("=" * 80)
        
        provider = config.embeddings_provider
        model_name = config.embeddings_model
        
        if not provider or not model_name:
            logger.warning(
                "Embeddings enabled but provider/model not configured. "
                "Skipping embedding rebuild."
            )
        else:
            try:
                logger.info(f"Initializing embedding provider: {provider}")
                logger.info(f"Model: {model_name}")
                
                kwargs = dict(config.embeddings_kwargs)
                if config.embeddings_cache_dir:
                    kwargs["cache_dir"] = config.embeddings_cache_dir
                if provider == "openai" and config.embeddings_api_key:
                    kwargs["api_key"] = config.embeddings_api_key
                
                embedding_provider = create_embedding_provider(
                    provider=provider,
                    model=model_name,
                    dimension=config.embeddings_dimension,
                    **kwargs
                )
                
                def embed_fn(texts):
                    embeddings_list = embedding_provider.embed_documents(list(texts))
                    return np.array(embeddings_list, dtype="float32")
                
                # Update index with embedding function
                index.embed_fn = embed_fn
                index.embed_model = f"{provider}:{model_name}"
                
                embedding_stats = {}
                for repo_name in repos.keys():
                    try:
                        stats = rebuild_embeddings_for_repo(
                            index,
                            repo_name,
                            embed_fn,
                            index.embed_model
                        )
                        embedding_stats[repo_name] = stats
                    except Exception as e:
                        logger.error(f"Error rebuilding embeddings for {repo_name}: {e}")
                
                print()
                print("Embedding rebuild summary:")
                for repo_name, stats in embedding_stats.items():
                    print(
                        f"  {repo_name}: {stats.get('documents_processed', 0)} docs, "
                        f"{stats.get('chunks_indexed', 0)} chunks"
                    )
            except Exception as e:
                logger.error(f"Error initializing embedding provider: {e}")
                logger.error("Skipping embedding rebuild.")
    else:
        logger.info("Embeddings disabled in config - skipping embedding rebuild")
    
    print()
    print("=" * 80)
    print("REBUILD COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  Deleted trigrams: {trigram_count}")
    print(f"  Deleted embeddings: {embedding_count}")
    print()
    print("Trigram rebuild summary:")
    for repo_name, stats in trigram_stats.items():
        print(
            f"  {repo_name}: {stats.get('files_indexed', 0)} files, "
            f"{stats.get('trigrams_built', 0)} trigrams"
        )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

