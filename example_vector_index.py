#!/usr/bin/env python3
# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
Example: Using the vector index with Sigil MCP Server.

This demonstrates how to build and use semantic embeddings for code search.
"""

from pathlib import Path
from sigil_mcp.indexer import SigilIndex
import numpy as np


# Example: Simple dummy embedding function for demonstration
# In production, replace with OpenAI, sentence-transformers, or similar
def dummy_embed_fn(texts):
    """
    Dummy embedding function that returns random vectors.
    
    In production, replace with:
    - OpenAI embeddings API
    - sentence-transformers (e.g., all-MiniLM-L6-v2)
    - Code-specific models (e.g., CodeBERT, GraphCodeBERT)
    """
    # Fixed dimensionality for consistency
    dim = 384
    embeddings = np.random.randn(len(texts), dim).astype('float32')
    # Normalize to unit vectors (common for cosine similarity)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    return embeddings


def main():
    # Initialize index with embedding function
    index_path = Path.home() / ".sigil-index"
    index = SigilIndex(
        index_path=index_path,
        embed_fn=dummy_embed_fn,
        embed_model="dummy-v1"
    )
    
    # First, index a repository (if not already indexed)
    repo_name = "my-project"
    repo_path = Path.cwd()  # or specify your repo path
    
    print(f"Indexing repository: {repo_name}")
    try:
        stats = index.index_repository(repo_name, repo_path, force=False)
        print(f"Indexed {stats['files_indexed']} files, "
              f"{stats['symbols_extracted']} symbols")
    except Exception as e:
        print(f"Note: Repository may already be indexed ({e})")
    
    # Build vector index
    print(f"\nBuilding vector index for {repo_name}...")
    vector_stats = index.build_vector_index(
        repo=repo_name,
        force=False  # Set True to rebuild
    )
    
    print("Vector index complete:")
    print(f"  Documents processed: {vector_stats['documents_processed']}")
    print(f"  Chunks indexed: {vector_stats['chunks_indexed']}")
    
    # Get stats
    stats = index.get_index_stats(repo=repo_name)
    print(f"\nIndex statistics for {repo_name}:")
    print(f"  Documents: {stats.get('documents', 0)}")
    print(f"  Symbols: {stats.get('symbols', 0)}")
    print(f"  Indexed at: {stats.get('indexed_at', 'N/A')}")


# Example: Using sentence-transformers (real embeddings)
def example_with_sentence_transformers():
    """
    Example using sentence-transformers for real embeddings.
    
    Install: pip install sentence-transformers
    """
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load model (first run downloads ~80MB)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        def embed_fn(texts):
            # Generate embeddings
            embeddings = model.encode(texts, show_progress_bar=False)
            return embeddings
        
        # Create index with real embeddings
        index = SigilIndex(
            index_path=Path.home() / ".sigil-index",
            embed_fn=embed_fn,
            embed_model="all-MiniLM-L6-v2"
        )
        
        print("Using sentence-transformers for embeddings")
        return index
    
    except ImportError:
        print("sentence-transformers not installed")
        print("Install with: pip install sentence-transformers")
        return None


# Example: Using OpenAI embeddings
def example_with_openai():
    """
    Example using OpenAI embeddings API.
    
    Install: pip install openai
    Set OPENAI_API_KEY environment variable
    """
    try:
        import openai
        import os
        
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        def embed_fn(texts):
            # Batch embed with OpenAI
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            embeddings = np.array([
                item.embedding for item in response.data
            ], dtype='float32')
            return embeddings
        
        # Create index with OpenAI embeddings
        index = SigilIndex(
            index_path=Path.home() / ".sigil-index",
            embed_fn=embed_fn,
            embed_model="text-embedding-3-small"
        )
        
        print("Using OpenAI embeddings")
        return index
    
    except ImportError:
        print("openai package not installed")
        print("Install with: pip install openai")
        return None
    except Exception as e:
        print(f"Error setting up OpenAI: {e}")
        return None


if __name__ == "__main__":
    main()
    
    print("\n" + "="*60)
    print("For production use, consider:")
    print("1. sentence-transformers: example_with_sentence_transformers()")
    print("2. OpenAI embeddings: example_with_openai()")
    print("="*60)
