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
from sigil_mcp.embeddings import create_embedding_provider
import numpy as np


# Example: Simple dummy embedding function for demonstration
# In production, use create_embedding_provider() from sigil_mcp.embeddings
def dummy_embed_fn(texts):
    """
    Dummy embedding function that returns random vectors.
    
    In production, use create_embedding_provider() which supports:
    - sentence-transformers (e.g., all-MiniLM-L6-v2)
    - OpenAI embeddings API (text-embedding-3-small, etc.)
    - llamacpp (local GGUF models)
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
        # Use the built-in provider factory
        provider = create_embedding_provider(
            provider='sentence-transformers',
            model='all-MiniLM-L6-v2',
            dimension=384
        )
        
        def embed_fn(texts):
            # Returns list[list[float]], convert to numpy array
            embeddings = provider.embed_documents(texts)
            return np.array(embeddings, dtype='float32')
        
        # Create index with real embeddings
        index = SigilIndex(
            index_path=Path.home() / ".sigil-index",
            embed_fn=embed_fn,
            embed_model="all-MiniLM-L6-v2"
        )
        
        print("Using sentence-transformers for embeddings")
        return index
    
    except ImportError as e:
        print(f"Error: {e}")
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
        import os
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable not set")
            return None
        
        # Use the built-in provider factory
        provider = create_embedding_provider(
            provider='openai',
            model='text-embedding-3-small',
            dimension=1536,  # text-embedding-3-small dimension
            api_key=api_key
        )
        
        def embed_fn(texts):
            # Returns list[list[float]], convert to numpy array
            embeddings = provider.embed_documents(texts)
            return np.array(embeddings, dtype='float32')
        
        # Create index with OpenAI embeddings
        index = SigilIndex(
            index_path=Path.home() / ".sigil-index",
            embed_fn=embed_fn,
            embed_model="text-embedding-3-small"
        )
        
        print("Using OpenAI embeddings")
        return index
    
    except ImportError as e:
        print(f"Error: {e}")
        print("Install with: pip install openai")
        return None
    except Exception as e:
        print(f"Error setting up OpenAI: {e}")
        return None


# Example: Using llamacpp (local GGUF models)
def example_with_llamacpp():
    """
    Example using llamacpp for local GGUF embedding models.
    
    Install: pip install llama-cpp-python
    Download a GGUF model (e.g., nomic-embed-text)
    """
    try:
        model_path = Path.home() / "models" / "nomic-embed-text-v1.5.Q4_K_M.gguf"
        
        if not model_path.exists():
            print(f"Error: Model not found at {model_path}")
            print("Download from: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF")
            return None
        
        # Use the built-in provider factory
        provider = create_embedding_provider(
            provider='llamacpp',
            model=str(model_path),
            dimension=768,  # nomic-embed dimension
            n_ctx=8192,
            n_batch=512
        )
        
        def embed_fn(texts):
            # Returns list[list[float]], convert to numpy array
            embeddings = provider.embed_documents(texts)
            return np.array(embeddings, dtype='float32')
        
        # Create index with llamacpp embeddings
        index = SigilIndex(
            index_path=Path.home() / ".sigil-index",
            embed_fn=embed_fn,
            embed_model="nomic-embed-text-v1.5"
        )
        
        print("Using llamacpp for local embeddings")
        return index
    
    except ImportError as e:
        print(f"Error: {e}")
        print("Install with: pip install llama-cpp-python")
        return None
    except Exception as e:
        print(f"Error setting up llamacpp: {e}")
        return None


if __name__ == "__main__":
    main()
    
    print("\n" + "="*60)
    print("For production use, consider:")
    print("1. sentence-transformers: example_with_sentence_transformers()")
    print("2. OpenAI embeddings: example_with_openai()")
    print("3. Local GGUF models: example_with_llamacpp()")
    print("="*60)
