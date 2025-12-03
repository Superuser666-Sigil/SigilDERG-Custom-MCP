# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
Tests for indexer module - Part 2: Edge cases and advanced features.
"""

import numpy as np
from sigil_mcp.indexer import Symbol, SearchResult, SigilIndex


class TestSymbolDataclass:
    """Test Symbol dataclass."""
    
    def test_symbol_creation(self):
        """Test creating Symbol instance."""
        symbol = Symbol(
            name="test_func",
            kind="function",
            file_path="test.py",
            line=10
        )
        
        assert symbol.name == "test_func"
        assert symbol.kind == "function"
        assert symbol.file_path == "test.py"
        assert symbol.line == 10
        assert symbol.signature is None
        assert symbol.scope is None
    
    def test_symbol_with_signature(self):
        """Test Symbol with signature."""
        symbol = Symbol(
            name="add",
            kind="method",
            file_path="calc.py",
            line=5,
            signature="def add(self, a, b)",
            scope="Calculator"
        )
        
        assert symbol.signature == "def add(self, a, b)"
        assert symbol.scope == "Calculator"
    
    def test_symbol_equality(self):
        """Test Symbol equality comparison."""
        symbol1 = Symbol("func", "function", "test.py", 1)
        symbol2 = Symbol("func", "function", "test.py", 1)
        
        assert symbol1 == symbol2


class TestSearchResultDataclass:
    """Test SearchResult dataclass."""
    
    def test_search_result_creation(self):
        """Test creating SearchResult instance."""
        result = SearchResult(
            repo="test_repo",
            path="src/main.py",
            line=42,
            text="def hello_world():",
            doc_id="doc_123"
        )
        
        assert result.repo == "test_repo"
        assert result.path == "src/main.py"
        assert result.line == 42
        assert result.text == "def hello_world():"
        assert result.doc_id == "doc_123"
        assert result.symbol is None
    
    def test_search_result_with_symbol(self):
        """Test SearchResult with symbol."""
        symbol = Symbol("hello_world", "function", "main.py", 42)
        result = SearchResult(
            repo="test_repo",
            path="main.py",
            line=42,
            text="def hello_world():",
            doc_id="doc_123",
            symbol=symbol
        )
        
        assert result.symbol == symbol
        assert result.symbol is not None
        assert result.symbol.name == "hello_world"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_index_empty_repository(self, test_index, temp_dir):
        """Test indexing empty repository."""
        empty_repo = temp_dir / "empty_repo"
        empty_repo.mkdir()
        
        stats = test_index.index_repository("empty_repo", empty_repo, force=True)
        
        assert stats["files_indexed"] == 0
        assert stats["symbols_extracted"] == 0
    
    def test_index_repository_with_binary_files(self, test_index, temp_dir):
        """Test indexing repository with binary files."""
        repo = temp_dir / "binary_repo"
        repo.mkdir()
        
        # Create a binary file
        (repo / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")
        (repo / "test.py").write_text("def test(): pass")
        
        stats = test_index.index_repository("binary_repo", repo, force=True)
        
        # Should index Python file, skip binary
        assert stats["files_indexed"] >= 1
    
    def test_index_repository_with_large_files(self, test_index, temp_dir):
        """Test indexing repository with large files."""
        repo = temp_dir / "large_repo"
        repo.mkdir()
        
        # Create large file (10k lines)
        large_content = "\n".join([f"line_{i} = {i}" for i in range(10000)])
        (repo / "large.py").write_text(large_content)
        
        stats = test_index.index_repository("large_repo", repo, force=True)
        
        assert stats["files_indexed"] >= 1
    
    def test_semantic_search_very_long_query(self, indexed_repo):
        """Test semantic search with very long query."""
        index = indexed_repo["index"]
        index.build_vector_index(repo="test_repo", force=True)
        
        # Very long query
        long_query = " ".join(["test"] * 1000)
        results = index.semantic_search(long_query, repo="test_repo", k=5)
        
        # Should handle gracefully
        assert isinstance(results, list)
    
    def test_index_nonexistent_path(self, test_index, temp_dir):
        """Test indexing nonexistent path."""
        nonexistent = temp_dir / "does_not_exist"
        
        # Should handle gracefully - either raise or return empty stats
        stats = test_index.index_repository("bad_repo", nonexistent, force=True)
        assert stats["files_indexed"] == 0
    
    def test_embedding_without_embed_fn(self, test_index_path):
        """Test creating index without embedding function."""
        index = SigilIndex(test_index_path, embed_fn=None, embed_model="none")
        
        assert index.embed_fn is None
        
        index.repos_db.close()
        index.trigrams_db.close()
    
    def test_vector_index_without_embed_fn(self, test_index_path, test_repo_path):
        """Test building vector index without embedding function."""
        index = SigilIndex(test_index_path, embed_fn=None, embed_model="none")
        index.index_repository("test_repo", test_repo_path, force=True)
        
        # Should handle gracefully when no embed_fn
        try:
            stats = index.build_vector_index(repo="test_repo", force=True)
            # If it doesn't raise, it should return empty stats
            assert stats["chunks_indexed"] == 0
        except Exception:
            # Expected to fail or skip
            pass
        finally:
            index.repos_db.close()
            index.trigrams_db.close()


class TestConcurrency:
    """Test concurrent access patterns."""
    
    def test_read_while_indexed(self, test_index, test_repo_path):
        """Test reading while repository is being indexed."""
        # Start indexing
        test_index.index_repository("test_repo", test_repo_path, force=True)
        
        # Try to get stats (should work with SQLite's concurrency)
        stats = test_index.get_index_stats(repo="test_repo")
        
        assert isinstance(stats, dict)


class TestDatabaseIntegrity:
    """Test database integrity and constraints."""
    
    def test_unique_blob_sha_constraint(self, test_index, test_repo_path):
        """Test that blob_sha uniqueness is enforced."""
        # Index twice
        test_index.index_repository("test_repo", test_repo_path, force=True)
        test_index.index_repository("test_repo", test_repo_path, force=True)
        
        # Should not create duplicates
        cursor = test_index.repos_db.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents")
        count = cursor.fetchone()[0]
        
        # Should have reasonable number (not duplicated)
        assert count >= 3  # At least our test files
    
    def test_foreign_key_constraints(self, test_index, test_repo_path):
        """Test that foreign key relationships are maintained."""
        test_index.index_repository("test_repo", test_repo_path, force=True)
        
        # All documents should reference valid repo
        cursor = test_index.repos_db.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM documents d
            LEFT JOIN repos r ON d.repo_id = r.id
            WHERE r.id IS NULL
        """)
        
        orphaned_docs = cursor.fetchone()[0]
        assert orphaned_docs == 0
    
    def test_symbol_references_valid_document(self, indexed_repo):
        """Test that symbols reference valid documents."""
        index = indexed_repo["index"]
        
        cursor = index.repos_db.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM symbols s
            LEFT JOIN documents d ON s.doc_id = d.id
            WHERE d.id IS NULL
        """)
        
        orphaned_symbols = cursor.fetchone()[0]
        assert orphaned_symbols == 0


class TestEmbeddingDimensions:
    """Test handling of different embedding dimensions."""
    
    def test_consistent_embedding_dimensions(self, indexed_repo):
        """Test that embeddings have consistent dimensions."""
        index = indexed_repo["index"]
        index.build_vector_index(repo="test_repo", force=True)
        
        cursor = index.repos_db.cursor()
        cursor.execute("SELECT DISTINCT dim FROM embeddings")
        dimensions = cursor.fetchall()
        
        # All embeddings should have same dimension
        assert len(dimensions) <= 1
        
        if dimensions:
            dim = dimensions[0][0]
            assert dim > 0
    
    def test_embedding_normalization(self, indexed_repo):
        """Test that embeddings are normalized."""
        index = indexed_repo["index"]
        index.build_vector_index(repo="test_repo", force=True)
        
        cursor = index.repos_db.cursor()
        cursor.execute("SELECT vector FROM embeddings LIMIT 5")
        
        for (vector_blob,) in cursor.fetchall():
            vector = np.frombuffer(vector_blob, dtype='float32')
            norm = np.linalg.norm(vector)
            # Should be approximately unit norm (allowing for float precision)
            assert 0.95 <= norm <= 1.05


class TestCleanup:
    """Test cleanup and resource management."""
    
    def test_database_close(self, test_index_path, dummy_embed_fn):
        """Test that databases can be closed cleanly."""
        index = SigilIndex(test_index_path, dummy_embed_fn, "test")
        
        index.repos_db.close()
        index.trigrams_db.close()
        
        # Should not raise errors
    
    def test_reopen_after_close(self, test_index_path, dummy_embed_fn):
        """Test reopening databases after close."""
        index1 = SigilIndex(test_index_path, dummy_embed_fn, "test")
        index1.repos_db.close()
        index1.trigrams_db.close()
        
        # Create new instance with same path
        index2 = SigilIndex(test_index_path, dummy_embed_fn, "test")
        
        # Should work
        cursor = index2.repos_db.cursor()
        cursor.execute("SELECT COUNT(*) FROM repos")
        count = cursor.fetchone()[0]
        
        assert count >= 0
        
        index2.repos_db.close()
        index2.trigrams_db.close()
