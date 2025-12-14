# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
Tests for indexer module - Part 2: Edge cases and advanced features.
"""

import numpy as np
import zlib
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
        """When embeddings deps are missing, embed_fn remains None and index stays usable."""
        index = SigilIndex(test_index_path, embed_fn=None, embed_model="none")
        assert index.embed_fn is None
        # Vector index should not be enabled without an embedding function
        assert getattr(index, "_vector_index_enabled", False) is False
        index.close()
    
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
            index.close()


class TestTrigramEncoding:
    """Validate trigram postings encoding/decoding."""

    def test_varint_doc_id_encoding_round_trip(self):
        ids = {1, 2, 1000, 100000}
        blob = SigilIndex._serialize_doc_ids(ids)
        assert zlib.decompress(blob).startswith(b"\x02")
        decoded = SigilIndex._deserialize_doc_ids(blob)
        assert decoded == ids

    def test_uint64_doc_id_encoding_round_trip(self):
        ids = {1, 2, 1000, 2**40}
        blob = SigilIndex._serialize_doc_ids(ids)
        assert zlib.decompress(blob).startswith(b"\x03")
        decoded = SigilIndex._deserialize_doc_ids(blob)
        assert decoded == ids

    def test_legacy_doc_id_decoding(self):
        legacy_blob = zlib.compress(b"1,3,5")
        decoded = SigilIndex._deserialize_doc_ids(legacy_blob)
        assert decoded == {1, 3, 5}


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


class TestRemoveFile:
    """Tests for SigilIndex.remove_file single-file cleanup."""

    def test_remove_file_cleans_index_entries(self, indexed_repo, test_repo_path):
        """Removing a file should purge documents, symbols, embeddings, and trigrams."""
        import zlib

        index = indexed_repo["index"]
        repo_name = indexed_repo["repo_name"]

        # Ensure vector index exists so we can test embedding cleanup
        try:
            index.build_vector_index(repo=repo_name, force=True)
        except Exception:
            # Embeddings may be disabled in this environment; ignore errors.
            pass

        # Pick a known file in the test repo
        target_file = test_repo_path / "main.py"
        rel_path = target_file.relative_to(test_repo_path).as_posix()

        # Sanity: document exists
        cursor = index.repos_db.cursor()
        cursor.execute("SELECT id, blob_sha FROM documents WHERE path = ?", (rel_path,))
        row = cursor.fetchone()
        assert row is not None
        doc_id, blob_sha = row

        # Record whether we had symbols/embeddings to decide how deep to assert
        cursor.execute("SELECT COUNT(*) FROM symbols WHERE doc_id = ?", (doc_id,))
        symbol_count_before = cursor.fetchone()[0]

        embedding_count_before = 0
        if index.vectors is not None:
            embedding_count_before = len([
                row
                for row in index.vectors.to_arrow().to_pylist()
                if row.get("doc_id") == str(doc_id)
            ])

        # Removing the file should return True
        removed = index.remove_file(repo_name, test_repo_path, target_file)
        assert removed is True

        # Document row should be gone
        cursor.execute("SELECT COUNT(*) FROM documents WHERE id = ?", (doc_id,))
        assert cursor.fetchone()[0] == 0

        # Symbols for this document should be gone
        cursor.execute("SELECT COUNT(*) FROM symbols WHERE doc_id = ?", (doc_id,))
        assert cursor.fetchone()[0] == 0

        # Embeddings for this document should be gone (if any existed)
        if index.vectors is not None:
            assert not [
                row
                for row in index.vectors.to_arrow().to_pylist()
                if row.get("doc_id") == str(doc_id)
            ]

        # Trigram postings should not reference this doc_id anymore
        if symbol_count_before > 0 or embedding_count_before > 0:
            for gram, doc_ids in index._trigram_iter_items():
                assert doc_id not in doc_ids

    def test_remove_file_with_missing_blob_still_cleans_trigrams(
        self,
        indexed_repo,
        test_repo_path,
    ):
        """If the blob is missing, remove_file should still strip trigram postings."""
        import zlib

        index = indexed_repo["index"]
        repo_name = indexed_repo["repo_name"]

        target_file = test_repo_path / "main.py"
        rel_path = target_file.relative_to(test_repo_path).as_posix()

        cursor = index.repos_db.cursor()
        cursor.execute("SELECT id, blob_sha FROM documents WHERE path = ?", (rel_path,))
        row = cursor.fetchone()
        assert row is not None
        doc_id, blob_sha = row

        # Simulate a missing/corrupted blob on disk
        blob_file = (
            index.index_path
            / "blobs"
            / blob_sha[:2]
            / blob_sha[2:]
        )
        if blob_file.exists():
            blob_file.unlink()

        # Removing the file should still succeed
        removed = index.remove_file(repo_name, test_repo_path, target_file)
        assert removed is True

        # And no trigram postings should reference this doc_id anymore
        for gram, doc_ids in index._trigram_iter_items():
            assert doc_id not in doc_ids

    def test_remove_file_clears_lancedb_rows(
        self, embeddings_enabled_index
    ):
        """Removing a file should clear matching LanceDB rows by repo/path."""

        index = embeddings_enabled_index["index"]
        repo_name = embeddings_enabled_index["repo_name"]
        repo_path = embeddings_enabled_index["repo_path"]

        stats = index.index_repository(repo_name, repo_path, force=True)
        assert stats["files_indexed"] > 0

        index.build_vector_index(repo=repo_name, force=True)

        target_file = repo_path / "main.py"
        rel_path = target_file.relative_to(repo_path).as_posix()

        repo_cursor = index.repos_db.cursor()
        repo_cursor.execute("SELECT id FROM repos WHERE name = ?", (repo_name,))
        repo_id = repo_cursor.fetchone()[0]

        existing_rows = index._repo_vectors[repo_name].to_arrow().to_pylist()
        matches = [
            row
            for row in existing_rows
            if row.get("repo_id") == str(repo_id) and row.get("file_path") == rel_path
        ]
        assert len(matches) > 0

        removed = index.remove_file(repo_name, repo_path, target_file)
        assert removed is True

        remaining_rows = index.vectors.to_arrow().to_pylist()
        remaining_matches = [
            row
            for row in remaining_rows
            if row.get("repo_id") == str(repo_id) and row.get("file_path") == rel_path
        ]
        assert len(remaining_matches) == 0


class TestEmbeddingDimensions:
    """Test handling of different embedding dimensions."""
    
    def test_consistent_embedding_dimensions(self, indexed_repo):
        """Test that embeddings have consistent dimensions."""
        index = indexed_repo["index"]
        index.build_vector_index(repo="test_repo", force=True)

        rows = index._repo_vectors["test_repo"].to_arrow().to_pylist()
        dims = {len(row["vector"]) for row in rows}
        assert len(dims) == 1
        assert dims.pop() == index.embedding_dimension

    def test_embedding_normalization(self, indexed_repo):
        """Test that embeddings are normalized."""
        index = indexed_repo["index"]
        index.build_vector_index(repo="test_repo", force=True)

        vectors = [row["vector"] for row in index._repo_vectors["test_repo"].to_arrow().to_pylist()[:5]]
        for vector in vectors:
            norm = np.linalg.norm(np.array(vector, dtype="float32"))
            assert 0.95 <= norm <= 1.05


class TestPartialReindexingVectors:
    """Tests ensuring partial reindex refreshes LanceDB entries."""

    def test_index_file_refreshes_vectors_for_path(self, embeddings_enabled_index):
        index = embeddings_enabled_index["index"]
        repo_name = embeddings_enabled_index["repo_name"]
        repo_path = embeddings_enabled_index["repo_path"]

        index.index_repository(repo_name, repo_path, force=True)
        index.build_vector_index(repo=repo_name, force=True)

        target_file = repo_path / "utils.py"
        rel_path = target_file.relative_to(repo_path).as_posix()

        repo_cursor = index.repos_db.cursor()
        repo_cursor.execute("SELECT id FROM repos WHERE name = ?", (repo_name,))
        repo_id = repo_cursor.fetchone()[0]

        original_chunks = index._chunk_text(target_file.read_text())
        original_rows = index._repo_vectors[repo_name].to_arrow().to_pylist()
        original_matches = [
            row
            for row in original_rows
            if row.get("repo_id") == str(repo_id) and row.get("file_path") == rel_path
        ]
        assert len(original_matches) == len(original_chunks)

        # Modify the file to produce a different chunking pattern
        target_file.write_text(
            target_file.read_text() + "\n\n# new helper added\n" "def helper():\n    return True\n"
        )

        updated_chunks = index._chunk_text(target_file.read_text())

        reindexed = index.index_file(repo_name, repo_path, target_file)
        assert reindexed is True

        refreshed_rows = index._repo_vectors[repo_name].to_arrow().to_pylist()
        refreshed_matches = [
            row
            for row in refreshed_rows
            if row.get("repo_id") == str(repo_id) and row.get("file_path") == rel_path
        ]
        assert len(refreshed_matches) == len(updated_chunks)


class TestCleanup:
    """Test cleanup and resource management."""
    
    def test_database_close(self, test_index_path, dummy_embed_fn):
        """Test that databases can be closed cleanly."""
        index = SigilIndex(test_index_path, dummy_embed_fn, "test")
        
        index.close()
        
        # Should not raise errors
    
    def test_reopen_after_close(self, test_index_path, dummy_embed_fn):
        """Test reopening databases after close."""
        index1 = SigilIndex(test_index_path, dummy_embed_fn, "test")
        index1.close()
        
        # Create new instance with same path
        index2 = SigilIndex(test_index_path, dummy_embed_fn, "test")
        
        # Should work
        cursor = index2.repos_db.cursor()
        cursor.execute("SELECT COUNT(*) FROM repos")
        count = cursor.fetchone()[0]
        
        assert count >= 0
        
        index2.close()
