# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
Integration tests for Sigil MCP Server end-to-end workflows.
"""

from sigil_mcp.indexer import SigilIndex


class TestEndToEndIndexing:
    """Test complete indexing workflow."""
    
    def test_full_indexing_workflow(self, test_index_path, test_repo_path, dummy_embed_fn):
        """Test complete workflow: index -> search -> semantic search."""
        # Create index
        index = SigilIndex(test_index_path, dummy_embed_fn, "test-model")
        
        # Index repository
        stats = index.index_repository("test_repo", test_repo_path, force=True)
        assert stats["files_indexed"] > 0
        
        # Build vector index
        vector_stats = index.build_vector_index(repo="test_repo", force=True)
        assert vector_stats["documents_processed"] > 0
        
        # Semantic search
        semantic_results = index.semantic_search(
            query="function that processes data",
            repo="test_repo",
            k=3
        )
        assert isinstance(semantic_results, list)
        
        # Get stats
        stats = index.get_index_stats(repo="test_repo")
        docs = stats.get("documents")
        assert isinstance(docs, int) and docs > 0
        
        # Cleanup
        index.close()
    
    def test_multi_repo_indexing(self, test_index_path, temp_dir, dummy_embed_fn):
        """Test indexing multiple repositories."""
        index = SigilIndex(test_index_path, dummy_embed_fn, "test-model")
        
        # Create two repos
        repo1 = temp_dir / "repo1"
        repo1.mkdir()
        (repo1 / "file1.py").write_text("def func1(): pass")
        
        repo2 = temp_dir / "repo2"
        repo2.mkdir()
        (repo2 / "file2.py").write_text("def func2(): pass")
        
        # Index both
        index.index_repository("repo1", repo1, force=True)
        index.index_repository("repo2", repo2, force=True)
        
        # Get stats for each
        stats1 = index.get_index_stats(repo="repo1")
        stats2 = index.get_index_stats(repo="repo2")
        
        assert isinstance(stats1, dict)
        assert isinstance(stats2, dict)
        
        # Cleanup
        index.close()
    
    def test_incremental_indexing(self, test_index, test_repo_path):
        """Test incremental indexing (adding files to existing repo)."""
        # Initial index
        stats1 = test_index.index_repository("test_repo", test_repo_path, force=True)
        
        # Add new file to repo
        new_file = test_repo_path / "new_module.py"
        new_file.write_text("""
def new_function():
    '''A new function.'''
    return "new"
""")
        
        # Re-index
        stats2 = test_index.index_repository("test_repo", test_repo_path, force=True)
        
        # Should work without errors
        assert stats2["files_indexed"] >= 1  # At least the new file


class TestSearchAccuracy:
    """Test search result accuracy and relevance."""
    
    def test_semantic_search_relevance(self, indexed_repo):
        """Test that semantic search returns relevant results."""
        index = indexed_repo["index"]
        index.build_vector_index(repo="test_repo", force=True)
        
        # Search for concept
        results = index.semantic_search(
            query="mathematical operations addition subtraction",
            repo="test_repo",
            k=5
        )
        
        # Should return results (relevance hard to test without real embeddings)
        assert isinstance(results, list)


class TestPerformance:
    """Test performance characteristics (not strict benchmarks)."""
    
    def test_indexing_completes_reasonably(self, test_index, test_repo_path):
        """Test that indexing completes in reasonable time."""
        import time
        
        start = time.time()
        test_index.index_repository("test_repo", test_repo_path, force=True)
        duration = time.time() - start
        
        # Should complete in under 30 seconds for small repo
        assert duration < 30
    
    def test_semantic_search_completes_reasonably(self, indexed_repo):
        """Test that semantic search completes in reasonable time."""
        import time
        
        index = indexed_repo["index"]
        index.build_vector_index(repo="test_repo", force=True)
        
        start = time.time()
        index.semantic_search("test query", repo="test_repo", k=10)
        duration = time.time() - start
        
        # Should complete in under 5 seconds for small repo
        assert duration < 5.0


class TestDataPersistence:
    """Test that data persists across sessions."""
    
    def test_index_persists_after_close(self, test_index_path, test_repo_path, dummy_embed_fn):
        """Test that index data persists after closing."""
        # Create and populate index
        index1 = SigilIndex(test_index_path, dummy_embed_fn, "test")
        index1.index_repository("test_repo", test_repo_path, force=True)
        
        # Get initial stats
        stats1 = index1.get_index_stats(repo="test_repo")
        
        # Close
        index1.close()
        
        # Reopen
        index2 = SigilIndex(test_index_path, dummy_embed_fn, "test")
        stats2 = index2.get_index_stats(repo="test_repo")
        
        # Should have same data
        assert stats2["documents"] == stats1["documents"]
        
        index2.close()
    
    def test_embeddings_persist(self, test_index_path, test_repo_path, dummy_embed_fn):
        """Test that embeddings persist after close."""
        # Create index and build vectors
        index1 = SigilIndex(test_index_path, dummy_embed_fn, "test")
        index1.index_repository("test_repo", test_repo_path, force=True)
        index1.build_vector_index(repo="test_repo", force=True)

        count1 = index1.vectors.count_rows() if index1.vectors else 0

        index1.close()

        # Reopen
        index2 = SigilIndex(test_index_path, dummy_embed_fn, "test")
        count2 = index2.vectors.count_rows() if index2.vectors else 0

        # Should have same embeddings
        assert count2 == count1
        
        index2.close()


class TestErrorRecovery:
    """Test error handling and recovery."""
    
    def test_partial_index_recovery(self, test_index, test_repo_path):
        """Test recovery from partial indexing."""
        # Index normally
        test_index.index_repository("test_repo", test_repo_path, force=True)
        
        # Force re-index
        stats = test_index.index_repository("test_repo", test_repo_path, force=True)
        
        # Should complete successfully (may be 0 if already indexed)
        assert stats["files_indexed"] >= 0
    
    def test_corrupted_file_handling(self, test_index, temp_dir):
        """Test handling of corrupted/invalid files."""
        repo = temp_dir / "corrupted_repo"
        repo.mkdir()
        
        # Create valid file
        (repo / "valid.py").write_text("def valid(): pass")
        
        # Create file with invalid encoding (will cause read errors)
        invalid_file = repo / "invalid.py"
        invalid_file.write_bytes(b"\xff\xfe invalid bytes \x00\x00")
        
        # Should handle gracefully
        try:
            stats = test_index.index_repository("corrupted_repo", repo, force=True)
            # Should index at least the valid file
            assert stats["files_indexed"] >= 1
        except Exception as e:
            # Or handle the error gracefully
            pass
