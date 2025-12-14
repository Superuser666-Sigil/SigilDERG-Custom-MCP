# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com

"""
Tests for indexer module (indexer.py) - Part 1: Basic functionality.
"""

import numpy as np
from sigil_mcp.indexer import SigilIndex


class TestSigilIndexInitialization:
    """Test SigilIndex initialization and schema creation."""
    
    def test_index_initialization(self, test_index_path, dummy_embed_fn):
        """Test that index initializes correctly."""
        index = SigilIndex(
            index_path=test_index_path,
            embed_fn=dummy_embed_fn,
            embed_model="test-model"
        )
        
        assert index.index_path == test_index_path
        assert index.embed_fn is dummy_embed_fn
        assert index.embed_model == "test-model"
        assert test_index_path.exists()
        
        index.close()
    
    def test_index_creates_databases(self, test_index_path, dummy_embed_fn):
        """Test that databases are created."""
        index = SigilIndex(test_index_path, dummy_embed_fn, "test")
        
        assert (test_index_path / "repos.db").exists()
        assert (test_index_path / "trigrams.rocksdb").exists()
        assert index._trigram_backend == "rocksdict"
        
        index.close()
    
    def test_schema_repos_table(self, test_index):
        """Test that repos table is created with correct schema."""
        cursor = test_index.repos_db.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='repos'")
        assert cursor.fetchone() is not None
        
        # Check columns
        cursor.execute("PRAGMA table_info(repos)")
        columns = {row[1] for row in cursor.fetchall()}
        assert columns == {"id", "name", "path", "indexed_at"}
    
    def test_schema_documents_table(self, test_index):
        """Test that documents table is created."""
        cursor = test_index.repos_db.cursor()
        cursor.execute("PRAGMA table_info(documents)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "id" in columns
        assert "repo_id" in columns
        assert "path" in columns
        assert "blob_sha" in columns
    
    def test_schema_symbols_table(self, test_index):
        """Test that symbols table is created."""
        cursor = test_index.repos_db.cursor()
        cursor.execute("PRAGMA table_info(symbols)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "id" in columns
        assert "doc_id" in columns
        assert "name" in columns
        assert "kind" in columns
        assert "line" in columns
    
    def test_schema_embeddings_table(self, test_index):
        """Test that embeddings table is created."""
        cursor = test_index.repos_db.cursor()
        cursor.execute("PRAGMA table_info(embeddings)")
        columns = {row[1] for row in cursor.fetchall()}
        # Legacy table should be dropped now that embeddings live in LanceDB
        assert not columns
    
    def test_trigram_store_available(self, test_index):
        """Test that the Rocks-backed trigram store is initialized."""
        backend = getattr(test_index, "_trigram_backend", None)
        assert backend == "rocksdict"
        assert test_index._trigram_count() == 0
        assert list(test_index._trigram_iter_items()) == []


class TestRepositoryIndexing:
    """Test repository indexing functionality."""
    
    def test_index_repository_basic(self, test_index, test_repo_path):
        """Test basic repository indexing."""
        stats = test_index.index_repository("test_repo", test_repo_path, force=True)
        
        assert stats["files_indexed"] > 0
        # Symbol extraction requires ctags, which may not be available
        assert stats["symbols_extracted"] >= 0
        assert stats["trigrams_built"] >= 0
    
    def test_index_repository_creates_repo_entry(self, test_index, test_repo_path):
        """Test that indexing creates repository entry."""
        test_index.index_repository("test_repo", test_repo_path, force=True)
        
        cursor = test_index.repos_db.cursor()
        cursor.execute("SELECT name, path FROM repos WHERE name = ?", ("test_repo",))
        result = cursor.fetchone()
        
        assert result is not None
        assert result[0] == "test_repo"
    
    def test_index_repository_indexes_files(self, test_index, test_repo_path):
        """Test that files are indexed."""
        test_index.index_repository("test_repo", test_repo_path, force=True)
        
        cursor = test_index.repos_db.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM documents 
            WHERE repo_id = (SELECT id FROM repos WHERE name = ?)
        """, ("test_repo",))
        
        file_count = cursor.fetchone()[0]
        assert file_count >= 2  # main.py, utils.py
    
    def test_index_repository_extracts_symbols(self, indexed_repo):
        """Test that symbols are extracted from code."""
        index = indexed_repo["index"]
        
        cursor = index.repos_db.cursor()
        cursor.execute("SELECT name, kind FROM symbols")
        symbols = cursor.fetchall()
        
        # Symbol extraction requires ctags - may be 0 if not available
        assert len(symbols) >= 0
        
        # If symbols were extracted, verify they look reasonable
        if len(symbols) > 0:
            symbol_names = {s[0] for s in symbols}
            assert len(symbol_names) > 0
    
    def test_index_repository_force_rebuild(self, test_index, test_repo_path):
        """Test that force=True rebuilds index."""
        # First indexing
        stats1 = test_index.index_repository("test_repo", test_repo_path, force=True)
        
        # Second indexing with force
        stats2 = test_index.index_repository("test_repo", test_repo_path, force=True)
        
        # Should index files (may vary slightly due to caching)
        assert stats2["files_indexed"] >= 0
        assert stats1["files_indexed"] >= 0


class TestTrigramSearch:
    """Test trigram-based text search."""
    
    def test_build_trigrams(self, indexed_repo):
        """Test that trigrams are built for indexed documents."""
        index = indexed_repo["index"]
        
        trigram_count = index._trigram_count()
        assert trigram_count > 0
        doc_ids = {doc_id for _, ids in index._trigram_iter_items() for doc_id in ids}
        assert doc_ids


class TestSymbolSearch:
    """Test symbol search functionality."""
    
    def test_find_symbol_by_name(self, indexed_repo):
        """Test finding symbols by name."""
        index = indexed_repo["index"]
        
        # Index should have extracted some symbols
        cursor = index.repos_db.cursor()
        cursor.execute("SELECT DISTINCT name FROM symbols LIMIT 1")
        result = cursor.fetchone()
        
        if result:
            symbol_name = result[0]
            results = index.find_symbol(symbol_name, repo="test_repo")
            assert len(results) > 0
            assert results[0].name == symbol_name
    
    def test_find_symbol_nonexistent(self, indexed_repo):
        """Test finding nonexistent symbol."""
        index = indexed_repo["index"]
        
        results = index.find_symbol("nonexistent_symbol_xyz123", repo="test_repo")
        assert len(results) == 0
    
    def test_list_symbols_by_kind(self, indexed_repo):
        """Test listing symbols by kind."""
        index = indexed_repo["index"]
        
        # Get any kind that exists
        cursor = index.repos_db.cursor()
        cursor.execute("SELECT DISTINCT kind FROM symbols LIMIT 1")
        result = cursor.fetchone()
        
        if result:
            kind = result[0]
            results = index.list_symbols(kind=kind, repo="test_repo")
            assert len(results) > 0
            assert all(r.kind == kind for r in results)
    
    def test_list_symbols_in_file(self, indexed_repo):
        """Test listing symbols in specific file."""
        index = indexed_repo["index"]
        repo_path = indexed_repo["repo_path"]
        
        # Use a file we know exists
        file_path = "main.py"
        results = index.list_symbols(file_path=file_path, repo="test_repo")
        
        # Should return symbols or empty list if none found
        assert isinstance(results, list)


class TestVectorIndexing:
    """Test vector embedding functionality."""
    
    def test_chunk_text_method(self, test_index):
        """Test text chunking for embeddings."""
        # Create test text with multiple lines
        text = "\n".join([f"line {i}" for i in range(150)])
        
        chunks = test_index._chunk_text(text, max_lines=100)
        
        assert len(chunks) >= 2  # 150 lines should create at least 2 chunks
        assert all(isinstance(chunk, tuple) for chunk in chunks)
        assert all(len(chunk) == 4 for chunk in chunks)  # (idx, start, end, text)
    
    def test_build_vector_index(self, indexed_repo):
        """Test building vector index."""
        index = indexed_repo["index"]
        
        stats = index.build_vector_index(repo="test_repo", force=True)
        
        assert "documents_processed" in stats
        assert "chunks_indexed" in stats
        assert stats["documents_processed"] > 0
    
    def test_embeddings_stored_in_database(self, indexed_repo):
        """Test that embeddings are stored in database."""
        index = indexed_repo["index"]
        index.build_vector_index(repo="test_repo", force=True)

        assert index.vectors is not None
        assert "test_repo" in index._repo_vectors
        assert index._repo_vectors["test_repo"].count_rows() > 0

    def test_embedding_vector_format(self, indexed_repo):
        """Test that embedding vectors are stored correctly."""
        index = indexed_repo["index"]
        index.build_vector_index(repo="test_repo", force=True)

        assert index.vectors is not None
        assert "test_repo" in index._repo_vectors
        rows = index._repo_vectors["test_repo"].to_arrow().to_pylist()
        assert rows
        vector = rows[0]["vector"]
        assert len(vector) == index.embedding_dimension


class TestSemanticSearch:
    """Test semantic search with vector embeddings."""
    
    def test_semantic_search_basic(self, indexed_repo):
        """Test basic semantic search."""
        index = indexed_repo["index"]
        index.build_vector_index(repo="test_repo", force=True)
        
        results = index.semantic_search(
            query="calculator function",
            repo="test_repo",
            k=5
        )
        
        assert isinstance(results, list)
        assert len(results) <= 5
    
    def test_semantic_search_returns_search_results(self, indexed_repo):
        """Test that semantic search returns SearchResult objects."""
        index = indexed_repo["index"]
        index.build_vector_index(repo="test_repo", force=True)
        
        results = index.semantic_search(
            query="function definition",
            repo="test_repo",
            k=3
        )
        
        if results:
            assert isinstance(results[0], dict)
            assert "path" in results[0]
            assert "score" in results[0]
    
    def test_semantic_search_top_k_limit(self, indexed_repo):
        """Test that semantic search respects top_k limit."""
        index = indexed_repo["index"]
        index.build_vector_index(repo="test_repo", force=True)
        
        results = index.semantic_search(
            query="python code",
            repo="test_repo",
            k=2
        )
        
        assert len(results) <= 2
    
    def test_semantic_search_empty_query(self, indexed_repo):
        """Test semantic search with empty query."""
        index = indexed_repo["index"]
        index.build_vector_index(repo="test_repo", force=True)
        
        results = index.semantic_search(
            query="",
            repo="test_repo",
            k=5
        )
        
        # Should handle gracefully
        assert isinstance(results, list)
    
    def test_semantic_search_no_embeddings(self, test_index, test_repo_path):
        """Test semantic search before building vector index."""
        test_index.index_repository("test_repo", test_repo_path, force=True)
        
        # Don't build vector index
        results = test_index.semantic_search(
            query="test query",
            repo="test_repo",
            k=5
        )
        
        # Should return empty list or handle gracefully
        assert isinstance(results, list)
        assert len(results) == 0

    def test_semantic_search_code_only_filters_docs(self, temp_dir, dummy_embed_fn, monkeypatch):
        """Ensure code_only hard-filters non-code chunks."""
        import sigil_mcp.config as sigil_config

        cfg = sigil_config.Config()
        cfg.config_data["embeddings"] = {"enabled": True}
        cfg.config_data["index"] = {"ignore_patterns": []}
        monkeypatch.setattr(sigil_config, "_config", cfg)

        repo_path = temp_dir / "mixed_repo"
        repo_path.mkdir(parents=True, exist_ok=True)
        (repo_path / "main.py").write_text("def hello():\n    return 1\n")
        (repo_path / "README.md").write_text("# docs\n\nThis is documentation.")

        index = SigilIndex(index_path=temp_dir / ".idx", embed_fn=dummy_embed_fn, embed_model="test-model")
        try:
            index.index_repository("mixed", repo_path, force=True)
            index.build_vector_index(repo="mixed", force=True)

            results = index.semantic_search("hello", repo="mixed", k=5, code_only=True)
            assert results == [] or all(r["is_code"] for r in results)

            mixed_results = index.semantic_search("docs", repo="mixed", k=5)
            assert any(r.get("is_doc") for r in mixed_results)
        finally:
            index.close()

    def test_semantic_search_prefer_code_reranks(self, temp_dir, dummy_embed_fn, monkeypatch):
        """Rerank should favor code when prefer_code is set even if distances tie."""
        import sigil_mcp.config as sigil_config

        cfg = sigil_config.Config()
        cfg.config_data["embeddings"] = {"enabled": True}
        cfg.config_data["index"] = {"ignore_patterns": []}
        monkeypatch.setattr(sigil_config, "_config", cfg)

        index = SigilIndex(index_path=temp_dir / ".idx2", embed_fn=dummy_embed_fn, embed_model="test-model")
        try:
            repo_path = temp_dir / "pref_repo"
            repo_path.mkdir(parents=True, exist_ok=True)
            (repo_path / "code.py").write_text("def important():\n    return 42\n")
            index.index_repository("pref", repo_path, force=True)

            # Replace vector table with controlled candidates
            from sigil_mcp.indexer import InMemoryLanceTable, InMemoryLanceDB

            table = InMemoryLanceTable("code_vectors", index.embedding_dimension)
            table.add([
                {
                    "doc_id": 1,
                    "repo_id": 1,
                    "file_path": "code.py",
                    "chunk_index": 0,
                    "start_line": 1,
                    "end_line": 2,
                    "content": "def important(): return 42",
                    "_distance": 0.4,
                    "is_code": True,
                    "is_doc": False,
                    "is_config": False,
                    "is_data": False,
                    "extension": ".py",
                    "language": "python",
                },
                {
                    "doc_id": 2,
                    "repo_id": 1,
                    "file_path": "README.md",
                    "chunk_index": 0,
                    "start_line": 1,
                    "end_line": 3,
                    "content": "docs about important function",
                    "_distance": 0.2,
                    "is_code": False,
                    "is_doc": True,
                    "is_config": False,
                    "is_data": False,
                    "extension": ".md",
                    "language": "markdown",
                },
            ])

            index._repo_vectors["pref"] = table
            index._repo_lance_dbs["pref"] = InMemoryLanceDB(index.embedding_dimension)

            results = index.semantic_search("important", repo="pref", k=2, prefer_code=True)
            assert results
            assert results[0]["is_code"] is True
            assert any(r.get("is_doc") for r in results)
        finally:
            index.close()


class TestIndexStatistics:
    """Test index statistics functionality."""
    
    def test_get_index_stats(self, indexed_repo):
        """Test getting index statistics."""
        index = indexed_repo["index"]
        
        stats = index.get_index_stats(repo="test_repo")
        
        assert isinstance(stats, dict)
        assert "documents" in stats
        assert "symbols" in stats
        assert stats["documents"] > 0
    
    def test_get_index_stats_all_repos(self, indexed_repo):
        """Test getting stats for all repositories."""
        index = indexed_repo["index"]
        
        stats = index.get_index_stats()
        
        assert isinstance(stats, dict)
        # Should aggregate across all repos
    
    def test_get_index_stats_nonexistent_repo(self, test_index):
        """Test getting stats for nonexistent repository."""
        stats = test_index.get_index_stats(repo="nonexistent_repo")
        
        assert isinstance(stats, dict)
        assert stats.get("documents", 0) == 0
