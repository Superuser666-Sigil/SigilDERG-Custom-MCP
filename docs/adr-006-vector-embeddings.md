<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# ADR-006: Vector Embeddings for Semantic Code Search

## Status

Superseded by [ADR-013: LanceDB Vector Store](adr-013-lancedb-vector-store.md)

> Note: Default provider/model selection is superseded by ADR-015 (llamacpp + Jina GGUF). This ADR still governs the embedding architecture and provider options.

## Context

While trigram-based text search (ADR-002) and symbol search (ADR-003) work well for exact substring matching and navigating to definitions, they cannot handle semantic queries where users describe what code does rather than exact text to match:

- "Find functions that parse JSON"
- "Locate authentication handlers"
- "Search for database connection setup"
- "Find error handling middleware"

Users interacting with AI assistants via MCP often make high-level requests that benefit from semantic understanding. Traditional keyword search fails when:

- Different terminology is used (e.g., "authenticate" vs "login")
- Code intent isn't obvious from identifiers alone
- Comments describe functionality differently than queries
- Cross-file patterns need conceptual understanding

Modern embedding models (BERT, CodeBERT, sentence-transformers) can encode code semantics into fixed-dimension vectors, enabling cosine similarity search for "meaning-based" matching.

## Decision

Extend `SigilIndex` with optional vector embedding capabilities:

1. **Schema Extension**: Add `embeddings` table to existing `repos.db`
   - Store embeddings as BLOBs keyed by `(doc_id, chunk_index, model)`
   - Track line ranges per chunk for precise result location
   - Support multiple embedding models simultaneously

2. **Chunking Strategy**: Split documents into ~100-line overlapping chunks
   - Maintains context for better embeddings
   - 10-line overlap prevents boundary issues
   - Line tracking enables jumping to exact locations

3. **Configuration-Based Provider Selection**: Support multiple embedding providers via config
   - **sentence-transformers**: Local models with GPU/CPU support
   - **OpenAI**: Cloud API for highest quality embeddings
   - **llamacpp**: GGUF models with multi-platform acceleration
   - Configuration properties:
     * `embeddings.enabled`: Enable/disable embedding features
     * `embeddings.provider`: Provider name (sentence-transformers, openai, llamacpp)
     * `embeddings.model`: Model identifier or path
     * `embeddings.dimension`: Expected embedding dimension
     * `embeddings.cache_dir`: Cache directory for downloaded models (optional)
     * `embeddings.api_key`: API key for cloud providers (optional, falls back to env vars)

4. **Pluggable Provider Architecture**: Factory pattern for provider instantiation
   - Type: `EmbeddingProvider` protocol with `embed_documents()` and `embed_query()` methods
   - Providers wrap external libraries (sentence_transformers, openai, llama_cpp_python)
   - Graceful handling of missing optional dependencies
   - Top-level imports with availability flags for testability

5. **Persistent Storage**: Vectors stored as float32 BLOBs in SQLite
   - Compressed blob storage already exists for file content
   - Metadata tracks model version for compatibility
   - Dimension stored per embedding for validation

5. **Incremental Building**: `build_vector_index(repo, force=False)` 
   - Respects existing embeddings unless `force=True`
   - Processes documents already indexed by trigram/symbol system
   - Returns statistics about chunks indexed

6. **Co-existence with Existing Indexes**: Treat embeddings as additional channel
   - Trigram index unchanged, still handles substring search
   - Symbol index unchanged, still handles definitions/outline
   - Semantic search complements rather than replaces

7. **Semantic Search Implementation**: Brute-force cosine similarity search
   - In-memory computation using NumPy for typical repo sizes
   - Query embedding compared against all chunk embeddings
   - Top-k results returned with scores and line ranges
   - O(N) complexity suitable for small-to-medium repos (1K-10K chunks)

## Implementation Details

### Configuration Example

```json
{
  "embeddings": {
    "enabled": true,
    "provider": "sentence-transformers",
    "model": "all-MiniLM-L6-v2",
    "dimension": 384,
    "cache_dir": "~/.cache/sigil-embeddings"
  }
}
```

### Provider Initialization

The server's `_create_embedding_function()` reads configuration and initializes the appropriate provider:

1. Checks `embeddings.enabled` flag
2. Validates provider and model configuration
3. Calls `create_embedding_provider()` from `sigil_mcp.embeddings` module
4. Wraps provider in numpy-returning function for compatibility with SigilIndex
5. Returns `(embed_fn, model_name)` tuple or `(None, None)` if disabled

### Python 3.12 Best Practices

The embedding provider module follows modern Python patterns:
- Optional dependencies imported at module level with availability flags
- No conditional imports inside functions (anti-pattern)
- TYPE_CHECKING used for type hints to avoid circular imports
- Graceful ImportError handling with clear error messages

## Consequences

### Positive

- **Semantic queries**: Users can describe what they're looking for conceptually
- **AI-friendly**: Perfect for MCP servers serving ChatGPT/Claude/etc.
- **Flexible models**: Swap embedding backends without schema changes
- **Minimal dependencies**: Core only requires numpy; embedding libs are optional
- **Reuses infrastructure**: Leverages existing `repos.db` and blob storage
- **Line-level precision**: Chunk metadata enables jumping to exact code locations
- **Multi-model support**: Can index same repo with multiple models
- **Incremental updates**: Only re-embed changed documents

### Negative

- **Storage overhead**: Embeddings add ~1.5KB per chunk (384-dim float32)
- **Build time**: Embedding generation is slower than trigram indexing
- **External dependencies**: Requires embedding model (OpenAI API, local model, etc.)
- **No vector DB yet**: Initial implementation uses brute-force cosine search
- **Memory scaling**: Loading all vectors into memory for similarity search
- **Model compatibility**: Changing models requires rebuilding embeddings

### Neutral

- Embeddings stored alongside existing data in `repos.db`
- Semantic search implemented using brute-force cosine similarity
- Default initialization without `embed_fn` maintains backward compatibility
- Chunks at ~100 lines balance context vs. granularity
- Model identifier stored for tracking/versioning
- MCP tools exposed: `build_vector_index` and `semantic_search`

## Alternatives Considered

### Alternative 1: External Vector Database (Pinecone, Weaviate, Qdrant)

Use dedicated vector database for embedding storage and search.

**Rejected because:**
- Adds external service dependency
- Complicates deployment and configuration
- Overkill for single-user local indexing
- Network latency for local searches
- Cost implications for cloud-hosted services
- Can add later if scaling requires it

### Alternative 2: Hardcode OpenAI Embeddings

Build with OpenAI API directly integrated.

**Rejected because:**
- Vendor lock-in
- Requires API key and internet connectivity
- Cost per embedding (though small)
- Users may prefer local models for privacy
- Sentence-transformers can run offline
- Pluggable approach supports all options

### Alternative 3: Whole-File Embeddings

Embed entire files as single vectors.

**Rejected because:**
- Large files lose granularity (can't pinpoint relevant section)
- Context window limits on embedding models
- Poor results for files with mixed content
- Can't return specific line numbers
- Chunking proven effective in RAG systems

### Alternative 4: Store in Separate Vector Index File

Create dedicated `.vec` files or separate database.

**Rejected because:**
- Complicates schema management
- Harder to maintain consistency with document metadata
- SQLite BLOBs perform adequately for brute-force search
- Existing `repos.db` already has document metadata
- Single database simpler for transactions

### Alternative 5: Use FAISS for Fast Similarity Search

Integrate Facebook's FAISS library for ANN search.

**Deferred because:**
- Adds significant dependency
- C++ library with complex installation
- Not needed for small-to-medium repos (brute-force is fast enough)
- Can add later if performance bottleneck emerges
- Premature optimization for initial implementation

## Related
- [ADR-002: Trigram Indexing](adr-002-trigram-indexing.md) - Complementary substring search
- [ADR-003: Symbol Search](adr-003-symbol-search-ctags.md) - Complementary symbol navigation
- [Vector Embeddings Usage Guide](VECTOR_EMBEDDINGS.md) - Implementation details and examples
- [Sentence-Transformers](https://www.sbert.net/) - Popular open-source embedding library
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings) - Cloud embedding API
- [Code search papers](https://arxiv.org/abs/2002.08653) - CodeBERT and related work53) - CodeBERT and related work
- Future ADR: Semantic search API and query interface
