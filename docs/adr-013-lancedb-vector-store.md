<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# ADR-013: LanceDB Vector Store for Semantic Search

**Status:** Accepted
**Date:** 2025-12-07

## Context

The existing vector index for semantic code search stores embeddings inside the `repos.db` SQLite database as BLOB columns. This approach has several issues:

- SQLite does not provide approximate nearest-neighbor search, leading to full-scan cosine similarity calculations.
- Concurrent readers and writers compete for database locks when embedding generation, file watching, and user queries overlap.
- Large binary blobs inflate the SQLite file, making replication and backups expensive.
- Updating or deleting embeddings requires custom reference counting and manual blob packing.
- Exporting and inspecting vectors (for quality checks or migrations) is cumbersome without a columnar format.

We need a vector-native store that supports ANN indices, efficient append/update semantics, and columnar inspection while remaining embeddable for per-repo indexes.

## Decision

Adopt **LanceDB** as the primary vector store for semantic search embeddings. The system will:

1. Create a `lancedb` directory under each repository's index path and connect to it on startup.
2. Store vectors in a `code_vectors` table with columns for repo name, file path, chunk metadata, model, dimension, and the embedding vector.
3. Build a product-quantization (PQ) index on the vector column to enable fast ANN queries with configurable recall/latency trade-offs.
4. Upsert chunks by `doc_id`, `chunk_index`, and `model` so re-indexing replaces stale embeddings without full rebuilds.
5. Keep SQLite (`repos.db`) for documents and symbols; trigram text search uses rocksdict (RocksDB bindings); only the vector store moves to LanceDB.
6. Provide migration utilities to copy existing SQLite-resident embeddings into LanceDB while preserving chunk boundaries and metadata.

## Consequences

### Positive

- ANN-backed queries reduce latency for semantic search compared to full-scan SQLite blobs.
- Columnar storage (Arrow) simplifies debugging, exporting, and validating embeddings.
- Upserts and delete-by-filter operations avoid bespoke garbage collection of binary blobs.
- LanceDB is embeddable and filesystem-backed, keeping deployment parity with current per-repo indexes.
- PQ indexes balance accuracy and performance without external services.

### Negative

- Introduces a new dependency (LanceDB/Arrow) and larger local binary footprint.
- Index files are separate from SQLite backups, so operators must include the `lancedb` directory in snapshots.
- Requires a migration step for existing repositories before LanceDB-backed queries are available.

### Neutral

- SQLite remains in use for metadata, so operational patterns (WAL, busy timeouts) stay the same.
- Vector search APIs remain unchanged for clients; the storage backend swap is internal.

## Alternatives Considered

### Alternative 1: Continue Using SQLite Blobs

- Would avoid adding dependencies but keeps full-scan queries, heavy BLOB growth, and lock contention.

### Alternative 2: PostgreSQL with pgvector

- Provides strong ANN support and transactional semantics but requires an external service and network connectivity, reducing ease of local deployment.

### Alternative 3: FAISS Flat or HNSW Indexes

- Strong performance but would require custom on-disk management and metadata handling that LanceDB already provides.

## Related

- [ADR-006: Vector Embeddings for Semantic Code Search](adr-006-vector-embeddings.md)
- [ADR-010: Thread Safety for SQLite Indexer](adr-010-thread-safety-sqlite.md)
