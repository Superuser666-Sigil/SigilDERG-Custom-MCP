<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# Vector Embeddings for Semantic Code Search

This document describes how to use vector embeddings in Sigil MCP Server for semantic code search capabilities.

## Overview

Vector embeddings enable semantic code search, allowing users to find code based on what it does rather than exact text matching. For example:

- "Find functions that validate user input"
- "Locate database connection setup"
- "Search for error handling middleware"

While trigram search (substring matching) and symbol search (definitions) are great for exact queries, embeddings understand meaning and context.

## Architecture

### Storage

Embeddings are stored in a LanceDB-backed `code_vectors` table under each repository's index directory (see [ADR-013](adr-013-lancedb-vector-store.md) for the rationale). The table tracks repository name, file path, chunk metadata, embedding model, dimension, and the vector itself. A product-quantization index on the vector column accelerates approximate-nearest-neighbor queries while keeping the on-disk footprint manageable. The files live inside your index path at `index_dir/lancedb/<repo_name>/code_vectors`.

Legacy repositories created before the LanceDB migration may still have embeddings in the `repos.db` SQLite table shown below. Run the migration tooling to move these rows into LanceDB if you need ANN-backed queries, then drop the legacy table (legacy installs only) to avoid double-counting and free disk space: `sqlite3 ~/.sigil_index/repos.db "DROP TABLE IF EXISTS embeddings;"`.

```sql
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL,
    chunk_index INTEGER NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    model TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    FOREIGN KEY(doc_id) REFERENCES documents(id),
    UNIQUE(doc_id, chunk_index, model)
)
```

### Chunking

Documents are split into overlapping chunks:
- **Default chunk size**: 100 lines
- **Overlap**: 10 lines between chunks
- **Line tracking**: Each chunk stores `start_line` and `end_line`

This approach:
- Maintains context for better embeddings
- Prevents boundary issues with overlap
- Enables precise navigation to results

### Embedding Models

The system accepts any embedding function with signature:
```python
EmbeddingFn = Callable[[Sequence[str]], np.ndarray]
```

The function takes a list of text strings and returns a numpy array of shape `(N, dim)` where:
- `N` = number of input texts
- `dim` = embedding dimension (384, 768, 1536, etc.)

## Usage

### 1. Initialize Index with Embedding Function

```python
from pathlib import Path
from sigil_mcp.indexer import SigilIndex
import numpy as np

# Option A: Dummy function for testing
def dummy_embed_fn(texts):
    dim = 384
    embeddings = np.random.randn(len(texts), dim).astype('float32')
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)

index = SigilIndex(
    index_path=Path.home() / ".sigil-index",
    embed_fn=dummy_embed_fn,
    embed_model="dummy-v1"
)
```

### 2. Index Repository (Trigrams + Symbols)

```python
# First index the repository normally
stats = index.index_repository(
    repo_name="my-project",
    repo_path=Path("/path/to/repo"),
    force=False
)
print(f"Indexed {stats['files_indexed']} files")
```

### 3. Build Vector Index

```python
# Generate and store embeddings
vector_stats = index.build_vector_index(
    repo="my-project",
    force=False  # Set True to rebuild
)

print(f"Chunks indexed: {vector_stats['chunks_indexed']}")
print(f"Documents processed: {vector_stats['documents_processed']}")
```

### 4. Query with Semantic Search

Search for code by describing what it does:

```python
# Semantic search for authentication code
results = index.semantic_search(
    query="user authentication and login handlers",
    repo="my-project",
    k=10  # top 10 results
)

for result in results:
    print(f"{result['path']}:{result['start_line']}-{result['end_line']}")
    print(f"  Score: {result['score']:.3f}")
```

## Embedding Model Options

### Option 1: Sentence-Transformers (Recommended for Local)

**Pros**: Open-source, runs locally, no API costs, privacy-friendly

**Setup**:
```bash
pip install sentence-transformers
```

**Usage**:
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def embed_fn(texts):
    return model.encode(texts, show_progress_bar=False)

index = SigilIndex(
    index_path=Path.home() / ".sigil-index",
    embed_fn=embed_fn,
    embed_model="all-MiniLM-L6-v2"
)
```

**Recommended models** (optimized for the LanceDB ANN index):
- `all-MiniLM-L12-v2`: Balanced quality/speed, **768-dim** (current default)
- `all-mpnet-base-v2`: Higher quality, 768-dim, slower
- `paraphrase-multilingual-MiniLM-L12-v2`: Multi-language support

### Option 2: OpenAI Embeddings

**Pros**: High quality, latest models, no local compute

**Cons**: API costs, requires internet, privacy considerations

**Setup**:
```bash
pip install openai
export OPENAI_API_KEY="sk-..."
```

**Usage**:
```python
import openai
import os
import numpy as np

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def embed_fn(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([
        item.embedding for item in response.data
    ], dtype='float32')

index = SigilIndex(
    index_path=Path.home() / ".sigil-index",
    embed_fn=embed_fn,
    embed_model="text-embedding-3-small"
)
```

**Available models**:
- `text-embedding-3-small`: 1536-dim, $0.02/1M tokens
- `text-embedding-3-large`: 3072-dim, $0.13/1M tokens
- `text-embedding-ada-002`: Legacy, 1536-dim

### Option 3: Code-Specific Models

For better code understanding, consider code-trained models:

**CodeBERT** (Microsoft):
```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

def embed_fn(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, 
                      return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Use [CLS] token embeddings
    embeddings = outputs.last_hidden_state[:, 0, :].numpy()
    return embeddings.astype('float32')
```

**GraphCodeBERT** (Microsoft):
- Similar to CodeBERT but trained with code structure
- Better for control flow understanding

**UniXcoder** (Microsoft):
- Unified cross-modal pre-training
- Works with code, AST, and natural language

## Metadata and Ranking

- Each embedding row includes metadata: `is_code`, `is_doc`, `is_config`, `is_data`, `extension`, `language`.
- Language is treated as authoritative for ranking only when `is_code` is `true`; for non-code files it is metadata for filtering.
- Semantic search pulls a candidate pool (top 50–200 by vector similarity) and reranks with metadata-aware boosts:
  - `code_only=true`: hard filter to code chunks.
  - `prefer_code=true`: boost code while still allowing docs/config when they are the best hits.

## Example: Complete Workflow

See `example_vector_index.py` for a complete example:

```bash
python example_vector_index.py
```

This demonstrates:
1. Initializing index with embedding function
2. Indexing a repository
3. Building vector index
4. Viewing statistics

## Multi-Model Support

You can index the same repository with different models:

```python
# Index with sentence-transformers
index.build_vector_index(
    repo="my-project",
    embed_fn=st_embed_fn,
    model="all-MiniLM-L6-v2"
)

# Also index with OpenAI
index.build_vector_index(
    repo="my-project",
    embed_fn=openai_embed_fn,
    model="text-embedding-3-small"
)
```

Each model's embeddings are stored separately, allowing comparison of results.

## Performance Considerations

### Build Time

Embedding generation is the bottleneck:
- **Local models**: 50-200 chunks/second (CPU-dependent)
- **GPU acceleration**: 500+ chunks/second with CUDA
- **OpenAI API**: Rate limited, ~100 requests/minute

For a 10,000-line repository (~100 chunks):
- Local: 1-2 seconds
- OpenAI: 1-2 minutes (rate limits)

### Storage Requirements

Per chunk (100 lines):
- **384-dim model**: ~1.5 KB per chunk
- **768-dim model**: ~3 KB per chunk
- **1536-dim model**: ~6 KB per chunk

Example: 1000 chunks with 384-dim = ~1.5 MB

### Search Performance (Future)

Initial implementation uses brute-force cosine similarity:
- **1K chunks**: <10ms
- **10K chunks**: ~50ms
- **100K chunks**: ~500ms

For larger repositories, future updates may add:
- FAISS for approximate nearest neighbor search
- Hierarchical indexing
- GPU acceleration

## Incremental Updates

The system tracks which documents are already embedded:

```python
# Only embeds new/changed documents
index.build_vector_index(repo="my-project", force=False)

# Force rebuild all embeddings
index.build_vector_index(repo="my-project", force=True)
```

If you're migrating from the old SQLite storage, run a full rebuild once to populate LanceDB and then drop the legacy table:

```bash
python scripts/rebuild_indexes.py  # or POST /admin/vector/rebuild per repo
sqlite3 ~/.sigil_index/repos.db "DROP TABLE IF EXISTS embeddings;"
```

## Troubleshooting

### Error: "No embedding function configured"

You must provide `embed_fn` when initializing or calling `build_vector_index`:

```python
index = SigilIndex(index_path, embed_fn=my_embed_fn)
# OR
index.build_vector_index(repo="x", embed_fn=my_embed_fn)
```

### Error: "Repository not indexed yet"

Build vector index requires the repository to be indexed first:

```python
index.index_repository("my-project", Path("/path"))  # First
index.build_vector_index("my-project")  # Then
```

### Dimension Mismatch

All embeddings for a model must have the same dimension. If you change embedding functions, use a different `model` name or force rebuild.

## Future Enhancements

Planned features:
1. **Hybrid search**: Combine trigram, symbol, and semantic results
2. **FAISS integration**: Fast approximate similarity search for large repos
3. **Re-ranking**: Use cross-encoders for better result ordering
4. **Metadata filtering**: Search within specific files/directories
5. **Context expansion**: Show more lines around matched chunks

## Related Documentation

- [ADR-006: Vector Embeddings](adr-006-vector-embeddings.md) - Design decisions
- [ADR-002: Trigram Indexing](adr-002-trigram-indexing.md) - Text search
- [ADR-003: Symbol Search](adr-003-symbol-search-ctags.md) - Definition lookup
- [example_vector_index.py](../example_vector_index.py) - Code examples
