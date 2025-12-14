<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# Quickstart & Common Knobs

This project ships with a full-featured config, but you can get running quickly with a minimal setup and a few easy switches.

## 1) Install the server

```bash
pip install "sigil-mcp-server[watch]"
```

Use the extras only if you need them:

- `embeddings-llamacpp-cpu` (local Jina code embeddings)
- `lancedb` (vector index backed by LanceDB; optional if you only want trigram search)

## 2) Create a minimal `config.json`

```json
{
  "mode": "dev",
  "server": { "host": "127.0.0.1", "port": 8000 },
  "index": { "path": "~/.sigil_index" },
  "repositories": {
    "my-repo": "/path/to/your/repo"
  },
  "embeddings": {
    "enabled": false
  },
  "admin": {
    "enabled": true,
    "host": "127.0.0.1",
    "port": 9999,
    "require_api_key": false,
    "allowed_ips": ["127.0.0.1"]
  },
  "admin_ui": { "auto_start": false }
}
```

Start the server:

```bash
python -m sigil_mcp.server --config config.json
# or, if installed as a console script:
sigil-mcp --config config.json
```

## 3) Flip the most common knobs

- **Index location**: `index.path` (default `~/.sigil_index`)
- **Repos**: add entries under `repositories` (name → path). For per-repo options, use an object: `{"path": "...", "respect_gitignore": true, "embeddings_include_solution": true}`
- **Embeddings**:
  - Disable entirely: `embeddings.enabled=false` (trigram search still works).
  - Enable llama.cpp: set `embeddings.enabled=true`, `provider="llamacpp"`, `model="/path/to/model.gguf"`, `dimension` (usually 768), optional `llamacpp_context_size`/`n_ctx` for larger context windows.
  - Prefer trigram-only indexing? Leave embeddings off.
- **Admin API**:
  - Prod hardening: set `mode="prod"`, `admin.require_api_key=true`, and set `admin.api_key`.
  - IP allowlist: `admin.allowed_ips` (and `authentication.allowed_ips` if auth is enabled).
- **Admin UI**: `admin_ui.auto_start=false` if you do not want the UI process launched automatically.
- **Watching**: `watch.enabled` (if present in your config) controls file-watcher auto-reindexing.

## 4) Troubleshooting essentials

- **Rebuild everything**: `python -m sigil_mcp.scripts.rebuild_indexes --config config.json`
- **Semantic search disabled?** Ensure `embeddings.enabled=true`, `embed_fn` loads cleanly (model path correct), and LanceDB is available (or enable the test stub via `SIGIL_MCP_LANCEDB_STUB=1` in dev).
- **Index location**: look under `~/.sigil_index/` for `repos.db`, `trigrams.rocksdb/`, and `lancedb/` (per-repo tables).

For deeper details, see:
- `docs/RUNBOOK.md` (operational runbook)
- `docs/EMBEDDING_SETUP.md` (embedding providers and config)
- `docs/LLAMACPP_SETUP.md` (GPU setup for llama.cpp: AMD/ROCm/Metal; see notes for Vulkan paths)
- `docs/TROUBLESHOOTING.md` (common failure modes and fixes)

## 5) GPU setup (llama.cpp embeddings)

- **NVIDIA**: install CUDA-capable `llama-cpp-python` wheels (e.g., `pip install "llama-cpp-python==<version>" --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu118`). Set `embeddings.provider="llamacpp"` and point `embeddings.model` to a GGUF. Use `llamacpp_context_size`/`n_ctx` for larger contexts. See `docs/LLAMACPP_SETUP.md` for flags and model recommendations.
- **AMD (ROCm)**: use the ROCm wheel (`pip install -e ".[embeddings-llamacpp-rocm]"` or appropriate `llama-cpp-python` ROCm wheel). Set `embeddings.provider="llamacpp"` and configure `embeddings.model` and `llamacpp_context_size`.
- **AMD (Vulkan)**: follow upstream `llama-cpp-python` Vulkan build instructions (compile with `LLAMA_VULKAN=on`); then set provider/model as above. Vulkan builds are more DIY; see `docs/LLAMACPP_SETUP.md` for pointers.
- **CPU-only**: use `embeddings-llamacpp-cpu` extra; embeddings run locally without GPU.

## 6) Run via Docker

We publish a container image (`sigilderg-custom-mcp:1.0.1`). To build locally:

```bash
docker build -t sigilderg-custom-mcp:1.0.1 .
```

Run with a mounted config and index/data volume:

```bash
docker run --rm -p 8000:8000 -p 8765:8765 \
  -v $PWD/config.json:/app/config.json \
  -v $HOME/.sigil_index:/data/index \
  -e SIGIL_INDEX_PATH=/data/index \
  sigilderg-custom-mcp:1.0.1
```

If you need embeddings with GPU inside Docker, ensure the host runtime (NVIDIA Container Toolkit or appropriate ROCm/Vulkan setup) is present and the image has the necessary GPU-enabled `llama-cpp-python` wheel installed (build or extend the image accordingly).
