<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# Embedding Provider Setup Guide

This guide helps you choose and install the right embedding provider for your hardware.

## Quick Decision Guide

All embeddings are stored in a **LanceDB vector store** under your index directory at `~/.sigil_index/lancedb/` (or the custom
`index.path` you configure). Each repository gets its own `code_vectors` table inside that directory. The current
recommendation is a **768-dimension model** to balance recall and storage size; the examples below use that dimension unless
otherwise noted.

**Default** (v0.6.0+): `llamacpp` with Jina v2 code embeddings at
`/home/dave/models/jina/jina-embeddings-v2-base-code-Q4_K_M.gguf` (768-dim). Override `embeddings.provider`/`embeddings.model` if your
environment uses a different backend or model path.

**Choose sentence-transformers if:**
- [YES] You have an NVIDIA GPU
- [YES] You want the simplest setup
- [YES] You're okay with ~500MB-1GB model downloads
- [YES] You have stable internet for initial model download

**Choose llama.cpp if:**
- [YES] You have an AMD GPU (ROCm)
- [YES] You want full control over model selection
- [YES] You prefer smaller, quantized models
- [YES] You need maximum CPU performance
- [YES] You're on Apple Silicon (Metal)

**Choose OpenAI API if:**
- [YES] You don't want to run models locally
- [YES] You have an OpenAI API key
- [YES] You're okay with API costs (~$0.0001 per 1K tokens)
- [YES] You want the highest quality embeddings

## Installation by Hardware

### NVIDIA GPU Systems

Best option: **sentence-transformers**

```bash
pip install -e .[embeddings-sentencetransformers]
```

Configuration in `config.json`:
```json
{
  "embeddings": {
    "enabled": true,
    "provider": "sentence-transformers",
    "model": "all-MiniLM-L12-v2",
    "dimension": 768,
    "cache_dir": "~/.cache/sigil-embeddings"
  }
}
```

**Configuration Properties:**
- `embeddings.enabled`: Enable/disable embedding features (default: `false`)
- `embeddings.provider`: Provider name: `sentence-transformers`, `openai`, or `llamacpp`
- `embeddings.model`: Model identifier (for sentence-transformers/openai) or path to GGUF file (for llamacpp)
- `embeddings.dimension`: Expected embedding dimension (e.g., 384, 768, 1536)
- `embeddings.cache_dir`: Cache directory for downloaded models (optional, defaults to system cache)
- `embeddings.api_key`: API key for cloud providers (optional, falls back to `OPENAI_API_KEY` env var)
- `embeddings.llamacpp_context_size` or `embeddings.n_ctx`: Optional override for llama.cpp context; mapped to provider `context_size` (default 2048 if unset)

**Why?** PyTorch has excellent CUDA support out of the box. sentence-transformers will automatically use your GPU.

---

### AMD GPU Systems (Radeon RX, PRO)

Best option: **llama.cpp with ROCm**

```bash
pip install -e .[embeddings-llamacpp-rocm]
```

Configuration in `config.json`:
```json
{
  "embeddings": {
    "enabled": true,
    "provider": "llamacpp",
    "model": "/path/to/model.gguf",
    "dimension": 768,
    "n_gpu_layers": 99,
    "n_ctx": 512,
    "llamacpp_context_size": 8192
  }
}
```

**Why?** llama.cpp has better ROCm support than PyTorch. See [LLAMACPP_SETUP.md](LLAMACPP_SETUP.md) for model recommendations.

**Alternative:** sentence-transformers can work with ROCm but requires manual PyTorch ROCm installation:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
pip install -e .[embeddings-sentencetransformers]
```

---

### Apple Silicon (M1/M2/M3/M4)

Best option: **llama.cpp with Metal**

```bash
pip install -e .[embeddings-llamacpp-metal]
```

Configuration in `config.json`:
```json
{
  "embeddings": {
    "enabled": true,
    "provider": "llamacpp",
    "model": "/path/to/model.gguf",
    "dimension": 768,
    "n_gpu_layers": 99,
    "n_ctx": 512
  }
}
```

**Why?** Metal provides the best GPU acceleration on Apple Silicon.

---

### CPU-Only Systems

Best option: **llama.cpp CPU**

```bash
pip install -e .[embeddings-llamacpp-cpu]
Configuration in `config.json`:
```json
{
  "embeddings": {
    "enabled": true,
    "provider": "llamacpp",
    "model": "/path/to/model.gguf",
    "dimension": 768,
    "n_threads": 8,
    "n_ctx": 512
  }
}
```

**Why?** llama.cpp is highly optimized for CPU inference with AVX2/AVX512 support.

**Why?** llama.cpp is highly optimized for CPU inference with AVX2/AVX512 support.

**Alternative:** sentence-transformers works on CPU but will be slower:
```bash
pip install -e .[embeddings-sentencetransformers]
```

---

### Cloud/Remote Systems

Best option: **OpenAI API**

```bash
pip install -e .[embeddings-openai]
Configuration in `config.json`:
```json
{
  "embeddings": {
    "enabled": true,
    "provider": "openai",
  "model": "text-embedding-3-small",
  "dimension": 1536,
    "api_key": "sk-..."
  }
}
```

Or use environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

**Why?** No local compute needed, highest quality embeddings, automatic scaling.

**Why?** No local compute needed, highest quality embeddings, automatic scaling.

## Hardware Detection

Check your system:

```bash
# Check CPU
lscpu | grep "Model name"

# Check GPU (Linux)
lspci | grep -i vga

# Check NVIDIA GPU
nvidia-smi

# Check AMD GPU (if ROCm installed)
rocm-smi

# Check available compute (PyTorch)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, ROCm: {torch.version.hip}')"
```

## Provider Comparison

| Provider | GPU Support | Model Size | Setup Difficulty | Quality | Cost |
|----------|-------------|------------|------------------|---------|------|
| sentence-transformers | NVIDIA (CUDA), AMD (ROCm*) | 80MB-1GB | Easy | Good | Free |
| llama.cpp | NVIDIA (CUDA), AMD (ROCm), Apple (Metal) | 25MB-500MB (quantized) | Medium | Good | Free |
| OpenAI API | N/A (cloud) | N/A | Easy | Excellent | $0.0001/1K tokens |

*ROCm requires manual PyTorch installation

### Minimal Config (Auto-detect)
```json
{
  "embeddings": {
    "enabled": true,
    "provider": "sentence-transformers",
    "model": "all-MiniLM-L6-v2",
    "dimension": 384
  }
}
```

### Performance Tuning
```json
{
  "embeddings": {
    "enabled": true,
    "provider": "llamacpp",
    "model": "./models/nomic-embed-text-v1.5.Q4_K_M.gguf",
    "dimension": 768,
    "n_gpu_layers": 99,
    "n_ctx": 512,
    "n_batch": 512,
    "n_threads": 8,
    "use_mlock": true
  }
}
```

### Multiple Repositories
```json
{
  "repositories": {
    "repo1": "/path/to/repo1",
    "repo2": "/path/to/repo2"
  },
  "embeddings": {
    "enabled": true,
    "provider": "llamacpp",
    "model": "./models/nomic-embed-text-v1.5.Q4_K_M.gguf",
    "dimension": 768
  }
}
```
}
```

## Troubleshooting

### Migrating from SQLite embeddings to LanceDB
- New deployments store vectors in LanceDB at `~/.sigil_index/lancedb/`. If you previously relied on the SQLite `embeddings`
  table inside `repos.db`, rebuild so every repository has a LanceDB-backed `code_vectors` table:
  1. Stop the server and back up your index directory.
  2. Run `python scripts/rebuild_indexes.py` (or `POST /admin/vector/rebuild` for individual repos) to regenerate embeddings into
     LanceDB.
  3. (Legacy installs only) remove the old SQLite table once you're confident in the new data: `sqlite3 ~/.sigil_index/repos.db "DROP TABLE IF EXISTS embeddings;"`.
- After rebuilding you should see a `lancedb` directory under your index path with per-repo subdirectories and `code_vectors`
  tables.

### "No module named 'sentence_transformers'"
You need to install the provider:
```bash
pip install -e .[embeddings-sentencetransformers]
```

### "llama_cpp not found"
Install the right variant for your hardware:
```bash
pip install -e .[embeddings-llamacpp-cuda]    # NVIDIA
pip install -e .[embeddings-llamacpp-rocm]    # AMD
pip install -e .[embeddings-llamacpp-metal]   # Apple
pip install -e .[embeddings-llamacpp-cpu]     # CPU only
```

### llama.cpp encoder requires n_ubatch >= n_tokens / hard abort
- llama.cpp will crash if a single embed call has more tokens than `n_ubatch`. llama-cpp-python also clamps `n_ubatch <= n_batch`.
- Fix: raise **both** `n_batch` and `n_ubatch` (e.g., `4096` each) so `n_ubatch` covers your largest input, or keep them small and let Sigil chunk input (now enforced by default).
- If you override via env/config, ensure `n_batch >= n_ubatch`; set both when in doubt to avoid the GGML_ASSERT.

### Slow embedding generation
1. Check if GPU is being used (add logging to see device)
2. Reduce `n_ctx` for llama.cpp (try 256 or 512)
3. Use smaller/quantized models
4. Increase `n_batch` for llama.cpp (try 512 or 1024)

### Out of memory errors
1. Use smaller model (Q4 or Q5 quantization for llama.cpp)
2. Reduce `n_ctx` and `n_batch`
3. Process fewer files at once
4. Use CPU mode if GPU memory is limited

## Next Steps

- For llama.cpp setup and model recommendations: [LLAMACPP_SETUP.md](LLAMACPP_SETUP.md)
- For vector index usage: [VECTOR_EMBEDDINGS.md](VECTOR_EMBEDDINGS.md)
- For general configuration: [../README.md](../README.md)
