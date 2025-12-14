<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# Llama.cpp Embedding Setup

This guide explains how to set up and use llama.cpp with Meta Llama 3.1 8B Instruct (or other models) for local embeddings in Sigil MCP Server.

**Default (v0.6.0+):** Sigil ships with `llamacpp` as the default provider and expects the Jina code embeddings model at
`/home/dave/models/jina/jina-embeddings-v2-base-code-Q4_K_M.gguf` (768-dim). If your environment differs, set
`embeddings.provider`/`embeddings.model` in `config.json` to point to your GGUF.

## Why Llama.cpp?

**Advantages:**
- **Fully Local**: No API calls, complete privacy
- **Zero Cost**: No API fees
- **Customizable**: Use any GGUF model
- **GPU Acceleration**: Optional GPU support for faster inference
- **Privacy**: Code never leaves your machine

**Trade-offs:**
- Slower than cloud APIs (especially CPU-only)
- Requires downloading large model files (4-8GB)
- Higher memory usage (4-8GB RAM minimum)

## Prerequisites

1. **Python 3.12+**
2. **4-8GB RAM** (8GB+ recommended for 8B models)
3. **GPU** (optional, but recommended for speed)
   - NVIDIA GPU with CUDA support
   - Or Apple Silicon Mac with Metal support

## Installation

### Step 1: Install llama-cpp-python

**CPU Only:**
```bash
pip install llama-cpp-python
```

**With GPU Support (NVIDIA/CUDA):**
```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
```

**With GPU Support (Apple Silicon/Metal):**
```bash
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python
```

**Or install as optional dependency:**
```bash
pip install -e ".[llamacpp]"
```

### Step 2: Download a GGUF Model

Download Meta Llama 3.1 8B Instruct in GGUF format:

**From Hugging Face:**
```bash
# Create models directory
mkdir -p ~/models
cd ~/models

# Download Q4_K_M quantization (recommended - good balance of quality/size)
wget https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf

# Or download Q5_K_M for higher quality
wget https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf
```

**Available Quantizations:**
- `Q4_K_M` - 4.9GB - **Recommended** (good balance)
- `Q5_K_M` - 5.7GB - Higher quality, slower
- `Q6_K` - 6.6GB - Even higher quality
- `Q8_0` - 8.5GB - Highest quality, slow

## Configuration

Edit your `config.json`:

```json
{
  "embeddings": {
    "enabled": true,
    "provider": "llamacpp",
    "model": "~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "dimension": 4096,
    "context_size": 2048,
    "llamacpp_context_size": 8192,
    "n_ctx": 8192,
    "n_gpu_layers": 0,
    "use_mlock": false
  }
}
```

### Configuration Options

**`model`** (required)
- Path to your GGUF model file
- Supports `~` for home directory expansion
- Example: `"~/models/llama-3.1-8b-instruct.gguf"`

**`dimension`** (default: 4096)
- Embedding vector dimension
- 4096 for Llama 3.1 8B
- Must match your model's hidden size

**`context_size` / `llamacpp_context_size` / `n_ctx`** (default: 2048)
- Maximum context window in tokens (all map to the provider's `context_size`)
- Llama 3.1 supports up to 128K; pick what your hardware can handle
- Larger = more memory usage

**`n_gpu_layers`** (default: 0)
- Number of layers to offload to GPU
- 0 = CPU only
- 32 = full offload for 8B models (recommended if you have GPU)
- Higher = faster but more VRAM

**`n_batch` / `n_ubatch`** (defaults: n_batch 2048, n_ubatch = n_batch)
- llama.cpp logical batch vs micro-batch. `n_ubatch` must be **>= tokens in any single embed call** or llama.cpp will hard-abort with `encoder requires n_ubatch >= n_tokens`.
- llama-cpp-python clamps `n_ubatch <= n_batch`, so raise both together (e.g., set both to `4096` to avoid crashes on long inputs).
- Sigil auto-splits long texts to fit the configured `n_ubatch`, but larger values reduce chunking and speed up indexing if you have headroom.

**`use_mlock`** (default: false)
- Lock model in RAM to prevent swapping
- Recommended if you have enough RAM
- Prevents slowdowns from disk I/O

### GPU Configuration Examples

**NVIDIA GPU (16GB+ VRAM):**
```json
{
  "embeddings": {
    "provider": "llamacpp",
    "model": "~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "dimension": 4096,
    "context_size": 2048,
    "n_gpu_layers": 32,
    "use_mlock": true
  }
}
```

**Apple Silicon (M1/M2/M3):**
```json
{
  "embeddings": {
    "provider": "llamacpp",
    "model": "~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "dimension": 4096,
    "context_size": 2048,
    "n_gpu_layers": 1,
    "use_mlock": true
  }
}
```
*Note: On Apple Silicon, layers are automatically offloaded to GPU via Metal*

**CPU Only (16GB+ RAM):**
```json
{
  "embeddings": {
    "provider": "llamacpp",
    "model": "~/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
    "dimension": 4096,
    "context_size": 2048,
    "n_gpu_layers": 0,
    "use_mlock": true
  }
}
```

## Usage

### Start the Server

```bash
python -m sigil_mcp.server
```

The server will:
1. Load the GGUF model (may take 10-30 seconds)
2. Begin indexing repositories
3. Generate embeddings using llama.cpp

### Index a Repository

Via MCP tool (from ChatGPT):
```
"Index the my_project repository"
```

Or programmatically:
```python
from sigil_mcp.indexer import SigilIndex
from pathlib import Path

index = SigilIndex(Path('~/.sigil_index').expanduser())
stats = index.index_repository('my_project', Path('/path/to/project'))
print(f"Indexed {stats['files']} files")
```

### Test Semantic Search

Via MCP tool:
```
"Search for authentication logic in my_project"
```

Or programmatically:
```python
from sigil_mcp.indexer import SigilIndex
from pathlib import Path

index = SigilIndex(Path('~/.sigil_index').expanduser())
results = index.semantic_search(
    query="implement authentication logic",
    repo_name="my_project",
    top_k=5
)

for result in results:
    print(f"{result['file_path']}: {result['score']:.3f}")
```

## Performance Tuning

### Speed Optimization

**1. Use GPU Acceleration**
```json
{
  "n_gpu_layers": 32  // Full offload for 8B model
}
```

**2. Use Smaller Quantization**
```bash
# Q4_K_M is fastest while maintaining quality
wget https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

**3. Reduce Context Size**
```json
{
  "context_size": 1024  // Smaller = faster
}
```

### Memory Optimization

**1. Use Lower Quantization**
```bash
# Q4_K_M uses less memory
wget https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

**2. Enable Memory Locking**
```json
{
  "use_mlock": true  // Prevents swapping
}
```

**3. Batch Processing**
The indexer automatically batches embedding generation to manage memory.

## Benchmarks

Performance on Meta Llama 3.1 8B Instruct (Q4_K_M):

| Configuration | Speed (files/sec) | Memory Usage |
|--------------|-------------------|--------------|
| CPU Only (16 cores) | ~5 files/sec | 6GB RAM |
| NVIDIA RTX 4090 | ~50 files/sec | 8GB VRAM + 4GB RAM |
| Apple M2 Max | ~25 files/sec | 8GB Unified |

*Benchmarks on repository with 1000 Python files, avg 200 lines each*

## Troubleshooting

### Model Won't Load

**Error:** `FileNotFoundError: Model file not found`

**Solution:**
```bash
# Check path expansion
python -c "from pathlib import Path; print(Path('~/models/llama.gguf').expanduser())"

# Verify file exists
ls -lh ~/models/*.gguf
```

### Out of Memory

**Error:** `RuntimeError: Failed to allocate memory`

**Solutions:**
1. Use smaller quantization (Q4_K_M instead of Q8_0)
2. Reduce `n_gpu_layers`
3. Reduce `context_size`
4. Close other applications

### Slow Performance

**Solutions:**
1. Enable GPU acceleration (`n_gpu_layers > 0`)
2. Use smaller quantization
3. Reduce `context_size`
4. Use `use_mlock` to prevent swapping

### GPU Not Being Used

**Check CUDA installation:**
```bash
python -c "import llama_cpp; print(llama_cpp.__version__)"
nvidia-smi  # Should show GPU usage when embedding
```

**Reinstall with CUDA:**
```bash
pip uninstall llama-cpp-python
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --no-cache-dir
```

## Alternative Models

You can use other GGUF models besides Llama 3.1:

### Mistral 7B
```bash
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

```json
{
  "model": "~/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
  "dimension": 4096
}
```

### Llama 2 13B (Higher Quality)
```bash
wget https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_K_M.gguf
```

```json
{
  "model": "~/models/llama-2-13b-chat.Q4_K_M.gguf",
  "dimension": 5120
}
```

### CodeLlama 7B (Code-Specialized)
```bash
wget https://huggingface.co/TheBloke/CodeLlama-7B-GGUF/resolve/main/codellama-7b.Q4_K_M.gguf
```

```json
{
  "model": "~/models/codellama-7b.Q4_K_M.gguf",
  "dimension": 4096
}
```

## See Also

- **Main Documentation:** [README.md](../README.md)
- **Vector Embeddings Guide:** [VECTOR_EMBEDDINGS.md](VECTOR_EMBEDDINGS.md)
- **Operations Runbook:** [RUNBOOK.md](RUNBOOK.md)
- **Troubleshooting:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **llama.cpp GitHub:** https://github.com/ggerganov/llama.cpp
- **llama-cpp-python Docs:** https://llama-cpp-python.readthedocs.io/
