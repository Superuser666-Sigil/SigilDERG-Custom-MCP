<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# ADR-015: Default Embeddings via llama.cpp (Jina v2 GGUF)

**Status:** Accepted  
**Date:** 2025-12-09  
**Supersedes:** ADR-006 default provider guidance  
**Related:** ADR-013 (LanceDB vector store)

## Context

- We now rely on a LanceDB-backed vector store (ADR-013) and want a default embedding stack that works fully offline, avoids cloud API costs, and runs on local GPU/CPU.
- The Jina `jina-embeddings-v2-base-code-Q4_K_M.gguf` model (768-dim) provides strong code embeddings with a modest 100MB footprint and runs well under llama.cpp across CPU, NVIDIA, AMD, and Apple Silicon.
- Previous defaults pointed to provider/model names that assumed sentence-transformers downloads. That can fail in air-gapped or CI environments and creates ambiguity about which model to run.
- Operational scripts (`rebuild_indexes`) and server startup should converge on a single, documented default so rebuilds are reproducible.

## Decision

- Set the default embedding provider to **`llamacpp`**.
- Set the default embedding model to **`/home/dave/models/jina/jina-embeddings-v2-base-code-Q4_K_M.gguf`** (768-dim).
- Keep LanceDB as the only vector store; embeddings are written to `index_dir/lancedb/code_vectors`.
- Allow overrides via `embeddings.provider`, `embeddings.model`, and `embeddings.dimension` in `config.json` or environment, but use the above defaults when unspecified.
- Document operational requirements (llama.cpp binary available, model present, dimension = 768) in README, RUNBOOK, and EMBEDDING_SETUP.

## Consequences

### Positive
- Offline-first default: no cloud dependency to get semantic search running.
- Consistent dimension (768) across rebuilds and deployments; aligns with LanceDB schema initialization.
- Works across CPU/GPU (CUDA/ROCm/Metal) because llama.cpp builds are available for all targets.
- Simplifies runbooks and scripts: one well-known model path and provider to validate.

### Negative
- Assumes the GGUF file exists at the default path; missing files will fail provider creation until configured.
- Requires llama.cpp runtime dependencies (e.g., Vulkan/ROCm/CUDA/Metal) on the target machine.
- Operators using sentence-transformers/OpenAI must now override the defaults explicitly.

### Neutral/Mitigations
- Configuration overrides remain backward compatible; setting `embeddings.provider` and `embeddings.model` reverts to prior behavior.
- The model path is user-specific; documentation calls this out and provides override examples.
- If the model is absent, the server degrades gracefully by disabling embeddings and logging a clear error.

## Notes

- This ADR does not change the vector store (still LanceDB per ADR-013).
- Future model upgrades should add a new ADR to capture quality/runtime trade-offs and any dimension changes.

