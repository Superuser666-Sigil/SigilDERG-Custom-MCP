FROM python:3.12-slim

# System dependencies for ctags and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    universal-ctags \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the package (plus watcher + one embedding provider)
# Copy minimal files first to leverage build cache
COPY pyproject.toml README.md MANIFEST.in ./
COPY sigil_mcp ./sigil_mcp

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir ".[watch,embeddings-llamacpp-cpu,lancedb]"

# Optional: provide a default config inside the image
# Users can override with a bind mount or env vars
COPY config.example.json /app/config.json

# Index data volume
# Index data volume and model cache
VOLUME ["/data", "/models"]

# Default index path inside container
ENV SIGIL_INDEX_PATH=/data/index
# Default model location (override to point at host-mounted models)
ENV SIGIL_MCP_MODELS=/models

# Expose MCP HTTP port
EXPOSE 8000

# Listen on all interfaces by default in the container
ENV SIGIL_MCP_HOST=0.0.0.0
ENV SIGIL_MCP_PORT=8000

# For local dev, you might also want:
# ENV SIGIL_MCP_AUTH_ENABLED=false
# ENV SIGIL_MCP_OAUTH_ENABLED=false
# To switch embeddings/vector store, update the extras installed above
# (e.g., replace embeddings-llamacpp-cpu with embeddings-openai) and adjust
# embeddings.provider/embeddings.model in your config.json.

# Start the MCP server
CMD ["sigil-mcp"]

