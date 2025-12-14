FROM python:3.12-slim
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.name="sigilderg-custom-mcp"

# System dependencies for ctags, compilation, and the admin UI runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    universal-ctags \
    build-essential \
    nodejs \
    npm \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the package (plus watcher + embeddings + LanceDB)
# Copy minimal files first to leverage build cache
COPY pyproject.toml README.md MANIFEST.in ./
COPY sigil_mcp ./sigil_mcp
COPY scripts ./scripts

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir ".[watch,embeddings-llamacpp-cpu,lancedb]"

# Build the admin UI once so it can auto-start inside the container
COPY sigil-admin-ui ./sigil-admin-ui
RUN npm ci --prefix sigil-admin-ui \
 && npm run build --prefix sigil-admin-ui

# Optional: provide a default config inside the image
# Users can override with a bind mount or env vars
COPY config.example.json /app/config.json

# Index data volume and model cache
VOLUME ["/data", "/models"]

# Default paths inside container
ENV SIGIL_INDEX_PATH=/data/index
ENV SIGIL_MCP_MODELS=/models
ENV SIGIL_MCP_VERSION=1.0.0

# Listen on all interfaces by default in the container
ENV SIGIL_MCP_HOST=0.0.0.0
ENV SIGIL_MCP_PORT=8000
ENV SIGIL_MCP_ADMIN_HOST=0.0.0.0
ENV SIGIL_MCP_ADMIN_PORT=8765

# Admin UI autostart configuration (served by npm dev server)
ENV SIGIL_ADMIN_UI_PATH=/app/sigil-admin-ui
ENV SIGIL_ADMIN_UI_COMMAND=npm
ENV SIGIL_ADMIN_UI_ARGS="run dev -- --host --port 5173"
ENV SIGIL_ADMIN_UI_AUTOSTART=true

# Expose MCP, Admin API, and Admin UI ports
EXPOSE 8000 8765 5173

# Start the MCP server
CMD ["sigil-mcp"]
