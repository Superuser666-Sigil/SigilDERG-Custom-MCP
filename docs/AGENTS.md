# <!--
# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com
# -->

# Sigil MCP Agent Playbook

## Overview
Sigil MCP gives AI agents IDE-like code navigation (trigram + symbols + embeddings) over configured repositories. Use this playbook to keep agent output predictable and verifiable.

## Project Quick Reference
- Start MCP + Admin UI: `./scripts/restart_servers.sh` (or `python -m sigil_mcp.server`, Admin UI autostarts on port 5173 when enabled).
- Config: `config.json` (`external_mcp_servers` for remote tools; `admin_ui`/`mcp_server` for transports/auth). Example in `config.example.json`.
- Tests: `pytest` (from repo root; venv at `.venv`).
- Health: `/readyz` on MCP server; Admin API at `/admin/*` (see docs).

## External MCP Tools
- Populate `external_mcp_servers` in `config.json` for remote tools (e.g., Playwright, Next.js MCP, MindsDB). Tokenized servers need `headers: { "authorization": "Bearer ..." }`.
- Tool names are prefixed with the server name (e.g., `playwright.click`, `mindsdb.query`).
- Verify registration: `GET /admin/mcp/status` (requires admin access).

## Working Rules
1. Keep changes small and tested (`pytest`).
2. When adding tools, document them in `config.json` and keep secrets in env/headers, not in code.
3. Prefer existing Sigil commands (search_code, list_symbols, goto_definition) before writing new ones.
4. Update docs when changing config surfaces (e.g., external MCP examples, admin UI ports).

## Progressive Disclosure
- `README.md` — feature overview, quick start.
- `config.example.json` — configuration shape and defaults.
- `docs/external_mcp.md` — configuring external MCP servers and presets.
- `docs/mcp.json` — client preset including Sigil + sample external servers.
