# <!--
# Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
# Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
# Commercial licenses are available. Contact: davetmire85@gmail.com
# -->

# Sigil MCP — Claude Agent Guide

## Overview
Sigil MCP provides code search/navigation tools over configured repos. Use this guide when prompting Claude so outputs stay consistent and verifiable.

## Run/Verify
- Start MCP + Admin UI: `./scripts/restart_servers.sh` (Admin UI autostarts on 5173 when enabled) or `python -m sigil_mcp.server`.
- Config file: `config.json` (`external_mcp_servers` for remote tools; `admin_ui`/`mcp_server` for transports/auth). Reference `config.example.json`.
- Tests: `pytest` from repo root (venv at `.venv`).
- Health: `GET /readyz`; Admin API under `/admin/*`.

## External MCP Servers
- Add remote tools via `external_mcp_servers` in `config.json` (Playwright, Next.js MCP, MindsDB, etc.). Use headers for tokens: `"headers": { "authorization": "Bearer <token>" }`.
- Tool names are prefixed with the server name (e.g., `playwright.click`, `mindsdb.query`).
- Check registration: `GET /admin/mcp/status` (admin access required).

## Rules of Engagement
1. Favor Sigil tools first (search_code, list_symbols, goto_definition).
2. Keep changes minimal, run `pytest`.
3. Don’t commit secrets; put tokens in headers/env, not in code.
4. Update docs when changing config surfaces or defaults.

## Progressive Disclosure
- `README.md` — feature overview, quick start.
- `config.example.json` — config structure and defaults.
- `docs/external_mcp.md` — how to configure external MCP servers and auth headers.
- `docs/mcp.json` — client preset including Sigil + sample external servers.
