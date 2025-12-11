<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# ADR-016: External MCP Aggregation

## Status

Accepted

## Context

Sigil should consume tools from other MCP servers (local stdio or remote SSE/streamable HTTP) and expose them through the same FastMCP instance with consistent naming. We also need operator controls to refresh/inspect external tools and optionally auto-install npm-based servers for convenience in development.

## Decision

1. Add `external_mcp_servers` to `config.json` (with env override `SIGIL_MCP_SERVERS`) describing servers (name, type: stdio/sse/streamable-http, url or command/args, headers/env, timeouts, disabled). Tool names are prefixed with the normalized server name.
2. Implement an aggregation layer that:
   - Connects and calls `list_tools` on startup.
   - Registers server-prefixed tools on the FastMCP instance.
   - Exposes diagnostics tools (`list_mcp_tools`, `external_mcp_prompt`) and admin endpoints (`/admin/mcp/status`, `/admin/mcp/refresh`).
3. Optional dev convenience: `external_mcp_auto_install` (and per-server `auto_install`) will run `npx`/`npm`/`bunx` or explicitly flagged commands on startup to fetch MCP servers. Disabled by default.

## Consequences

### Positive
- Unified MCP tool surface (Sigil + external servers) under one FastMCP endpoint.
- Server discovery is automatic at startup; refreshable via admin endpoint.
- Token/header support for secured servers; stdio support for local tools.

### Negative
- Startup may slow/fail if external servers are unreachable or misconfigured.
- Auto-install can be noisy or fail if npm/bun is missing (kept opt-in).

### Neutral / Mitigations
- Failures to discover a server are logged and surfaced via `/admin/mcp/status` but do not crash Sigil.
- Tool names are prefixed to avoid collisions.
- Prompt helper (`external_mcp_prompt`) provides an optional snippet for agent system prompts; no behavior change if unused.

## Alternatives Considered

- Do nothing: requires clients to connect separately to each MCP server (rejected: worse UX).
- Proxy-only approach without registration: loses tool metadata and prompt integration.
- Mandatory auto-install: rejected; too invasive; kept opt-in.
