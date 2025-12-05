<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# ADR-011: Admin API for Operational Management

## Status

Accepted

## Context

The Sigil MCP Server requires operational management capabilities for:
- Monitoring server status and configuration
- Triggering index rebuilds without restarting the server
- Viewing index statistics
- Accessing recent log entries for troubleshooting
- Inspecting current configuration

Previously, these operations required:
- Direct server restarts
- Manual database access
- Log file inspection via filesystem
- Configuration file inspection

This created operational friction and made it difficult to manage the server in production environments.

## Decision

Implement an Admin API that provides HTTP endpoints for operational management. The Admin API:

1. **Runs integrated into the main MCP server** on the same port (default: 8000) at `/admin/*` endpoints
2. **Uses IP-based access control** (default: localhost only)
3. **Supports optional API key authentication** for additional security
4. **Provides RESTful endpoints** for common operations
5. **Is enabled by default** and accessible when the main server is running

### Architecture

The Admin API is implemented as a Starlette application (`sigil_mcp/admin_api.py`) that:
- **Runs in the same process as the main MCP server** (integrated via parent ASGI app)
- Shares the same configuration system
- **Shares the same index instance** (no database lock conflicts)
- Uses operational helper functions from `server.py` (`rebuild_index_op`, `build_vector_index_op`, `get_index_stats_op`)
- A React-based Admin UI is available on port 5173 (started via `scripts/restart_servers.sh`)

### Endpoints

- `GET /admin/status` - Server status, repositories, index info, watcher status
- `POST /admin/index/rebuild` - Rebuild trigram/symbol index (all repos or specific repo)
- `GET /admin/index/stats` - Get index statistics (all repos or specific repo)
- `POST /admin/vector/rebuild` - Rebuild vector embeddings index
- `GET /admin/logs/tail` - Get last N lines from server log file
- `GET /admin/config` - View current configuration (read-only)

### Security Model

1. **IP Whitelist**: Only allows connections from configured IPs (default: 127.0.0.1, ::1)
2. **Optional API Key**: If `admin.api_key` is set, requires `X-Admin-Key` header
3. **Separate Service**: Runs on different port than main MCP server
4. **Localhost Default**: By default, only accessible from localhost

### Configuration

```json
{
  "admin": {
    "enabled": true,
    "host": "127.0.0.1",
    "port": 8000,
    "api_key": null,
    "allowed_ips": ["127.0.0.1", "::1"]
  }
}
```

Environment variables:
- `SIGIL_MCP_ADMIN_ENABLED` (default: "true")
- `SIGIL_MCP_ADMIN_HOST` (default: "127.0.0.1")
- `SIGIL_MCP_ADMIN_PORT` (legacy, no longer used - Admin API runs on main server port)
- `SIGIL_MCP_ADMIN_API_KEY` (optional)
- `SIGIL_MCP_ADMIN_ALLOWED_IPS` (default: "127.0.0.1,::1")

## Consequences

### Positive

- **Operational Flexibility**: Can trigger index rebuilds without server restart
- **Better Monitoring**: Easy access to server status and statistics
- **Improved Debugging**: Log tail endpoint for quick troubleshooting
- **Separation of Concerns**: Admin operations separate from MCP protocol
- **Security**: IP-based access control and optional API key authentication
- **Non-Intrusive**: Disabled by default, doesn't affect main server if not used

### Negative

- **Additional Service**: Requires running a second HTTP service
- **Port Management**: Uses the same port as the main MCP server (no additional port needed)
- **Configuration Complexity**: Additional configuration options to manage
- **Security Surface**: Additional attack surface (mitigated by IP whitelist)

### Neutral

- **Shared State**: Admin API accesses same index/watcher instances as main server
- **No MCP Integration**: Admin API is separate from MCP protocol (intentional)

## Implementation Details

### Service Startup

**Recommended: Use the restart script (starts MCP server + Admin API + Admin UI):**

```bash
./scripts/restart_servers.sh
```

The Admin API is automatically available when the main server is running. No separate process needed.

**Note:** The `admin_api_main.py` module is deprecated. The Admin API is now integrated into the main server process and starts automatically when `admin.enabled=true` (default).

### Operational Helpers

The Admin API uses shared operational helper functions from `server.py`:
- `rebuild_index_op()` - Rebuilds trigram/symbol index
- `build_vector_index_op()` - Rebuilds vector embeddings index
- `get_index_stats_op()` - Gets index statistics

These functions are also used by MCP tools, ensuring consistency.

### Error Handling

All endpoints return structured JSON error responses:
```json
{
  "error": "error_code",
  "detail": "Human-readable error message"
}
```

HTTP status codes:
- 200: Success
- 401: Unauthorized (invalid API key)
- 403: Forbidden (IP not allowed)
- 404: Not Found (log file not found)
- 500: Internal Server Error
- 503: Service Unavailable (admin API disabled)

## Future Enhancements

- Web UI for admin operations (Stage 2)
- Metrics endpoint (prometheus-compatible)
- Health check endpoint with detailed status
- Index backup/restore operations
- Repository management (add/remove repos via API)

## Related

- [ADR-004: Configuration System](adr-004-configuration-system.md)
- [ADR-008: Granular Indexing](adr-008-granular-indexing.md)
- [RUNBOOK.md](RUNBOOK.md) - Admin API usage procedures


