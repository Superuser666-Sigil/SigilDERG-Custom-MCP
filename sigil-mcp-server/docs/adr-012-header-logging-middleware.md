<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# ADR-012: ASGI Header Logging Middleware

## Status

Accepted

## Context

When troubleshooting MCP server issues, especially with external clients like ChatGPT, we need visibility into:
- All HTTP requests hitting the server
- Request headers (for debugging client compatibility issues)
- Response status codes and latency
- Client IP addresses
- Cloudflare ray IDs (for correlation with Cloudflare logs)

Previously, we only logged headers in specific OAuth routes, which meant:
- MCP tool calls had no header visibility
- Streamable HTTP transport had no logging
- Health check endpoints had no logging
- No correlation IDs for request/response pairs
- Sensitive headers (API keys, tokens) were logged in plaintext

This made it difficult to:
- Debug ChatGPT integration issues
- Correlate errors with specific requests
- Provide support ticket information
- Verify server responses to clients

## Decision

Implement ASGI-level middleware (`HeaderLoggingASGIMiddleware`) that:
1. **Intercepts all HTTP requests** before they reach FastMCP
2. **Logs request metadata** (method, path, headers, client IP, cf-ray)
3. **Redacts sensitive headers** before logging (authorization, cookies, API keys)
4. **Logs response metadata** (status code, duration)
5. **Generates request IDs** for correlation
6. **Passes through non-HTTP traffic** (websockets, etc.)

### Header Redaction

Sensitive headers are redacted to `<redacted>` before logging:
- `authorization`
- `cookie`
- `x-api-key`
- `x-admin-key`
- `x-openai-session`
- `x-openai-session-token`

This allows full request visibility for debugging while protecting credentials.

### Logging Format

**Incoming Request:**
```
Incoming MCP HTTP request
  request_id=<uuid>
  method=POST
  path=/
  client_ip=203.0.113.42
  cf_ray=8978a4bf1c5a1234-DFW
  headers={...}
```

**Outgoing Response:**
```
Outgoing MCP HTTP response
  request_id=<uuid>
  status_code=200
  duration_ms=143
  path=/
  method=POST
```

The same `request_id` is used for both request and response logs, enabling correlation.

### Client IP Extraction

The middleware extracts client IP from:
1. `X-Forwarded-For` header (first hop if multiple)
2. ASGI scope `client` tuple (direct connection)

This handles both direct connections and proxied connections (Cloudflare, ngrok, etc.).

### Cloudflare Ray ID

The middleware extracts `cf-ray` header for correlation with Cloudflare logs, making it easier to trace requests through Cloudflare's infrastructure.

## Consequences

### Positive

- **Full Request Visibility**: All HTTP requests are logged, not just specific routes
- **Security**: Sensitive headers are redacted before logging
- **Correlation**: Request IDs enable matching requests with responses
- **Debugging**: Headers, IPs, and ray IDs help diagnose client issues
- **Support**: Structured logs make it easy to provide information for support tickets
- **Performance Monitoring**: Duration logging helps identify slow requests

### Negative

- **Log Volume**: Every HTTP request generates 2 log entries (request + response)
- **Performance Overhead**: Minimal (header extraction and logging)
- **Storage**: Increased log file size

### Neutral

- **OAuth Route Logging**: Reduced redundant header logging in OAuth routes (now handled by middleware)
- **Non-HTTP Traffic**: Passes through unchanged (websockets, etc.)

## Implementation Details

### Middleware Installation

The middleware is installed automatically when the server starts by:
1. Accessing FastMCP's underlying ASGI app (`mcp.app` or `mcp.asgi_app`)
2. Wrapping it with `HeaderLoggingASGIMiddleware`
3. Replacing the app on the FastMCP instance

If the underlying app cannot be accessed, a warning is logged but the server continues to function.

### ASGI Scope Handling

The middleware:
- Only processes HTTP requests (`scope["type"] == "http"`)
- Passes through other scope types (websockets, lifespan, etc.)
- Extracts headers from ASGI format `[(b"name", b"value"), ...]`
- Decodes headers using latin-1 encoding (ASGI standard)

### Error Handling

If an error occurs during request processing:
- The error is logged with request context (request_id, path, method, duration)
- The error is re-raised (not swallowed)
- Response logging still occurs (if response was started)

### OAuth Route Changes

OAuth routes (`oauth_authorize_http`, `oauth_token_http`) were updated to:
- Remove direct `dict(request.headers)` logging
- Rely on middleware for header logging (with redaction)
- Keep route-specific logging for OAuth flow context

## Future Enhancements

- Configurable log level for middleware (separate from main server log level)
- Optional request/response body logging (for debugging)
- Metrics export (request rate, latency percentiles)
- Log sampling for high-traffic scenarios

## Related

- [ADR-005: FastMCP Custom Routes](adr-005-fastmcp-custom-routes.md)
- [ADR-009: ChatGPT Compatibility](adr-009-chatgpt-compatibility.md)
- [RUNBOOK.md](RUNBOOK.md) - Logging and troubleshooting procedures
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Using logs for debugging


