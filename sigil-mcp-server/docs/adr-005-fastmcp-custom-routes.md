<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# ADR-005: FastMCP Custom Routes for OAuth HTTP Endpoints

## Status

Accepted

## Context

The Model Context Protocol (MCP) is a JSON-RPC based protocol for tool calling. MCP clients (like ChatGPT) communicate with MCP servers using:

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "search_code",
    "arguments": {"query": "async"}
  }
}
```

However, OAuth 2.0 authentication requires standard HTTP endpoints:

- `GET /.well-known/oauth-authorization-server` - OAuth metadata (RFC 8414)
- `GET/POST /oauth/authorize` - Authorization endpoint
- `POST /oauth/token` - Token exchange endpoint
- `POST /oauth/revoke` - Token revocation endpoint

These endpoints need to:
- Use standard HTTP methods (GET, POST)
- Accept form-encoded data (application/x-www-form-urlencoded)
- Return HTTP redirects (302 Found)
- Work outside the MCP JSON-RPC protocol

The FastMCP library (our MCP server framework) is built on Starlette, which supports HTTP routing. However, FastMCP's default behavior is to handle all requests as JSON-RPC.

## Decision

Use FastMCP's `@mcp.custom_route()` decorator to add standard HTTP endpoints alongside MCP tools:

```python
@mcp.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])
async def oauth_metadata(request: Request) -> JSONResponse:
    # Return OAuth server metadata
    pass

@mcp.custom_route("/oauth/authorize", methods=["GET", "POST"])
async def oauth_authorize(request: Request) -> RedirectResponse:
    # Handle OAuth authorization
    pass

@mcp.custom_route("/oauth/token", methods=["POST"])
async def oauth_token(request: Request) -> JSONResponse:
    # Handle token exchange
    pass
```

Key characteristics:
1. **HTTP routes coexist with MCP tools**: Same server handles both protocols
2. **Starlette request/response**: Full access to HTTP primitives (headers, cookies, redirects)
3. **Standard OAuth flow**: Clients use normal OAuth 2.0, no MCP-specific extensions
4. **Auto-discovery**: Metadata endpoint allows clients to discover OAuth configuration

## Consequences

### Positive

- **Standards-compliant OAuth**: No custom protocol extensions needed
- **ChatGPT compatibility**: ChatGPT's OAuth implementation works out of the box
- **Single server process**: No need for separate OAuth server
- **Shared authentication state**: OAuth tokens and MCP tools use same auth manager
- **Flexible routing**: Can add other HTTP endpoints (health checks, metrics) easily
- **Clean separation**: OAuth logic separate from MCP tool logic

### Negative

- **Mixed protocols**: Same server handles both JSON-RPC (MCP) and REST (OAuth)
- **Documentation confusion**: Users need to understand both MCP tools and HTTP endpoints
- **Testing complexity**: Need to test both MCP and HTTP endpoints
- **Framework dependency**: Relies on FastMCP's custom_route feature (less standard)

### Neutral

- HTTP endpoints use Starlette Request/Response objects
- MCP tools still use standard FastMCP decorators (@mcp.tool)
- OAuth endpoints documented separately from MCP tools
- Both protocols share same host and port

## Alternatives Considered

### Alternative 1: Separate OAuth Server

Run a separate HTTP server on different port for OAuth.

**Rejected because:**
- Requires two server processes
- Port management complexity (need two ports)
- Can't share authentication state easily
- More complex deployment and configuration
- Users need to configure two URLs (MCP and OAuth)

### Alternative 2: MCP Tool-Based OAuth

Implement OAuth flow using MCP tools instead of HTTP endpoints.

**Rejected because:**
- ChatGPT and other clients expect standard OAuth HTTP endpoints
- Would require custom client implementation
- Not compatible with existing OAuth libraries and clients
- Defeats purpose of using standard OAuth protocol
- Auto-discovery (RFC 8414) requires HTTP endpoint

### Alternative 3: External OAuth Provider

Use external OAuth provider (Auth0, Okta, etc.) instead of built-in.

**Rejected because:**
- Requires external service signup and configuration
- Network dependency (can't work offline)
- Privacy concerns (third party sees authentication)
- Overkill for local development server
- Defeats purpose of self-contained MCP server

### Alternative 4: HTTP Server Framework (Flask/FastAPI)

Don't use FastMCP, build custom server with Flask/FastAPI handling both MCP and OAuth.

**Rejected because:**
- Reinventing MCP server implementation
- FastMCP handles JSON-RPC protocol correctly
- Would need to manually implement MCP protocol
- More code to maintain
- FastMCP's custom_route solves the problem

### Alternative 5: Proxy Server

Put nginx/Apache in front, route OAuth requests separately.

**Rejected because:**
- Adds external dependency (nginx/Apache)
- Complex configuration for simple use case
- Still need OAuth logic in server anyway
- Poor fit for local development tool
- Harder to deploy and document

## Related

- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Starlette Documentation](https://www.starlette.io/)
- [OAuth 2.0 ADR](adr-001-oauth2-authentication.md)
- [RFC 8414: OAuth Authorization Server Metadata](https://tools.ietf.org/html/rfc8414)
- `server.py` - Implementation with custom routes
