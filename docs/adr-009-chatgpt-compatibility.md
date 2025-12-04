<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# ADR-009: ChatGPT MCP Connector Compatibility

## Status

Accepted

## Context

OpenAI's ChatGPT includes an MCP (Model Context Protocol) connector that allows ChatGPT to connect to custom MCP servers for extended functionality. However, ChatGPT's MCP implementation has compatibility issues with the standard MCP streamable-http transport specification:

1. **Content-Type Issue**: ChatGPT sends `Content-Type: application/octet-stream` instead of the required `application/json`
2. **Host Header Issue**: ChatGPT sends ngrok domain names in the Host header, triggering DNS rebinding protection
3. **Path Expectation**: ChatGPT expects the MCP endpoint at root `/` rather than `/mcp`

These issues prevented ChatGPT from connecting to our Sigil MCP server when using FastMCP's security features.

## Server Behavior with Security Enabled

When using FastMCP's `TransportSecuritySettings` with default or standard configuration:

```python
# This configuration blocks ChatGPT
transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=["localhost", "127.0.0.1"]
)
```

Server logs showed:
```
Invalid Content-Type header: application/octet-stream
INFO: "POST / HTTP/1.1" 400 Bad Request

Invalid Host header: abc123.ngrok-free.app
INFO: "POST / HTTP/1.1" 421 Misdirected Request
```

## Decision

Disable DNS rebinding protection and mount MCP endpoint at root path to enable ChatGPT compatibility:

```python
# Disable ALL transport security for ChatGPT compatibility
transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=False
)

mcp = FastMCP(
    name=config.server_name,
    json_response=True,
    streamable_http_path="/",  # Mount at root, not /mcp
    transport_security=transport_security
)
```

This configuration:
1. **Disables DNS rebinding protection** - Accepts any Host header
2. **Disables Content-Type validation** - Accepts application/octet-stream
3. **Mounts at root** - Makes MCP endpoint available at `/` 
4. **Enables JSON responses** - Returns JSON instead of SSE streams
5. **Maintains OAuth** - Authentication still required and enforced

## Consequences

### Positive

- [YES] **ChatGPT works**: Full compatibility with OpenAI's MCP connector
- [YES] **OAuth still active**: Authentication layer protects against unauthorized access
- [YES] **Token validation**: Bearer tokens still required and validated
- [YES] **Simpler deployment**: No need for complex proxy configurations
- [YES] **Standard OAuth flow**: Industry-standard authentication works correctly

### Negative

- [NO] **No DNS rebinding protection**: Server accepts any Host header
- [NO] **No Content-Type validation**: Accepts non-standard content types
- [NO] **Expanded attack surface**: Relies solely on OAuth for security
- [NO] **Non-compliant client**: Accommodating ChatGPT's protocol violations

### Neutral

- Local bypass still works (localhost connections skip auth)
- Other MCP clients can still connect normally
- ngrok still provides TLS encryption automatically
- Server configuration is simpler (fewer security settings)

## Security Analysis

### What We Lost

**DNS Rebinding Protection:**
- **Purpose**: Prevents malicious websites from making requests to localhost servers
- **Impact of disabling**: External sites could theoretically make requests to your server if they know the ngrok URL
- **Mitigation**: OAuth tokens are still required; attacker would need valid tokens

**Content-Type Validation:**
- **Purpose**: Ensures clients send properly formatted requests
- **Impact of disabling**: Non-standard request formats are accepted
- **Mitigation**: Request parsing still validates JSON structure

### What We Kept

**OAuth 2.0 Authentication:**
- [YES] Authorization code flow with PKCE
- [YES] Client ID and Secret validation
- [YES] Access token generation and validation
- [YES] Token expiration (1 hour)
- [YES] Refresh token support
- [YES] Single-use authorization codes

**Network Security:**
- [YES] ngrok provides TLS encryption
- [YES] HTTPS prevents token interception
- [YES] OAuth credentials stored securely (0600 permissions)

**Application Security:**
- [YES] Path traversal protection
- [YES] Repository boundary enforcement
- [YES] Input validation on all tool parameters

### Risk Assessment

**Low Risk:**
- OAuth provides strong authentication barrier
- Attacker needs valid OAuth tokens to access anything
- ngrok URLs are not easily guessable (long random strings)
- Only code repositories are exposed (no system access)

**Medium Risk:**
- DNS rebinding attacks theoretically possible if attacker has valid token
- No defense against stolen/leaked OAuth tokens beyond expiration

**Recommended Mitigations:**
1. **Rotate OAuth credentials** periodically
2. **Monitor access logs** for suspicious patterns
3. **Use ngrok auth** for additional layer (ngrok's own authentication)
4. **Limit repository exposure** - only index what's needed
5. **Short token lifetimes** - keep 1-hour expiration

## Alternatives Considered

### Alternative 1: Use Wildcard allowed_hosts

Configure specific allowed hosts instead of disabling protection:

```python
transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=True,
    allowed_hosts=["*"]  # or ["*.ngrok-free.app", "*.ngrok.io"]
)
```

**Rejected because:**
- Still fails Content-Type validation (application/octet-stream)
- Wildcard effectively disables Host validation anyway
- More configuration complexity for same security result
- ngrok free tier changes domains on restart

### Alternative 2: Custom Middleware

Write custom middleware to accept ChatGPT's non-standard requests:

```python
async def chatgpt_compatibility_middleware(request, call_next):
    # Rewrite Content-Type header
    # Validate and rewrite Host header
    response = await call_next(request)
    return response
```

**Rejected because:**
- Complex to implement correctly
- Fragile (breaks if ChatGPT changes behavior)
- Still need to disable FastMCP's built-in validation
- Maintenance burden for edge case

### Alternative 3: Proxy Server

Put nginx/Apache proxy in front to normalize requests:

```nginx
location / {
    proxy_set_header Content-Type "application/json";
    proxy_set_header Host "localhost";
    proxy_pass http://127.0.0.1:8000;
}
```

**Rejected because:**
- Adds external dependency
- Complex configuration for local development
- Doesn't work with ngrok's dynamic URLs
- Overkill for the problem

### Alternative 4: Wait for ChatGPT Fix

Report issue to OpenAI and wait for them to fix their MCP implementation.

**Rejected because:**
- Unknown timeline for fix
- Users want ChatGPT integration now
- OpenAI may consider their implementation "correct"
- Pragmatic solution needed immediately

### Alternative 5: Use Different MCP Client

Recommend users use a different MCP client that's spec-compliant.

**Rejected because:**
- ChatGPT is the primary use case for most users
- Defeating the purpose of MCP integration
- Alternative clients may not exist or be practical

## Comparison with Original Working Version

The original working version (`/home/dave/dev/server.py`) had:

```python
mcp = FastMCP(name=config.server_name, json_response=True)
# No transport_security parameter
```

When `transport_security` is not specified, FastMCP defaults to:
```python
TransportSecuritySettings(enable_dns_rebinding_protection=False)
```

So the original version **accidentally had the correct configuration** by not specifying security settings at all. When we added explicit security configuration, we broke ChatGPT compatibility.

Our final solution matches the original behavior but is **explicit and documented**.

## Implementation

Changed in `sigil_mcp/server.py`:

```python
# Before (broken with ChatGPT)
if config.allowed_hosts and config.allowed_hosts != ["*"]:
    transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=config.allowed_hosts
    )
else:
    transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=False,
        allowed_hosts=["*"]
    )

# After (works with ChatGPT)
transport_security = TransportSecuritySettings(
    enable_dns_rebinding_protection=False
)

mcp = FastMCP(
    name=config.server_name,
    json_response=True,
    streamable_http_path="/",
    transport_security=transport_security
)
```

## Future Considerations

If OpenAI fixes their MCP implementation to be spec-compliant:

1. **Can re-enable DNS rebinding protection** with specific allowed_hosts
2. **Can remove streamable_http_path="/"** override if they support /mcp
3. **Content-Type validation** can be re-enabled

Until then, this configuration provides the best balance of:
- **Functionality**: ChatGPT works correctly
- **Security**: OAuth provides strong authentication
- **Simplicity**: Minimal configuration needed

## Related

- [MCP Streamable HTTP Specification](https://modelcontextprotocol.io/specification/2025-06-18/transport/streamable-http)
- [FastMCP TransportSecuritySettings](https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/transport_security.py)
- [ADR-001: OAuth 2.0 Authentication](adr-001-oauth2-authentication.md)
- [ADR-005: FastMCP Custom Routes](adr-005-fastmcp-custom-routes.md)
- [ChatGPT Setup Guide](CHATGPT_SETUP.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)

## References

- GitHub Issue: ChatGPT MCP connector compatibility (if filed)
- FastMCP PR#861: DNS rebinding protection implementation
- OpenAI MCP Documentation: https://platform.openai.com/docs/mcp
