<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# ADR-001: OAuth 2.0 Authentication for Remote Access

## Status

Accepted

## Context

The Sigil MCP server needs to be accessible remotely via tools like ngrok so that ChatGPT and other AI assistants can connect to local code repositories. However, exposing a local server to the internet creates significant security risks:

- Anyone with the server URL could access private source code
- Simple API keys can be intercepted or leaked
- Need to balance security with ease of use for local development
- MCP clients (like ChatGPT) need standardized authentication flows

The server must support both secure remote access and frictionless local development.

## Decision

Implement OAuth 2.0 with PKCE (Proof Key for Code Exchange) as the primary authentication mechanism, with these characteristics:

1. **OAuth 2.0 Authorization Code Flow**: Industry-standard authentication protocol
2. **PKCE Support (RFC 7636)**: Prevents authorization code interception attacks using S256 challenge method
3. **Local Connection Bypass**: Connections from localhost (127.0.0.1, ::1) automatically bypass authentication
4. **API Key Fallback**: Simple API key authentication available for basic use cases
5. **Auto-discovery**: OAuth metadata endpoint (RFC 8414) at `/.well-known/oauth-authorization-server`
6. **Token Lifecycle**: Access tokens expire after 1 hour, refresh tokens allow renewal
7. **Flexible Redirect URIs**: Accept any HTTPS redirect URI to support various MCP clients

## Consequences

### Positive

- Industry-standard authentication that ChatGPT and other MCP clients understand natively
- PKCE prevents authorization code interception without requiring client secrets
- Local development remains frictionless (no auth needed for localhost)
- OAuth metadata endpoint enables auto-discovery by clients
- Refresh tokens allow long-lived sessions without re-authentication
- Secure token storage with restricted file permissions (0600)
- Compatible with ngrok and other tunneling solutions

### Negative

- More complex than simple API key authentication
- Requires OAuth client credentials management (Client ID and Secret)
- First-time setup requires saving OAuth credentials
- Token management adds server-side state
- ngrok free tier URLs change on restart, requiring client reconfiguration

### Neutral

- OAuth credentials stored in `~/.sigil_mcp_server/oauth/`
- Client secrets are hashed before storage using SHA-256
- Authorization codes expire after 10 minutes
- State parameters expire after 10 minutes for CSRF protection

## Alternatives Considered

### Alternative 1: API Key Only

Simple bearer token authentication with API keys passed in `X-API-Key` header.

**Rejected because:**
- Not compatible with ChatGPT's OAuth-based MCP connector
- No token expiration or refresh mechanism
- Less secure for long-lived remote connections
- No standardized discovery mechanism

### Alternative 2: mTLS (Mutual TLS)

Client certificate authentication where both client and server verify each other's certificates.

**Rejected because:**
- Complex certificate management and distribution
- ChatGPT and most MCP clients don't support mTLS easily
- Overkill for the threat model (protecting code access, not government secrets)
- Certificate expiration and renewal adds operational burden

### Alternative 3: No Authentication with IP Whitelist Only

Rely solely on IP whitelisting to restrict access.

**Rejected because:**
- ChatGPT's IP addresses are not static or publicly documented
- Defeats the purpose of using ngrok (dynamic IPs)
- No protection if IP whitelist is misconfigured
- Doesn't prevent unauthorized access from whitelisted networks

### Alternative 4: Session-Based Authentication

Traditional cookie-based session authentication.

**Rejected because:**
- Not RESTful, requires server-side session storage
- Cookies are not standard for API authentication
- MCP protocol and ChatGPT expect token-based auth
- Poor fit for machine-to-machine communication

## Related

- [RFC 6749: The OAuth 2.0 Authorization Framework](https://tools.ietf.org/html/rfc6749)
- [RFC 7636: Proof Key for Code Exchange (PKCE)](https://tools.ietf.org/html/rfc7636)
- [RFC 8414: OAuth 2.0 Authorization Server Metadata](https://tools.ietf.org/html/rfc8414)
- [OAuth Setup Documentation](OAUTH_SETUP.md)
- [Security Documentation](SECURITY.md)
