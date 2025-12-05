<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# OAuth2 Setup for Sigil MCP Server

This guide explains how to set up OAuth2 authentication for your Sigil MCP server, allowing secure remote access from ChatGPT and other MCP clients.

## Authentication Modes

Your Sigil MCP server supports three authentication modes:

1. **Local Bypass** (Default) - Local connections don't need authentication
2. **OAuth2** - Secure token-based authentication for remote connections
3. **API Key** - Simple key-based authentication (fallback)

##  Local Development (No Auth Needed)

For local development, you don't need to configure anything! The server automatically allows connections from `localhost` and `127.0.0.1` without authentication.

```bash
# Just start the server
python server.py

# Connects from localhost work automatically
# No credentials needed!
```

##  Remote Access via ngrok (OAuth Required)

When exposing your server via ngrok or any public URL, use OAuth2 for security.

### Step 1: Start the Server

```bash
python server.py
```

On first run, you'll see:

```
🔐 OAuth2 Authentication
============================================================
🆕 NEW OAuth client created!

Client ID:     sigil_xxxxxxxxxxxxxxxx
Client Secret: yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy

[WARNING]  SAVE THESE CREDENTIALS SECURELY!
```

**Save these credentials!** You'll need them to configure ChatGPT.

### Step 2: Expose via ngrok

```bash
ngrok http 8000
```

You'll get a public URL like: `https://abc123.ngrok-free.app`

### Step 3: Configure ChatGPT MCP Connection

1. In ChatGPT, go to **Settings** → **Third-party integrations** → **Model Context Protocol**
2. Click **Add new connection**
3. Fill in:
   - **Name**: `Sigil Code Search`
   - **Server URL**: `https://abc123.ngrok-free.app`
   - **Authentication**: `OAuth 2.0`
   
4. OAuth Configuration:
   - **Client ID**: (paste the client_id from Step 1)
   - **Client Secret**: (paste the client_secret from Step 1)
   - **Authorization URL**: `https://abc123.ngrok-free.app/oauth/authorize`
   - **Token URL**: `https://abc123.ngrok-free.app/oauth/token`
   - **Scopes**: `read write` (optional)

5. Click **Save** and authorize when prompted

### Step 4: Test the Connection

In ChatGPT, try:
```
Search my code for "async def"
```

ChatGPT will use your MCP server to search your repositories!

##  Configuration Options

### Environment Variables

```bash
# Authentication
export SIGIL_MCP_AUTH_ENABLED=true          # Enable auth (default: true)
export SIGIL_MCP_OAUTH_ENABLED=true         # Enable OAuth (default: true)
export SIGIL_MCP_ALLOW_LOCAL_BYPASS=true    # Allow localhost without auth (default: true)

# Fallback API Key (optional)
export SIGIL_MCP_API_KEY=your_api_key_here  # For local testing

# IP Whitelist (optional)
export SIGIL_MCP_ALLOWED_IPS="1.2.3.4,5.6.7.8"

# Repository Configuration
export SIGIL_REPO_MAP="myproject:/path/to/project;other:/path/to/other"
```

### Disable Authentication (Not Recommended)

```bash
export SIGIL_MCP_AUTH_ENABLED=false
python server.py
```

[WARNING] **Warning**: Only disable auth for completely isolated local development.

## OAuth Flow Details

### Authorization Code Flow (with PKCE)

1. **Authorization Request**
   ```
   GET /oauth/authorize?
       client_id=sigil_xxx&
       response_type=code&
       redirect_uri=http://localhost:8080/callback&
       state=random_state&
       code_challenge=challenge_hash&
       code_challenge_method=S256
   ```

2. **Authorization Response**
   ```
   HTTP/1.1 302 Found
   Location: http://localhost:8080/callback?code=auth_code&state=random_state
   ```

3. **Token Exchange**
   ```
   POST /oauth/token
   Content-Type: application/x-www-form-urlencoded
   
   grant_type=authorization_code&
   code=auth_code&
   redirect_uri=http://localhost:8080/callback&
   client_id=sigil_xxx&
   code_verifier=verifier_string
   ```

4. **Token Response**
   ```json
   {
     "access_token": "abc123...",
     "token_type": "Bearer",
     "expires_in": 3600,
     "refresh_token": "def456...",
     "scope": "read write"
   }
   ```

5. **Using the Token**
   ```
   GET /api/tools
   Authorization: Bearer abc123...
   ```

### Token Refresh

```
POST /oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=refresh_token&
refresh_token=def456...&
client_id=sigil_xxx
```

### Token Revocation

```
POST /oauth/revoke
Content-Type: application/x-www-form-urlencoded

token=abc123...&
client_id=sigil_xxx
```

##  MCP Tools for OAuth Management

### Get OAuth Client Info

```python
oauth_client_info()
```

Returns:
```json
{
  "client_id": "sigil_xxxxxxxxxxxxxxxx",
  "redirect_uris": ["http://localhost:8080/oauth/callback"],
  "authorization_endpoint": "/oauth/authorize",
  "token_endpoint": "/oauth/token",
  "revocation_endpoint": "/oauth/revoke"
}
```

### Manual Authorization (for testing)

```python
oauth_authorize(
    client_id="sigil_xxx",
    redirect_uri="http://localhost:8080/oauth/callback",
    state="random_state",
    code_challenge="challenge_hash",
    code_challenge_method="S256"
)
```

### Manual Token Exchange

```python
oauth_token(
    grant_type="authorization_code",
    code="auth_code",
    redirect_uri="http://localhost:8080/oauth/callback",
    client_id="sigil_xxx",
    code_verifier="verifier_string"
)
```

### Revoke Token

```python
oauth_revoke(
    token="access_token_here",
    client_id="sigil_xxx"
)
```

##  Security Features

### Local Bypass
- Connections from `127.0.0.1` or `::1` automatically bypass authentication
- Safe for local development without compromising remote security

### PKCE Support
- Proof Key for Code Exchange prevents authorization code interception
- Uses SHA-256 hashing of code verifier
- Recommended for all OAuth clients

### Token Security
- Access tokens expire after 1 hour
- Refresh tokens allow obtaining new access tokens
- Tokens stored with secure permissions (0600)
- Constant-time comparison prevents timing attacks

### State Parameter
- CSRF protection for authorization flow
- States expire after 10 minutes
- Validated before issuing authorization codes

### Secure Storage
- OAuth credentials stored in `~/.sigil_mcp_server/oauth/`
- Client secrets hashed before storage
- File permissions restricted to owner only

## Token Lifecycle

1. **Issue**: New access token valid for 1 hour
2. **Use**: Include in `Authorization: Bearer <token>` header
3. **Expire**: After 1 hour, token becomes invalid
4. **Refresh**: Use refresh token to get new access token
5. **Revoke**: Manually revoke if compromised

##  Troubleshooting

### "OAuth not enabled" error

```bash
export SIGIL_MCP_OAUTH_ENABLED=true
python server.py
```

### "Invalid client" error

Make sure you're using the correct `client_id` and `client_secret` shown when the server first started.

To reset OAuth credentials:
```bash
rm -rf ~/.sigil_mcp_server/oauth/
python server.py  # Will generate new credentials
```

### "Invalid authorization code" error

Authorization codes expire after 10 minutes and can only be used once. Request a new authorization code.

### "Token expired" error

Use the refresh token to get a new access token:
```python
oauth_token(
    grant_type="refresh_token",
    refresh_token="your_refresh_token",
    client_id="sigil_xxx"
)
```

### Local connections failing

Make sure local bypass is enabled:
```bash
export SIGIL_MCP_ALLOW_LOCAL_BYPASS=true
```

And connect from `localhost` or `127.0.0.1`.

##  Additional Resources

- [OAuth 2.0 RFC](https://tools.ietf.org/html/rfc6749)
- [PKCE RFC](https://tools.ietf.org/html/rfc7636)
- [OpenAI MCP Documentation](https://platform.openai.com/docs/mcp)

## 🤝 Support

For issues or questions:
1. Check this documentation
2. Review server logs for error messages
3. Verify your environment variables
4. Try resetting OAuth credentials

## Best Practices

1. **Never commit credentials** - Keep `.sigil_mcp_server/` in `.gitignore`
2. **Use HTTPS in production** - ngrok provides this automatically
3. **Rotate tokens regularly** - Revoke and reissue periodically
4. **Enable local bypass** - Simplifies development without compromising security
5. **Monitor access logs** - Check for unauthorized access attempts
6. **Use strong ngrok domains** - Avoid predictable URLs
7. **Keep credentials separate** - Don't share Client ID and Secret together publicly
