<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# ChatGPT MCP OAuth Configuration

## OAuth Credentials

Your server has OAuth enabled and credentials are stored at:
`~/.sigil_mcp_server/oauth/client.json`

## ChatGPT Configuration

When setting up your MCP connector in ChatGPT, use these **exact values**:

### Server URL
```
https://your-domain.example.com
```

### Authentication Type
Select: **OAuth**

### OAuth Settings

**Authorization URL:**
```
https://your-domain.example.com/oauth/authorize
```

**Token URL:**
```
https://your-domain.example.com/oauth/token
```

**Client ID:**
```
<your-client-id-from-client.json>
```

**Client Secret:**
```
<your-client-secret-from-client.json>
```

**Scope:** (leave empty or use default)
```

```

## Allowed Redirect URIs

Your server accepts these redirect URIs in `~/.sigil_mcp_server/oauth/client.json`:
- `http://localhost:8080/oauth/callback` (for local testing)
- `https://chatgpt.com/aip/oauth/callback`
- `https://chat.openai.com/aip/oauth/callback`
- `https://chatgpt.com/connector_platform_oauth_redirect` ← **Currently used by ChatGPT**

## Common Issues

### 1. "Something went wrong" Error
- **Cause:** Cloudflare Bot Fight Mode blocking ChatGPT's backend token exchange
- **Fix:** Disable Bot Fight Mode in Cloudflare Dashboard → Security → Bots
- **Details:** See [CLOUDFLARE_OAUTH_ISSUE.md](CLOUDFLARE_OAUTH_ISSUE.md)

### 2. Invalid Redirect URI
- **Cause:** ChatGPT's redirect URI not in allowed list
- **Fix:** Verify `~/.sigil_mcp_server/oauth/client.json` contains `https://chatgpt.com/connector_platform_oauth_redirect`

### 3. Token Verification Failed
- **Cause:** Expired or invalid tokens
- **Fix:** Clear tokens: `rm ~/.sigil_mcp_server/oauth/tokens.json` and reconnect

## Testing OAuth Flow

1. Test metadata endpoint:
```bash
curl https://your-domain.example.com/.well-known/oauth-authorization-server
```

2. Verify consent screen appears:
   - Navigate to authorization endpoint in browser
   - Should see purple gradient consent page
   - Click "Authorize" to test flow

3. Check server logs:
```bash
tail -f ~/Documents/sigil_server.log
```

## Server Status

Check status:
```bash
# If using Cloudflare Tunnel
sudo systemctl status cloudflared

# Check MCP server
ps aux | grep "python.*sigil_mcp"
```

## Next Steps

1. Get your OAuth credentials:
```bash
cat ~/.sigil_mcp_server/oauth/client.json
```

2. Create new ChatGPT connector with your values
3. Complete OAuth authorization flow
4. Verify connection in ChatGPT

## Troubleshooting

If still having issues:

1. Restart MCP server:
```bash
pkill -f "python.*sigil_mcp"
cd /home/dave/dev/sigil-mcp-server
python -m sigil_mcp.server
```

2. Check Cloudflare Tunnel (if using):
```bash
sudo systemctl status cloudflared
```

3. Verify endpoints are accessible:
```bash
curl -I https://your-domain.example.com/oauth/authorize
curl -I https://your-domain.example.com/oauth/token
```

## References

- [MCP OAuth Documentation](https://modelcontextprotocol.io/docs/concepts/authentication)
- [Cloudflare Bot Fight Mode Issue](CLOUDFLARE_OAUTH_ISSUE.md)
- Server Code: `sigil_mcp/oauth.py`
- Server Routes: `sigil_mcp/server.py`
