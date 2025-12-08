<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# Fixing 502 Bad Gateway Errors with Cloudflare Tunnel

## Problem

When connecting ChatGPT to your MCP server through Cloudflare Tunnel, you may experience consistent **502 Bad Gateway** errors. This is typically caused by Cloudflare buffering HTTP responses.

## Root Cause

Cloudflare (and Nginx) buffer HTTP responses by default. For MCP servers that need immediate response delivery (especially for streaming or real-time communication), this buffering can cause:

1. **Connection timeouts** - Cloudflare waits for the full response before sending, causing timeouts
2. **502 errors** - If the connection times out while buffering, Cloudflare returns 502
3. **Delayed responses** - Even if successful, responses are delayed

## Solution

The Sigil MCP Server automatically adds headers to prevent buffering:

### Headers Added Automatically

The server's `HeaderLoggingASGIMiddleware` now adds these headers to all responses:

- **`X-Accel-Buffering: no`** - Instructs Cloudflare/Nginx not to buffer the response
- **`Cache-Control: no-cache, no-store, must-revalidate`** - Prevents caching of responses
- **`Connection: keep-alive`** - Keeps the connection alive for better performance

These headers are added automatically - no configuration needed.

### Verification

To verify the headers are being sent:

```bash
# Test the endpoint
curl -I https://mcp.yourdomain.com/

# Look for these headers in the response:
# x-accel-buffering: no
# cache-control: no-cache, no-store, must-revalidate
# connection: keep-alive
```

## Additional Cloudflare Configuration

### Cloudflare Tunnel Config

If you're still experiencing issues, you can add explicit configuration to your `~/.cloudflared/config.yml`:

```yaml
tunnel: sigil-mcp
credentials-file: /home/username/.cloudflared/<TUNNEL-ID>.json

ingress:
  - hostname: mcp.yourdomain.com
    service: http://localhost:8000
    originRequest:
      connectTimeout: 30s
      tcpKeepAlive: 30s
      noHappyEyeballs: false
      httpHostHeader: mcp.yourdomain.com
  - service: http_status:404
```

### Cloudflare Dashboard Settings

1. **Disable Bot Fight Mode** (if enabled):
   - Go to **Security** → **Bots**
   - Set **Bot Fight Mode** to **Off**
   - This can interfere with OAuth authentication

2. **Check Page Rules**:
   - Go to **Rules** → **Page Rules**
   - Ensure no rules are caching or buffering your MCP endpoint

3. **SSL/TLS Settings**:
   - Go to **SSL/TLS** → **Overview**
   - Set to **Full** or **Full (strict)**
   - This ensures proper HTTPS termination

## Testing

After applying the fix:

1. **Restart the MCP server**:
   ```bash
   sudo systemctl restart sigil-mcp
   ```

2. **Restart Cloudflare Tunnel** (if needed):
   ```bash
   sudo systemctl restart cloudflared
   ```

3. **Test the connection**:
   ```bash
   curl -v https://mcp.yourdomain.com/
   ```

4. **Check headers**:
   ```bash
   curl -I https://mcp.yourdomain.com/ | grep -i "x-accel-buffering\|cache-control\|connection"
   ```

## Troubleshooting

### Still Getting 502 Errors?

1. **Check server logs**:
   ```bash
   journalctl -u sigil-mcp -f
   ```

2. **Check tunnel logs**:
   ```bash
   sudo journalctl -u cloudflared -f
   ```

3. **Verify server is responding locally**:
   ```bash
   curl http://localhost:8000/
   ```

4. **Check Cloudflare Analytics**:
   - Go to Cloudflare Dashboard → **Analytics** → **HTTP Requests**
   - Look for error patterns

### Connection Timeouts

If you're seeing timeouts:

1. **Increase timeout in tunnel config** (requires paid plan for some settings)
2. **Check server performance** - ensure it's not overloaded
3. **Monitor response times** in server logs

## Related Documentation

- [CLOUDFLARE_TUNNEL.md](CLOUDFLARE_TUNNEL.md) - Complete Cloudflare Tunnel setup
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - General troubleshooting guide
- [CHATGPT_SETUP.md](CHATGPT_SETUP.md) - ChatGPT integration guide

## References

- [Cloudflare Tunnel Documentation](https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/)
- [Nginx X-Accel-Buffering](https://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_buffering)
- [Server-Sent Events Best Practices](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)


