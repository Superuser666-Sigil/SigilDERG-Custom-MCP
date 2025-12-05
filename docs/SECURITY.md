<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# Security Quick Reference

## Is My Connection Secure?

**Yes!** Here's what protects you:

### 1. HTTPS Encryption (via ngrok)
- [YES] **ngrok automatically provides TLS/HTTPS** - even on the free tier
- [YES] All traffic between ChatGPT and your server is encrypted
- [YES] No one can sniff your code or API requests in transit

### 2. API Key Authentication (enabled by default)
- [YES] Prevents unauthorized access to your repositories
- [YES] Keys are securely hashed (SHA-256) before storage
- [YES] Only works with the correct API key

### 3. Optional IP Whitelisting
- [YES] Restrict access to known IP addresses
- [YES] Additional layer of security

### 4. Admin API Security
- [YES] Admin API integrated into main server (port 8000, `/admin/*` endpoints)
- [YES] Localhost-only by default (127.0.0.1, ::1)
- [YES] IP whitelist enforcement
- [YES] Optional API key authentication
- [YES] Separate from main MCP server (isolated attack surface)

## What ngrok Free Tier Gives You

| Feature | Free Tier | Notes |
|---------|-----------|-------|
| HTTPS/TLS | Yes | Full encryption |
| Random URL | Yes | Changes on restart |
| Static URL | No | Requires paid plan |
| Custom domain | No | Requires paid plan |
| Rate limits | ~40 req/min | Usually sufficient |

## Security Checklist

- [x] Use HTTPS URL from ngrok (not HTTP)
- [x] Keep API key authentication enabled (default)
- [x] Save your API key securely when generated
- [x] Configure API key in ChatGPT custom connector
- [x] Update ChatGPT URL when ngrok restarts
- [ ] Optional: Set IP whitelist if you know your source IPs
- [ ] Optional: Upgrade ngrok for static URL
- [ ] Optional: Set Admin API key if exposing Admin API beyond localhost
- [ ] Optional: Configure Admin API allowed_ips if needed

## Common Questions

**Q: Is my code exposed to the internet?**
A: Only if someone has your API key. Authentication is enabled by default.

**Q: Is the Admin API secure?**
A: Yes. The Admin API is localhost-only by default and uses IP whitelisting. If you need to expose it beyond localhost, set an API key for additional security.

**Q: Are my credentials logged?**
A: No. Sensitive headers (authorization, cookies, API keys) are automatically redacted before logging. Only non-sensitive headers are logged for debugging purposes.

**Q: Can someone intercept my traffic?**
A: No, ngrok uses HTTPS/TLS encryption.

**Q: What if I lose my API key?**
A: Run `python manage_auth.py reset` to generate a new one.

**Q: Do I need to pay for ngrok?**
A: No, the free tier provides HTTPS. Paid plans offer convenience (static URLs) but aren't required for security.

**Q: What happens if someone gets my ngrok URL?**
A: They still need your API key to access the server. Keep your API key secret!

**Q: Should I disable authentication for local testing?**
A: Yes, you can set `export SIGIL_MCP_AUTH_ENABLED=false` for local testing, but **never disable it when using ngrok**.

## Best Practices

1. **Always use HTTPS URLs** from ngrok (they start with `https://`)
2. **Never commit API keys** to git or share them publicly
3. **Rotate API keys periodically** using `python manage_auth.py reset`
4. **Use environment variables** to store API keys, not hardcoded values
5. **Monitor server logs** for authentication failures (potential attacks)
6. **Consider IP whitelisting** if you know your source IPs

## Security Limitations (and workarounds)

### Free Tier Limitations
- URL changes on ngrok restart
  - **Workaround**: Update ChatGPT config each time
  - **Better solution**: Upgrade to ngrok paid plan for static URLs

- No custom domain
  - **Workaround**: Not needed for security, just convenience
  - **Better solution**: Upgrade to ngrok paid plan

### What This Setup Does NOT Protect Against
- Compromised API keys (keep them secret!)
- Malicious code in your repositories (don't index untrusted code)
- Server vulnerabilities (keep dependencies updated)

## Testing Your Security Setup

```bash
# Should FAIL without API key
curl https://your-ngrok-url.ngrok-free.app

# Should SUCCEED with API key
curl -H "X-API-Key: your_api_key_here" https://your-ngrok-url.ngrok-free.app

# Check authentication in logs
tail -f server.log | grep -i auth
```

## If You're Still Concerned

### Option 1: Use SSH Tunneling Instead
Set up a reverse SSH tunnel instead of ngrok:
```bash
ssh -R 8000:localhost:8000 your-remote-server.com
```
Pros: More control, no third party
Cons: Need a server, more complex

### Option 2: Run Locally Only
Don't expose to internet at all. Use MCP locally with desktop clients.

### Option 3: Self-Host on a VPS
Deploy to a cloud server with proper firewall rules and HTTPS certificates.
