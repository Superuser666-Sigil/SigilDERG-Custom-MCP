<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# Cloudflare Tunnel Deployment

## Overview

Cloudflare Tunnel provides a secure way to expose your local MCP server to the internet without:
- Requiring a static IP address
- Opening inbound firewall ports
- Dealing with dynamic ngrok URLs
- Running a public-facing server

This is ideal for connecting ChatGPT or other remote clients to your MCP server while keeping your repositories local with real-time file watching.

## Why Use Cloudflare Tunnel?

### Problems It Solves

1. **No Static IP Required**: Most ISPs don't offer static IPs to residential customers
2. **Persistent URL**: Unlike ngrok's free tier, your URL never changes
3. **Professional Domain**: Use your own domain (`mcp.yourdomain.com`) instead of random subdomains
4. **Better Security**: Outbound-only connections, DDoS protection, WAF available
5. **Always-On Service**: Runs as systemd service, survives reboots
6. **Keep Repos Local**: Index your local repositories with file watching enabled

### Comparison with Alternatives

| Feature | Cloudflare Tunnel | ngrok Free | Tailscale Funnel | Remote Hosting |
|---------|------------------|------------|------------------|----------------|
| Custom domain | Yes | Random | `.ts.net` only | Yes |
| Static URL | Always | Changes | Always | Always |
| Cost | Free | Free | Free | $5-20/mo |
| Local repos | Yes | Yes | Yes | No |
| File watching | Yes | Yes | Yes | No |
| Setup complexity | Moderate | Simple | Simple | Complex |
| DDoS protection | Yes | Basic | Basic | Depends |
| Bandwidth limit | None | Limited | None | Depends |

## Prerequisites

1. **Domain name** registered with any registrar (e.g., Namecheap, GoDaddy, Cloudflare)
2. **Free Cloudflare account** at https://dash.cloudflare.com
3. **Linux system** with systemd (Ubuntu, Debian, etc.)
4. **MCP server running** locally on a port (default: 8000)

## Setup Instructions

### Step 1: Add Domain to Cloudflare

1. Log in to https://dash.cloudflare.com
2. Click **"Add a site"**
3. Enter your domain (e.g., `yourdomain.com`)
4. Select the **Free plan**
5. Cloudflare will scan your DNS records
6. Click **"Continue"**
7. Note the **nameservers** Cloudflare provides (e.g., `alice.ns.cloudflare.com`)

### Step 2: Update Nameservers at Your Registrar

1. Log in to your domain registrar (Namecheap, GoDaddy, etc.)
2. Find **DNS Management** or **Nameservers** for your domain
3. Change to **Custom Nameservers**
4. Replace with Cloudflare's nameservers
5. Save changes
6. **Wait 5-30 minutes** for DNS propagation (Cloudflare will email when active)

### Step 3: Install cloudflared

```bash
# Download latest cloudflared
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb

# Install the package
sudo dpkg -i cloudflared-linux-amd64.deb

# Verify installation
cloudflared --version
```

### Step 4: Authenticate with Cloudflare

```bash
# This will open a browser window for authorization
cloudflared tunnel login
```

You should see: `You have successfully logged in.`

Credentials are saved to `~/.cloudflared/cert.pem`

### Step 5: Create the Tunnel

```bash
# Create a named tunnel (choose any name)
cloudflared tunnel create sigil-mcp
```

Output will show:
```
Tunnel credentials written to /home/username/.cloudflared/<TUNNEL-ID>.json
Created tunnel sigil-mcp with id <TUNNEL-ID>
```

**Save the tunnel ID** - you'll need it for the config file.

### Step 6: Route DNS to Tunnel

```bash
# Create DNS record pointing to your tunnel
cloudflared tunnel route dns sigil-mcp mcp.yourdomain.com
```

This automatically creates a CNAME record in Cloudflare DNS.

### Step 7: Configure the Tunnel

Create `~/.cloudflared/config.yml`:

```yaml
tunnel: sigil-mcp
credentials-file: /home/username/.cloudflared/<TUNNEL-ID>.json

ingress:
  - hostname: mcp.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
```

**Replace:**
- `<TUNNEL-ID>` with your actual tunnel ID
- `yourdomain.com` with your domain
- `8000` with your MCP server port if different

### Step 8: Test the Tunnel

```bash
# Run tunnel in foreground to test
cloudflared tunnel run sigil-mcp
```

You should see:
```
Registered tunnel connection connIndex=0
```

Test in another terminal:
```bash
curl https://mcp.yourdomain.com
```

You should get an HTTP response (406 is expected from FastAPI without proper headers).

Press `Ctrl+C` to stop the test.

### Step 9: Install as Service

```bash
# Install cloudflared as systemd service
sudo cloudflared --config /home/username/.cloudflared/config.yml service install

# Enable service to start on boot
sudo systemctl enable cloudflared


# Start the service
sudo systemctl start cloudflared

# Check service status
sudo systemctl status cloudflared
```

You should see: `Active: active (running)`

### Step 10: Update MCP Server Config

Update your `config.json` to include your domain in `allowed_hosts`:

```json
{
    "server": {
        "allowed_hosts": [
            "*",
            "127.0.0.1",
            "localhost",
            "mcp.yourdomain.com"
        ]
    }
}
```

Restart your MCP server:
```bash
# If running with systemd
sudo systemctl restart sigil-mcp

# Or if running manually
pkill -f "python.*sigil_mcp" && python -m sigil_mcp
```

## Configuration for ChatGPT

Update your ChatGPT GPT Action with:

**OpenAPI Schema:**
```json
{
  "openapi": "3.1.0",
  "info": {
    "title": "Sigil MCP Server",
    "version": "1.0.1"
  },
  "servers": [
    {
      "url": "https://mcp.yourdomain.com"
    }
  ]
}
```

**OAuth 2.0 Settings:**
- **Authorization URL:** `https://mcp.yourdomain.com/oauth/authorize`
- **Token URL:** `https://mcp.yourdomain.com/oauth/token`
- **Client ID:** (from your OAuth setup)
- **Client Secret:** (from your OAuth setup)
- **Scope:** `read write`
- **Callback URL:** (provided by ChatGPT, starts with `https://chatgpt.com/...`)

## Advanced Configuration

### Multiple Services

Route different subdomains to different services:

```yaml
tunnel: sigil-mcp
credentials-file: /home/username/.cloudflared/<TUNNEL-ID>.json

ingress:
  - hostname: mcp.yourdomain.com
    service: http://localhost:8000
  - hostname: api.yourdomain.com
    service: http://localhost:3000
  - hostname: docs.yourdomain.com
    service: http://localhost:4000
  - service: http_status:404
```

Don't forget to route each subdomain:
```bash
cloudflared tunnel route dns sigil-mcp api.yourdomain.com
cloudflared tunnel route dns sigil-mcp docs.yourdomain.com
```

### Access Control

Add Cloudflare Access rules to restrict who can connect:

1. Go to **Cloudflare Dashboard** → **Zero Trust** → **Access**
2. Create an **Application**
3. Set domain: `mcp.yourdomain.com`
4. Add policies (IP ranges, email domains, etc.)

Example policy: Only allow ChatGPT IPs
```
IP ranges:
- 23.102.140.0/24 (ChatGPT)
- Your home IP
```

### Connection Options

Fine-tune tunnel behavior in `config.yml`:

```yaml
tunnel: sigil-mcp
credentials-file: /home/username/.cloudflared/<TUNNEL-ID>.json

ingress:
  - hostname: mcp.yourdomain.com
    service: http://localhost:8000
    originRequest:
      connectTimeout: 30s
      noTLSVerify: false
      httpHostHeader: mcp.yourdomain.com
  - service: http_status:404
```

## Troubleshooting

### Tunnel Won't Start

Check service logs:
```bash
sudo journalctl -u cloudflared -f
```

Common issues:
- **Config file not found**: Ensure path in service is correct
- **Credentials invalid**: Run `cloudflared tunnel login` again
- **Port not accessible**: Check MCP server is running on specified port

### DNS Not Resolving

Check DNS propagation:
```bash
dig mcp.yourdomain.com
nslookup mcp.yourdomain.com
```

Should show a CNAME record pointing to `<TUNNEL-ID>.cfargotunnel.com`

If missing:
```bash
cloudflared tunnel route dns sigil-mcp mcp.yourdomain.com
```

### Connection Refused

1. **Check MCP server is running:**
   ```bash
   curl http://localhost:8000
   ```

2. **Check tunnel status:**
   ```bash
   sudo systemctl status cloudflared
   ```

3. **Check firewall (if using one):**
   ```bash
   sudo ufw status
   # Cloudflare Tunnel only needs OUTBOUND, no inbound ports
   ```

### 502 Bad Gateway

**Common Causes:**

1. **MCP server is down** - Check if server is running:
   ```bash
   # Check if server is running
   ps aux | grep sigil_mcp
   
   # Check server logs
   journalctl -u sigil-mcp -f
   
   # Restart MCP server
   sudo systemctl restart sigil-mcp
   ```

2. **Cloudflare buffering responses** - The server automatically adds headers to prevent buffering:
   - `X-Accel-Buffering: no` - Tells Cloudflare/Nginx not to buffer
   - `Cache-Control: no-cache, no-store, must-revalidate` - Prevents caching
   - `Connection: keep-alive` - Keeps connection alive
   
   These headers are added automatically by the server's middleware. If you're still getting 502 errors:
   - Ensure you're running the latest version of the server
   - Check server logs for any errors
   - Verify the tunnel is connected: `sudo systemctl status cloudflared`

3. **Connection timeout** - Cloudflare may timeout if the server takes too long to respond:
   - Check server response times in logs
   - Ensure the server isn't overloaded
   - Consider increasing Cloudflare timeout settings (requires paid plan)

### ICMP Proxy Warnings

These warnings are safe to ignore:
```
WRN ICMP proxy feature is disabled
```

ICMP (ping) proxy is optional and not needed for HTTP/HTTPS tunnels.

## Monitoring

### Check Tunnel Status

```bash
# Service status
sudo systemctl status cloudflared

# Live logs
sudo journalctl -u cloudflared -f

# List all tunnels
cloudflared tunnel list
```

### Cloudflare Dashboard

View tunnel metrics at:
- https://dash.cloudflare.com → **Traffic** → **Analytics**
- See request count, bandwidth, error rates
- Monitor for suspicious activity

## Security Considerations

### Best Practices

1. **Keep credentials secure**: 
   - `~/.cloudflared/` should be `chmod 700`
   - Credential files should be `chmod 600`

2. **Use OAuth authentication**: Always enable OAuth for public endpoints

3. **Monitor access logs**: Check Cloudflare dashboard regularly

4. **Update regularly**: 
   ```bash
   sudo apt update && sudo apt upgrade cloudflared
   ```

5. **Use Cloudflare Access**: Add IP restrictions or email-based auth

### What's Protected

- No inbound firewall rules needed
- Your home IP stays hidden
- DDoS attacks absorbed by Cloudflare
- Automatic HTTPS with valid certificate
- Connection between you and Cloudflare encrypted

### What's NOT Protected

- Anyone can reach your MCP server URL (use OAuth!)
- Cloudflare can see your traffic (it's their proxy)
- Backend (MCP server) security is still your responsibility

## Cost

Cloudflare Tunnel is **100% free** for:
- Unlimited bandwidth
- Unlimited tunnels
- Unlimited domains
- Basic DDoS protection

Paid features (not required):
- Load Balancing: $5/month
- Advanced DDoS: Enterprise only
- Argo Smart Routing: $5/month + usage

## Alternative: ngrok

If you don't have a domain or prefer simpler setup, use ngrok:

```bash
# Install ngrok
sudo snap install ngrok

# Authenticate
ngrok authtoken YOUR_TOKEN

# Start tunnel
ngrok http 8000
```

**Pros:**
- Simpler setup (no domain needed)
- Works immediately

**Cons:**
- Random URLs that change on restart (free tier)
- Less professional
- Limited to 1 tunnel on free tier

For production use, Cloudflare Tunnel is recommended.

## References

- **Cloudflare Tunnel Docs**: https://developers.cloudflare.com/cloudflare-one/connections/connect-apps/
- **cloudflared GitHub**: https://github.com/cloudflare/cloudflared
- **DNS Propagation Checker**: https://dnschecker.org/

## Support

For issues specific to:
- **Cloudflare Tunnel**: https://community.cloudflare.com/
- **Sigil MCP Server**: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **ChatGPT Integration**: See [CHATGPT_SETUP.md](CHATGPT_SETUP.md)
