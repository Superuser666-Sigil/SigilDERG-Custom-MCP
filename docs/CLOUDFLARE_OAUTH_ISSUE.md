<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# Cloudflare OAuth Issue - RESOLVED

## Problem
ChatGPT OAuth flow fails with Cloudflare Tunnel but works with ngrok.

## Symptoms
1. Authorization endpoint works (consent screen displays) ✓
2. User clicks "Authorize" → 302 redirect with code ✓  
3. Token exchange request **never arrives** at server ✗
4. ChatGPT shows "Something went wrong" error

## Root Cause ✅ CONFIRMED
**Cloudflare's Bot Fight Mode blocks ChatGPT's automated backend token exchange requests.**

The browser-based OAuth steps (authorization, consent) work fine because they use a normal browser user-agent. However, when ChatGPT's backend attempts the token exchange using `Python/3.12 aiohttp/3.11.18`, Cloudflare's Bot Fight Mode identifies it as automated traffic and blocks it.

### Evidence from Logs

**Ngrok (Working):**
```
INFO: 167.160.247.4:0 - "GET /oauth/authorize..." 200 OK
INFO: 167.160.247.4:0 - "POST /oauth/authorize" 302 Found
INFO: 52.173.123.8:0 - "POST /oauth/token" 200 OK  ← Token exchange arrives
```

**Cloudflare (Failing):**
```
INFO: 167.160.247.4:0 - "GET /oauth/authorize..." 200 OK
INFO: 167.160.247.4:0 - "POST /oauth/authorize" 302 Found
(no token exchange request - ChatGPT's backend is blocked)
```

### Key Differences
- **Source IP for token exchange:** `52.173.123.*` (Microsoft Azure - ChatGPT backend)
- **User-Agent:** `Python/3.12 aiohttp/3.11.18` (automated request, not browser)
- **Cloudflare Bot Fight Mode:** Blocks automated requests with non-browser user-agents
- **Ngrok:** No bot protection, proxies all traffic transparently

## Solution ✅ VERIFIED WORKING

**Disable Bot Fight Mode in Cloudflare Dashboard:**

1. Log into Cloudflare Dashboard
2. Select your domain (`sigilderg.tech`)
3. Navigate to **Security → Bots**
4. Set **Bot Fight Mode** to **Off**
5. Save changes

**Result:** ChatGPT's OAuth token exchange (`POST /oauth/token`) now successfully reaches your server through Cloudflare Tunnel.

## Alternative Solutions (if you need bot protection)

If you want to keep Bot Fight Mode enabled for other endpoints but allow ChatGPT OAuth:

### Option 1: Cloudflare WAF Rule (Recommended if keeping Bot Fight Mode)
Add a WAF rule to bypass bot protection for OAuth endpoints:
1. Go to Cloudflare dashboard → Security → WAF
2. Create Custom Rule:
   - **Rule name:** "Allow ChatGPT OAuth"
   - **If:** `(http.request.uri.path eq "/oauth/token" and http.user_agent contains "Python")`
   - **Then:** Skip → Bot Fight Mode

### Option 2: IP Allow List
Add ChatGPT's Azure IP ranges to allow list:
- Common ranges: `52.173.123.0/24`, `40.88.0.0/16`, etc.
- Note: OpenAI/ChatGPT may use multiple Azure regions

## Testing & Verification

After disabling Bot Fight Mode, test the complete OAuth flow:

1. Create new ChatGPT MCP connector with `https://mcp.sigilderg.tech`
2. Complete OAuth authorization (consent screen)
3. Verify successful token exchange in server logs:
   ```
   INFO: 52.173.123.x:0 - "POST /oauth/token" 200 OK
   Issued access token for client sigil_xxx...
   ```
4. Confirm ChatGPT connector shows as "Connected"

## References
- **Issue Status:** ✅ RESOLVED (December 3, 2025)
- **Solution:** Disable Cloudflare Bot Fight Mode
- ChatGPT backend IPs: Azure datacenter ranges (`52.173.*`, `40.88.*`, etc.)
- Cloudflare Bot Fight Mode: https://developers.cloudflare.com/bots/get-started/free/
- Cloudflare Tunnel docs: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/
- Cloudflare WAF: https://developers.cloudflare.com/waf/
