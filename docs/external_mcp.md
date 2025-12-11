<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# External MCP Servers with Sigil

Sigil can aggregate remote/local MCP servers and expose their tools with server-prefixed names (e.g., `playwright.click`, `mindsdb.query`). Tools are registered at startup when `external_mcp_servers` is populated.

## Configure `config.json`

```json
{
  "external_mcp_servers": [
    {
      "name": "playwright",
      "type": "streamable-http",
      "url": "http://127.0.0.1:3001/",
      "headers": {
        "authorization": "Bearer <token-if-required>"
      },
      "init_timeout": 10,
      "tool_timeout": 120,
      "verify": true,
      "disabled": false
    },
    {
      "name": "nextjs-mcp",
      "type": "streamable-http",
      "url": "http://127.0.0.1:3002/",
      "headers": {},
      "disabled": false
    },
    {
      "name": "mindsdb",
      "type": "streamable-http",
      "url": "https://cloud.mindsdb.com/api/mcp",
      "headers": {
        "authorization": "Bearer YOUR_MINDSDB_API_KEY"
      },
      "disabled": false
    },
    {
      "name": "local-stdio-example",
      "type": "stdio",
      "command": "python",
      "args": ["./path/to/server.py"],
      "env": {},
      "disabled": true
    }
  ]
}
```

**Key fields**
- `name`: unique prefix for tools (normalized to lowercase/underscores).
- `type`: `stdio`, `sse`, or `streamable-http`.
- `url` or `command`/`args`: required based on type.
- `headers`: include `Authorization: Bearer <token>` for secured servers (MindsDB, tokenized Playwright/Next.js deployments).
- `disabled`: set `true` to keep a server configured but inactive.

**Environment variable override**
- `SIGIL_MCP_SERVERS='[{"name":"playwright","type":"streamable-http","url":"http://127.0.0.1:3001/"}]'`
- Auto-install (dev convenience): set `"external_mcp_auto_install": true` or `SIGIL_MCP_AUTO_INSTALL=true` to run `npx`/`npm`/`bunx` or servers with `"auto_install": true` on startup. Disabled by default.

## Startup & verification
- Tools are discovered and registered at server startup. Check status via:
  - `GET /admin/mcp/status` (admin access).
  - Tools appear as `serverprefix.tool` in the MCP registry (also exposed via `list_mcp_tools` tool).
- Get prompt text for your agent: call `external_mcp_prompt` to receive a human-readable summary of external tools to include in a system prompt.
- If a server requires auth, set headers with your token; unauthenticated servers will fail discovery.

## Client preset
- See `docs/mcp.json` for a ready-to-use MCP client config that includes Sigil, Playwright, Next.js MCP, and MindsDB. Update URLs/tokens to match your environment.

## Best practices
- Keep secrets out of source control; set tokens via env or config injection.
- Use `disabled: true` to stage configs without starting them.
- Prefer streamable HTTP for remote servers; use SSE only if required by the server.
- For stdio servers, ensure the command is executable and in PATH (or absolute).
