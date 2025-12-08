<!--
Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
Commercial licenses are available. Contact: davetmire85@gmail.com
-->

# Sigil MCP Admin UI

A modern React-based admin interface for managing the Sigil MCP Server.

## Features

- **Status Dashboard** - View server status, repositories, index info, and watcher status
- **Index Management** - View statistics and rebuild trigram/symbol indexes
- **Vector Index** - Manage semantic search embeddings
- **Logs Viewer** - Real-time log viewing with auto-refresh
- **Configuration Viewer** - View current server configuration

## Prerequisites

- Node.js 18+ and npm
- Sigil MCP Admin API running on `http://127.0.0.1:8765`

## Installation

```bash
cd sigil-admin-ui
npm install
```

## Development

Start the development server:

```bash
npm run dev
```

The UI will be available at `http://localhost:5173`

## Building for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

## Configuration

The Admin UI connects to the Admin API at `http://127.0.0.1:8765` by default. This is configured in `src/utils/api.ts`.

If your Admin API requires an API key, you can set it in the browser's localStorage:

```javascript
localStorage.setItem('admin_api_key', 'your-api-key')
```

Set a custom Admin API base URL through Vite env variables:

```bash
# .env.local
VITE_ADMIN_API_BASE_URL=https://sigil.example.com/admin
```

During development, the Vite dev server proxies `/admin/*` calls to `http://127.0.0.1:8000` (the MCP server). For standalone deployments, build the UI and serve it behind the same hostname as the MCP server or configure a reverse proxy to rewrite `/admin` requests.

## Testing & Coverage

The Admin UI ships with Vitest + Testing Library suites that exercise all critical flows (Status, Index, Vector, Logs, Config pages; API utilities; shared dialogs). To keep the UI reliable alongside the server:

1. Install deps and run the suite:
   ```bash
   npm install
   npm run test -- --coverage
   ```
2. Maintain overall coverage ≥70%.
3. Keep the critical flows (Status/Index/Vector/Logs/Config pages and `src/utils/api.ts`) at 100% line coverage.
4. When adding timers or async UI, prefer deterministic patterns (e.g., storing interval IDs, guarding browser-only APIs) so tests stay stable without fake timers.

See [../docs/adr-014-admin-ui-testing.md](../docs/adr-014-admin-ui-testing.md) for the rationale behind these guardrails.

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool and dev server
- **Tailwind CSS** - Styling
- **shadcn/ui** - UI components
- **React Router** - Navigation
- **Axios** - HTTP client

## Project Structure

```
sigil-admin-ui/
├── src/
│   ├── components/      # Reusable components
│   │   ├── ui/         # shadcn/ui components
│   │   └── ...         # Layout and shared components
│   ├── pages/          # Page components
│   ├── types/          # TypeScript type definitions
│   ├── utils/          # Utility functions (API client)
│   ├── App.tsx         # Main app component with routing
│   └── main.tsx        # Entry point
├── public/             # Static assets
└── package.json        # Dependencies
```

## License

Same as the main Sigil MCP Server project (AGPL-3.0-or-later).


