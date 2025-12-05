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


