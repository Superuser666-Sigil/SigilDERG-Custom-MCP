// Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial licenses are available. Contact: davetmire85@gmail.com

import { Link, useLocation } from 'react-router-dom'
import { Activity } from 'lucide-react'
import { ThemePicker } from './ThemePicker'

export function Header() {
  const location = useLocation()

  return (
    <header className="border-b bg-background">
      <div className="container flex h-16 items-center px-4">
        <Link to="/" className="flex items-center space-x-2">
          <Activity className="h-6 w-6" />
          <span className="text-xl font-bold">Sigil MCP Admin</span>
        </Link>
        <nav className="ml-auto flex items-center space-x-4">
          <Link
            to="/status"
            className={`text-sm font-medium transition-colors hover:text-primary ${location.pathname === '/status' ? 'text-primary' : 'text-muted-foreground'
              }`}
          >
            Status
          </Link>
          <Link
            to="/index"
            className={`text-sm font-medium transition-colors hover:text-primary ${location.pathname === '/index' ? 'text-primary' : 'text-muted-foreground'
              }`}
          >
            Index
          </Link>
          <Link
            to="/vector"
            className={`text-sm font-medium transition-colors hover:text-primary ${location.pathname === '/vector' ? 'text-primary' : 'text-muted-foreground'
              }`}
          >
            Vector
          </Link>
          <Link
            to="/logs"
            className={`text-sm font-medium transition-colors hover:text-primary ${location.pathname === '/logs' ? 'text-primary' : 'text-muted-foreground'
              }`}
          >
            Logs
          </Link>
          <Link
            to="/config"
            className={`text-sm font-medium transition-colors hover:text-primary ${location.pathname === '/config' ? 'text-primary' : 'text-muted-foreground'
              }`}
          >
            Config
          </Link>
          <ThemePicker />
        </nav>
      </div>
    </header>
  )
}


