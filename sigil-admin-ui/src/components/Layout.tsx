// Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial licenses are available. Contact: davetmire85@gmail.com

import { ReactNode } from 'react'
import { Header } from './Header'
import { AlertCircle } from 'lucide-react'

interface LayoutProps {
  children: ReactNode
}

export function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-background">
      {import.meta.env.DEV && (
        <div className="fixed left-1/2 top-3 z-50 -translate-x-1/2">
          <div className="inline-flex items-center gap-2 rounded-full bg-red-600 px-3 py-1 text-white text-sm font-semibold shadow">
            <AlertCircle className="h-4 w-4" />
            <span>DEV</span>
          </div>
        </div>
      )}
      <Header />
      <main className="container mx-auto px-4 py-8">
        {children}
      </main>
    </div>
  )
}


