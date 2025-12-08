// Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial licenses are available. Contact: davetmire85@gmail.com

import { createContext, useContext, useEffect, useState, ReactNode } from 'react'

export type Theme = 'light' | 'dark' | 'high-contrast'

interface ThemeContextType {
    theme: Theme
    setTheme: (theme: Theme) => void
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

export function ThemeProvider({ children }: { children: ReactNode }) {
    const [theme, setThemeState] = useState<Theme>(() => {
        // Load from localStorage or default to light
        const saved = localStorage.getItem('sigil-admin-theme') as Theme | null
        return saved && ['light', 'dark', 'high-contrast'].includes(saved)
            ? saved
            : 'light'
    })

    useEffect(() => {
        // Apply theme class to root element
        const root = document.documentElement
        root.classList.remove('light', 'dark', 'high-contrast')
        root.classList.add(theme)

        // Save to localStorage
        localStorage.setItem('sigil-admin-theme', theme)
    }, [theme])

    const setTheme = (newTheme: Theme) => {
        setThemeState(newTheme)
    }

    return (
        <ThemeContext.Provider value={{ theme, setTheme }}>
            {children}
        </ThemeContext.Provider>
    )
}

export function useTheme() {
    const context = useContext(ThemeContext)
    if (context === undefined) {
        throw new Error('useTheme must be used within a ThemeProvider')
    }
    return context
}


