import { Moon, Sun, Contrast } from 'lucide-react'
import { useTheme } from '../contexts/ThemeContext'
import { Button } from './ui/button'
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from './ui/dropdown-menu'

export function ThemePicker() {
    const { theme, setTheme } = useTheme()

    const themes: { value: 'light' | 'dark' | 'high-contrast'; label: string; icon: React.ReactNode }[] = [
        { value: 'light', label: 'Light', icon: <Sun className="h-4 w-4" /> },
        { value: 'dark', label: 'Dark', icon: <Moon className="h-4 w-4" /> },
        { value: 'high-contrast', label: 'High Contrast', icon: <Contrast className="h-4 w-4" /> },
    ]

    const currentTheme = themes.find(t => t.value === theme) || themes[0]

    return (
        <DropdownMenu>
            <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="h-9 w-9">
                    {currentTheme.icon}
                    <span className="sr-only">Change theme</span>
                </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
                {themes.map((themeOption) => (
                    <DropdownMenuItem
                        key={themeOption.value}
                        onClick={() => setTheme(themeOption.value)}
                        className="flex items-center gap-2"
                    >
                        {themeOption.icon}
                        <span>{themeOption.label}</span>
                        {theme === themeOption.value && (
                            <span className="ml-auto text-xs">✓</span>
                        )}
                    </DropdownMenuItem>
                ))}
            </DropdownMenuContent>
        </DropdownMenu>
    )
}

