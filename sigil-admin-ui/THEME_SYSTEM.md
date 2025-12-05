# Theme System Documentation

## Overview

The Sigil Admin UI now includes a comprehensive theme system with three themes:
- **Light** (default) - Standard light theme
- **Dark** - Dark mode for reduced eye strain
- **High Contrast** - WCAG AAA compliant theme for accessibility

## Implementation Details

### Theme Context (`src/contexts/ThemeContext.tsx`)

- Manages theme state across the application
- Persists theme preference to `localStorage` (key: `sigil-admin-theme`)
- Applies theme class to `document.documentElement` for global styling
- Provides `useTheme()` hook for components to access/change theme

### Theme Picker Component (`src/components/ThemePicker.tsx`)

- Accessible from any page via the header
- Dropdown menu with icons for each theme
- Shows checkmark (✓) for currently active theme
- Uses Lucide React icons:
  - ☀️ Sun icon for Light theme
  - 🌙 Moon icon for Dark theme
  - ⚡ Contrast icon for High Contrast theme

### Theme Definitions (`src/index.css`)

All themes use CSS custom properties (CSS variables) for consistent theming:

#### Light Theme (Default)
- White backgrounds with dark text
- Standard contrast ratios (WCAG AA compliant)
- Modern, clean appearance

#### Dark Theme
- Dark backgrounds (`222.2 84% 4.9%`) with light text (`210 40% 98%`)
- Reduced eye strain in low-light environments
- Maintains good contrast for readability

#### High Contrast Theme
- **WCAG AAA Compliant** - 7:1+ contrast ratios
- Pure black background (`0 0% 0%`) with bright yellow text (`60 100% 50%`)
- Enhanced borders (2px minimum) for maximum visibility
- Bolder text (font-weight: 500) for better readability
- Designed specifically for users with visual impairments

### Key Differences: Dark vs High Contrast

| Feature | Dark Theme | High Contrast Theme |
|---------|-----------|---------------------|
| **Purpose** | Reduce eye strain, modern aesthetic | Maximum accessibility for visual impairments |
| **Background** | Dark gray/blue (`222.2 84% 4.9%`) | Pure black (`0 0% 0%`) |
| **Text Color** | Light gray/white (`210 40% 98%`) | Bright yellow (`60 100% 50%`) |
| **Contrast Ratio** | ~4.5:1 - 7:1 (WCAG AA) | 7:1+ (WCAG AAA) |
| **Borders** | Standard (1px) | Enhanced (2px minimum) |
| **Text Weight** | Normal | Slightly bolder (500) |
| **Color Palette** | Full color range | Limited, high-contrast colors only |

### High Contrast Theme Specifications

The high contrast theme follows WCAG 2.1 Level AAA guidelines:

- **Normal Text**: 7:1 contrast ratio minimum
- **Large Text**: 4.5:1 contrast ratio minimum
- **UI Components**: 3:1 contrast ratio minimum
- **Borders**: 2px minimum width for visibility
- **Focus Indicators**: High-contrast ring colors

Color choices:
- **Background**: Pure black for maximum contrast
- **Foreground**: Bright yellow (HSL: 60 100% 50%) - provides 7:1+ contrast on black
- **Borders**: Bright yellow for maximum visibility
- **Destructive Actions**: Bright red for clear indication

## Usage

### For Users

1. Click the theme icon in the header (top right)
2. Select your preferred theme from the dropdown
3. Theme persists across page refreshes and sessions
4. Theme is applied immediately without page reload

### For Developers

```tsx
import { useTheme } from '../contexts/ThemeContext'

function MyComponent() {
  const { theme, setTheme } = useTheme()
  
  return (
    <div>
      <p>Current theme: {theme}</p>
      <button onClick={() => setTheme('dark')}>Switch to Dark</button>
    </div>
  )
}
```

## Technical Implementation

### Theme Application Flow

1. **Initial Load**: Script in `index.html` applies saved theme before React loads (prevents flash)
2. **React Hydration**: `ThemeProvider` reads from localStorage and applies theme
3. **Theme Change**: User selects theme → `setTheme()` → Updates state → Applies class → Saves to localStorage

### CSS Variable System

All themes use the same CSS variable names, allowing seamless theme switching:

```css
--background
--foreground
--card
--card-foreground
--primary
--primary-foreground
--secondary
--secondary-foreground
--muted
--muted-foreground
--accent
--accent-foreground
--destructive
--destructive-foreground
--border
--input
--ring
```

### Persistence

- Theme preference stored in `localStorage` with key `sigil-admin-theme`
- Automatically loaded on app initialization
- Persists across browser sessions

## Accessibility

- ✅ WCAG AAA compliant high contrast theme
- ✅ Keyboard accessible theme picker
- ✅ Screen reader support (sr-only labels)
- ✅ No flash of unstyled content (FOUC)
- ✅ Smooth theme transitions

## Future Enhancements

Potential improvements:
- System theme detection (respects OS dark/light mode preference)
- Custom theme creation
- Theme preview before applying
- Per-page theme preferences


