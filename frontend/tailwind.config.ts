import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Electric cyan accent — replaces generic blue
        brand: {
          50:  '#e6faff',
          100: '#b3f0ff',
          300: '#33dcff',
          500: '#00d4ff',
          600: '#00b8e0',
          700: '#0099c0',
        },
        // Dark surface palette
        surface: {
          base:     '#0d0f14',
          card:     '#141720',
          elevated: '#1a1f2e',
          border:   '#252b3b',
          muted:    '#2e3548',
        },
      },
      fontFamily: {
        sans: ['Outfit', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      boxShadow: {
        'card-dark': '0 1px 3px rgba(0,0,0,0.4), 0 0 0 1px rgba(37,43,59,0.8)',
        'glow-cyan': '0 0 12px rgba(0,212,255,0.25)',
      },
      animation: {
        'fade-in':   'fadeIn 0.15s ease-out',
        'slide-down':'slideDown 0.18s ease-out',
        'card-in':   'cardIn 0.25s cubic-bezier(0.16,1,0.3,1) both',
        'chip-pulse':'chipPulse 0.2s ease-out',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideDown: {
          '0%': { opacity: '0', transform: 'translateY(-4px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        cardIn: {
          '0%': { opacity: '0', transform: 'translateY(6px) scale(0.98)' },
          '100%': { opacity: '1', transform: 'translateY(0) scale(1)' },
        },
        chipPulse: {
          '0%':   { transform: 'scale(1)' },
          '50%':  { transform: 'scale(0.95)' },
          '100%': { transform: 'scale(1)' },
        },
      },
    },
  },
  plugins: [],
} satisfies Config
