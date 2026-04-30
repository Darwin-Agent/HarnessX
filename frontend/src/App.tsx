import { useEffect } from 'react'
import { Routes, Route, Navigate } from 'react-router-dom'
import { TopBar } from './components/layout/TopBar'
import { BuilderPage } from './pages/BuilderPage'
import { ComparePage } from './pages/ComparePage'
import { SettingsSheet } from './components/config/SettingsSheet'
import { ModelSheet }    from './components/config/ModelSheet'
import { DocsPanel } from './components/docs/DocsPanel'
import { useUIStore } from './store/ui'

export default function App() {
  const theme    = useUIStore((s) => s.theme)
  const fontSize = useUIStore((s) => s.fontSize)

  // Sync theme class to <html> for CSS var + Tailwind dark: variant support
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
  }, [theme])

  // Apply font scale as a CSS variable on :root
  useEffect(() => {
    document.documentElement.style.setProperty('--font-scale', String(fontSize ?? 1))
  }, [fontSize])

  return (
    <div className="flex flex-col h-screen" style={{ background: 'var(--bg-base)' }}>
      <TopBar />
      <main className="flex flex-1 min-h-0 overflow-hidden">
        <Routes>
          <Route path="/" element={<Navigate to="/builder" replace />} />
          <Route path="/builder" element={<BuilderPage />} />
          <Route path="/compare" element={<ComparePage />} />
        </Routes>
      </main>
      <ModelSheet />
      <SettingsSheet />
      <DocsPanel />
    </div>
  )
}
