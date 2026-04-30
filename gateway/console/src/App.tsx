import { useEffect, useState } from 'react'
import { Activity, Sun, Moon } from 'lucide-react'
import { useUIStore } from '@lab/store/ui'
import { useSlotsStore } from '@lab/store/slots'
import { api } from '@gw/api/client'
import { useT } from '@gw/i18n'
import { ChannelsPage } from './pages/ChannelsPage'
import { SettingsPage } from './pages/SettingsPage'
import { SessionsPage } from './pages/SessionsPage'
import { ChatPage } from './pages/ChatPage'
import { CronPage } from './pages/CronPage'
import { DocsPage } from './pages/DocsPage'

type Page = 'chat' | 'channels' | 'settings' | 'sessions' | 'cron' | 'docs'

export default function App() {
  const t = useT()
  const theme    = useUIStore((s) => s.theme)
  const fontSize = useUIStore((s) => s.fontSize)
  const lang     = useUIStore((s) => s.lang)
  const toggleTheme = useUIStore((s) => s.toggleTheme)
  const setLang     = useUIStore((s) => s.setLang)

  const setVendors        = useSlotsStore((s) => s.setVendors)
  const setToolInfos      = useSlotsStore((s) => s.setToolInfos)
  const setSkillInfos     = useSlotsStore((s) => s.setSkillInfos)
  const setAgentId        = useSlotsStore((s) => s.setAgentId)
  const setCurrentProject = useSlotsStore((s) => s.setCurrentProject)
  const useAgentId        = useSlotsStore((s) => s.agentId)

  const [page, setPage] = useState<Page>('chat')

  // Bootstrap useSlotsStore with metadata from the Lab API (vendors, tools, skills)
  // and sync gateway agent_id + project=web_ui (console chat channel)
  useEffect(() => {
    api.vendors().then(setVendors).catch(console.error)
    api.tools().then(setToolInfos).catch(console.error)
    api.skills().then(setSkillInfos).catch(console.error)
    setCurrentProject('web_ui')
    api.getGatewayConfig()
      .then((cfg) => {
        const aid = (cfg.gateway?.agent_id as string | undefined)?.trim()
        if (aid) setAgentId(aid)
        else if (!useAgentId || useAgentId === 'hxagent') setAgentId('gateway')
      })
      .catch(console.error)
  }, [setVendors, setToolInfos, setSkillInfos, setAgentId, setCurrentProject])

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
  }, [theme])

  useEffect(() => {
    document.documentElement.style.setProperty('--font-scale', String(fontSize ?? 1))
  }, [fontSize])

  const NAV: { id: Page; label: string }[] = [
    { id: 'chat',     label: t('gw.nav.chat') },
    { id: 'channels', label: t('gw.nav.channels') },
    { id: 'settings', label: t('gw.nav.settings') },
    { id: 'sessions', label: t('gw.nav.sessions') },
    { id: 'cron',     label: t('gw.nav.cron') },
    { id: 'docs',     label: t('gw.nav.docs') },
  ]

  return (
    <div className="flex flex-col h-screen" style={{ background: 'var(--bg-base)' }}>
      {/* Header */}
      <header
        className="flex items-center shrink-0 gap-4 px-4"
        style={{ height: 48, background: 'var(--bg-card)', borderBottom: '1px solid var(--border)' }}
      >
        <div className="flex items-center gap-2">
          <Activity
            size={17}
            strokeWidth={2.5}
            style={{ color: 'var(--accent)', filter: 'drop-shadow(0 0 5px var(--accent-ring))' }}
          />
          <span className="font-semibold" style={{ fontSize: 15, color: 'var(--text-1)', letterSpacing: '-0.02em' }}>
            Gateway Console
          </span>
        </div>

        {/* Nav tabs */}
        <nav className="flex items-center gap-1 ml-4">
          {NAV.map(({ id, label }) => (
            <button
              key={id}
              onClick={() => setPage(id)}
              className="px-3 py-1 rounded-lg font-medium transition-colors"
              style={{
                fontSize: 13,
                color: page === id ? 'var(--accent)' : 'var(--text-3)',
                background: page === id ? 'var(--accent-bg)' : 'transparent',
                border: page === id ? '1px solid var(--accent-ring)' : '1px solid transparent',
              }}
            >
              {label}
            </button>
          ))}
        </nav>

        <div className="flex-1" />

        {/* Language + theme */}
        <div className="flex items-center gap-2">
          {(['en', 'zh'] as const).map((l) => (
            <button
              key={l}
              onClick={() => setLang(l)}
              className="px-1.5 py-1 rounded-lg transition-all"
              style={{
                fontSize: 12,
                fontWeight: 500,
                color: lang === l ? 'var(--accent)' : 'var(--text-3)',
                background: lang === l ? 'var(--accent-bg)' : 'transparent',
                fontFamily: l === 'zh' ? 'system-ui, sans-serif' : 'monospace',
              }}
            >
              {l === 'en' ? 'EN' : '中'}
            </button>
          ))}

          <span style={{ color: 'var(--border)', margin: '0 2px' }}>|</span>

          <button
            onClick={toggleTheme}
            className="p-2 rounded-xl transition-all"
            style={{ color: 'var(--text-3)', background: 'transparent' }}
            onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--text-1)'; e.currentTarget.style.background = 'var(--bg-elevated)' }}
            onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-3)'; e.currentTarget.style.background = 'transparent' }}
          >
            {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
          </button>
        </div>
      </header>

      {/* Main */}
      <main className="flex flex-1 min-h-0 overflow-hidden">
        {page === 'chat'     && <ChatPage />}
        {page === 'channels' && <ChannelsPage />}
        {page === 'settings' && <SettingsPage />}
        {page === 'sessions' && <SessionsPage />}
        {page === 'cron'     && <CronPage />}
        {page === 'docs'     && <DocsPage />}
      </main>
    </div>
  )
}
