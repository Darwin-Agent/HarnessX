import { useEffect, useState } from 'react'
import { ArrowLeft, FolderOpen, Wrench, BookOpen, Puzzle, Sun, Moon, Type } from 'lucide-react'
import { useUIStore } from '../../store/ui'
import { useT } from '../../i18n'
import { WorkspacePage } from '../settings/WorkspacePage'
import { ToolsPage }     from '../settings/ToolsPage'
import { SkillsPage }    from '../settings/SkillsPage'
import { PluginsPage }   from '../settings/PluginsPage'

type SettingsPage = 'workspace' | 'tools' | 'skills' | 'plugins'

const NAV_ITEMS: { id: SettingsPage; icon: React.ReactNode; labelKey: string }[] = [
  { id: 'workspace', icon: <FolderOpen size={16} />, labelKey: 'settings.page.workspace' },
  { id: 'tools',     icon: <Wrench size={16} />,     labelKey: 'settings.page.tools' },
  { id: 'skills',    icon: <BookOpen size={16} />,   labelKey: 'settings.page.skills' },
  { id: 'plugins',   icon: <Puzzle size={16} />,     labelKey: 'settings.page.plugins' },
]

const PAGE_TITLES: Record<SettingsPage, { en: string; zh: string; desc: string }> = {
  workspace: { en: 'Workspace', zh: '工作区', desc: 'Set up your local workspace directory and browse files.' },
  tools:     { en: 'Tools',     zh: '工具',  desc: 'Enable or disable built-in tools and configure MCP servers.' },
  skills:    { en: 'Skills',    zh: '技能',  desc: 'Manage built-in skills that are auto-injected into each step.' },
  plugins:   { en: 'Plugins',   zh: '插件',  desc: 'Discover, enable, and manage installed plugins.' },
}

export function SettingsSheet() {
  const t = useT()
  const {
    settingsOpen,
    setSettingsOpen,
    theme,
    lang,
    toggleTheme,
    setLang,
    fontSize,
    setFontSize,
    thinkingPostStream,
    setThinkingPostStream,
  } = useUIStore()
  const [activePage, setActivePage] = useState<SettingsPage>('workspace')

  useEffect(() => {
    if (!settingsOpen) return
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') setSettingsOpen(false) }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [settingsOpen, setSettingsOpen])

  if (!settingsOpen) return null

  const pageInfo = PAGE_TITLES[activePage]

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col"
      style={{
        background: 'var(--bg-base)',
        animation: 'slideInUp 0.20s cubic-bezier(0.16, 1, 0.3, 1)',
      }}
    >
      {/* ── Top header ── */}
      <div
        className="flex items-center gap-3 px-5 shrink-0"
        style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-card)', height: '52px' }}
      >
        {/* Back button — prominent CTA */}
        <button
          onClick={() => setSettingsOpen(false)}
          className="flex items-center gap-2 px-3 py-1.5 rounded-xl font-medium transition-all duration-150"
          style={{ fontSize: '14px', color: 'var(--text-2)', border: '1px solid var(--border)', background: 'var(--bg-elevated)' }}
          onMouseEnter={(e) => {
            e.currentTarget.style.color = 'var(--accent)'
            e.currentTarget.style.borderColor = 'var(--accent-ring)'
            e.currentTarget.style.background = 'var(--accent-bg)'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.color = 'var(--text-2)'
            e.currentTarget.style.borderColor = 'var(--border)'
            e.currentTarget.style.background = 'var(--bg-elevated)'
          }}
        >
          <ArrowLeft size={14} />
          {t('settings.back')}
        </button>

        <span style={{ color: 'var(--border)', fontSize: '16px' }}>/</span>

        <span className="font-semibold" style={{ fontSize: '15px', color: 'var(--text-1)', letterSpacing: '-0.02em' }}>
          {t('settings.env')}
        </span>
        <span style={{ color: 'var(--border)', fontSize: '14px' }}>/</span>
        <span style={{ fontSize: '14px', color: 'var(--accent)', fontWeight: 500 }}>
          {lang === 'zh' ? pageInfo.zh : pageInfo.en}
        </span>

        <div className="flex-1" />

        {/* Language toggle */}
        <div className="flex items-center gap-0.5" style={{ borderRadius: '8px', border: '1px solid var(--border)', padding: '2px', background: 'var(--bg-elevated)' }}>
          {(['en', 'zh'] as const).map((l) => (
            <button
              key={l}
              onClick={() => setLang(l)}
              className="px-2 py-1 rounded-md transition-all duration-150 font-medium"
              style={{
                fontSize: '12px',
                color: lang === l ? 'var(--accent)' : 'var(--text-3)',
                background: lang === l ? 'var(--bg-card)' : 'transparent',
                fontFamily: l === 'zh' ? 'system-ui, sans-serif' : 'JetBrains Mono, monospace',
              }}
            >
              {l === 'en' ? 'EN' : '中'}
            </button>
          ))}
        </div>

        {/* Theme toggle */}
        <button
          onClick={toggleTheme}
          className="p-2 rounded-xl transition-all duration-150"
          style={{ color: 'var(--text-3)' }}
          onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--bg-elevated)'; e.currentTarget.style.color = 'var(--text-1)' }}
          onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-3)' }}
        >
          {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
        </button>
      </div>

      {/* ── Body: sidebar + content ── */}
      <div className="flex flex-1 min-h-0">

        {/* Left sidebar */}
        <aside
          className="shrink-0 flex flex-col py-4"
          style={{ width: '220px', borderRight: '1px solid var(--border)', background: 'var(--bg-card)' }}
        >
          <nav className="flex-1 px-3 space-y-0.5">
            {NAV_ITEMS.map(({ id, icon, labelKey }) => {
              const active = activePage === id
              return (
                <button
                  key={id}
                  onClick={() => setActivePage(id)}
                  className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-100 text-left"
                  style={{
                    fontSize: '14px',
                    fontWeight: active ? 600 : 400,
                    background: active ? 'var(--accent-bg)' : 'transparent',
                    color: active ? 'var(--accent)' : 'var(--text-2)',
                    borderLeft: `3px solid ${active ? 'var(--accent)' : 'transparent'}`,
                  }}
                  onMouseEnter={(e) => { if (!active) e.currentTarget.style.background = 'var(--bg-elevated)' }}
                  onMouseLeave={(e) => { if (!active) e.currentTarget.style.background = 'transparent' }}
                >
                  <span style={{ opacity: active ? 1 : 0.65 }}>{icon}</span>
                  {t(labelKey)}
                </button>
              )
            })}
          </nav>

          {/* Font size scaler — pinned to sidebar bottom */}
          <div className="px-4 pt-4 mt-2" style={{ borderTop: '1px solid var(--border)' }}>
            <div className="flex items-center gap-1.5 mb-2">
              <Type size={11} style={{ color: 'var(--text-3)' }} />
              <span style={{ fontSize: '11px', color: 'var(--text-3)', fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.05em', textTransform: 'uppercase' }}>
                {t('settings.font_size')}
              </span>
              <span style={{ fontSize: '11px', color: 'var(--accent)', fontFamily: 'JetBrains Mono, monospace', marginLeft: 'auto' }}>
                {Math.round((fontSize ?? 1) * 100)}%
              </span>
            </div>
            <input
              type="range"
              min={0.80}
              max={1.30}
              step={0.05}
              value={fontSize ?? 1}
              onChange={(e) => setFontSize(parseFloat(e.target.value))}
              className="range-slider w-full"
            />
            <div className="flex justify-between mt-1" style={{ fontSize: '10px', color: 'var(--text-4)', fontFamily: 'JetBrains Mono, monospace' }}>
              <span>80%</span><span>130%</span>
            </div>

            <div className="mt-4">
              <div className="mb-1.5" style={{ fontSize: '11px', color: 'var(--text-3)', fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.05em', textTransform: 'uppercase' }}>
                {t('settings.thinking_post_stream')}
              </div>
              <div
                className="flex items-center gap-0.5 rounded-lg p-0.5"
                style={{ border: '1px solid var(--border)', background: 'var(--bg-elevated)' }}
              >
                {(['collapse', 'keep'] as const).map((mode) => (
                  <button
                    key={mode}
                    onClick={() => setThinkingPostStream(mode)}
                    className="px-2 py-1 rounded-md transition-all duration-150"
                    style={{
                      fontSize: '11px',
                      color: thinkingPostStream === mode ? 'var(--accent)' : 'var(--text-3)',
                      background: thinkingPostStream === mode ? 'var(--bg-card)' : 'transparent',
                      fontFamily: 'JetBrains Mono, monospace',
                    }}
                  >
                    {t(`settings.thinking.${mode}`)}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </aside>

        {/* Right content area — full width, no max-w constraint */}
        <main className="flex-1 min-w-0 overflow-y-auto">
          <div className="px-8 py-7">
            {/* Page header */}
            <div className="mb-7">
              <h1 className="font-bold" style={{ fontSize: '22px', color: 'var(--text-1)', letterSpacing: '-0.03em' }}>
                {lang === 'zh' ? pageInfo.zh : pageInfo.en}
              </h1>
              <p className="mt-1.5" style={{ fontSize: '14px', color: 'var(--text-3)', lineHeight: 1.6 }}>
                {pageInfo.desc}
              </p>
            </div>

            {/* Page content */}
            {activePage === 'workspace' && <WorkspacePage />}
            {activePage === 'tools'     && <ToolsPage />}
            {activePage === 'skills'    && <SkillsPage />}
            {activePage === 'plugins'   && <PluginsPage />}
          </div>
        </main>
      </div>
    </div>
  )
}
