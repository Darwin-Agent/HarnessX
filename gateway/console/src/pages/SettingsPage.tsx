import { useEffect, useState } from 'react'
import { Settings, FolderOpen, Wrench, BookOpen, Puzzle, Home, Globe, RefreshCw } from 'lucide-react'
import { api } from '@gw/api/client'
import type { HarnessConfig } from '@gw/api/types'
import { DEFAULT_HARNESS_CONFIG } from '@gw/api/types'
import { useSlotsStore } from '@lab/store/slots'
import { useUIStore } from '@lab/store/ui'
import { useT } from '@gw/i18n'
import { ModelPage } from '@lab/components/settings/ModelPage'
import { ToolsPage } from '@lab/components/settings/ToolsPage'
import { SkillsPage } from '@lab/components/settings/SkillsPage'
import { PluginsPage } from '@lab/components/settings/PluginsPage'
import { FileManager } from '@lab/components/settings/FileManager'
import { HarnessReadonlySection } from '../components/HarnessReadonlySection'

type SettingsTab = 'models' | 'harness' | 'env'
type EnvPage = 'gateway' | 'workspace' | 'tools' | 'skills' | 'plugins'

const ENV_NAV: { id: EnvPage; icon: React.ReactNode; labelKey: string }[] = [
  { id: 'gateway',   icon: <Settings size={16} />,   labelKey: 'gw.settings.agent_id' },
  { id: 'workspace', icon: <FolderOpen size={16} />, labelKey: 'settings.page.workspace' },
  { id: 'tools',     icon: <Wrench size={16} />,     labelKey: 'settings.page.tools' },
  { id: 'skills',    icon: <BookOpen size={16} />,   labelKey: 'settings.page.skills' },
  { id: 'plugins',   icon: <Puzzle size={16} />,     labelKey: 'settings.page.plugins' },
]

const ENV_PAGE_INFO: Record<EnvPage, { en: string; zh: string; desc: string }> = {
  gateway:   { en: 'Gateway',   zh: '网关',  desc: 'Configure the gateway agent identity and IM workspace path.' },
  workspace: { en: 'Workspace', zh: '工作区', desc: 'IM session workspaces are stored under im-workspaces/{agent_id}/.' },
  tools:     { en: 'Tools',     zh: '工具',  desc: 'Enable or disable built-in tools and configure MCP servers.' },
  skills:    { en: 'Skills',    zh: '技能',  desc: 'Manage built-in skills that are auto-injected into each step.' },
  plugins:   { en: 'Plugins',   zh: '插件',  desc: 'Discover, enable, and manage installed plugins.' },
}

// ── Gateway workspace page ────────────────────────────────────────────────────

function GwWorkspacePage() {
  const agentId = useSlotsStore((s) => s.agentId)
  const { sandboxType, setSandboxType, sandboxUrl, setSandboxUrl } = useSlotsStore()
  const [homeInfo, setHomeInfo] = useState<{ home: string } | null>(null)
  const [homeLoading, setHomeLoading] = useState(false)
  const [urlInput, setUrlInput] = useState(sandboxUrl)

  function loadHome() {
    setHomeLoading(true)
    api.getHome()
      .then(setHomeInfo)
      .catch(() => setHomeInfo(null))
      .finally(() => setHomeLoading(false))
  }

  useEffect(() => { loadHome() }, [])

  function applyUrl() { setSandboxUrl(urlInput.trim()) }

  const fieldLabel: React.CSSProperties = {
    fontSize: '11.5px',
    color: 'var(--text-3)',
    fontFamily: 'JetBrains Mono, monospace',
    letterSpacing: '0.05em',
    textTransform: 'uppercase',
    display: 'block',
    marginBottom: '7px',
  }

  const homePath = homeInfo?.home ?? null
  const imBase = homePath
    ? `${homePath}/im-workspaces/${agentId}`
    : `~/.harnessx/im-workspaces/${agentId}`
  const imWorkspacePath = homePath ? `${homePath}/im-workspaces/${agentId}` : null

  return (
    <div className="space-y-5">

      {/* ── AGENT_HOME + IM workspace context ─────────────────────────────────── */}
      <div
        className="rounded-2xl overflow-hidden"
        style={{ border: '1px solid var(--border)', background: 'var(--bg-card)' }}
      >
        {/* Header — AGENT_HOME */}
        <div
          className="flex items-center gap-3 px-5 py-3.5"
          style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-elevated)' }}
        >
          <Home size={14} style={{ color: 'var(--accent)', flexShrink: 0 }} />
          <div className="flex-1 min-w-0">
            <div style={fieldLabel}>AGENT_HOME</div>
            <span className="font-mono truncate block" style={{ fontSize: '12.5px', color: 'var(--text-1)' }}>
              {homeLoading ? '…' : (homeInfo?.home ?? '~/.harnessx')}
            </span>
          </div>
          <button
            onClick={loadHome}
            className="p-1.5 rounded-lg transition-colors shrink-0"
            style={{ color: 'var(--text-3)', border: '1px solid var(--border)' }}
            title="Refresh"
          >
            <RefreshCw size={12} className={homeLoading ? 'animate-spin' : ''} />
          </button>
        </div>

        {/* IM workspace path */}
        <div
          className="flex items-center gap-2 px-5 py-2.5"
          style={{ borderBottom: '1px solid var(--border)', background: 'rgba(79,70,229,0.04)' }}
        >
          <span style={{ fontSize: '11px', color: 'var(--accent)', fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.06em', textTransform: 'uppercase', flexShrink: 0 }}>
            IM Workspace →
          </span>
          <span className="font-mono truncate" style={{ fontSize: '12px', color: 'var(--text-2)' }}>
            {imBase}/
          </span>
        </div>

        {/* Info note */}
        <div className="px-5 py-3">
          <p style={{ fontSize: '12px', color: 'var(--text-4)', lineHeight: 1.6 }}>
            Each IM channel gets its own subdirectory under <span className="font-mono" style={{ color: 'var(--text-2)' }}>im-workspaces/{agentId}/</span>. The <span className="font-mono" style={{ color: 'var(--text-2)' }}>web_ui</span> channel is used by the console chat page.
          </p>
        </div>
      </div>

      {/* ── IM workspace file browser ────────────────────────────────────────── */}
      {imWorkspacePath && (
        <div
          className="rounded-2xl overflow-hidden"
          style={{ border: '1px solid var(--border)', background: 'var(--bg-card)' }}
        >
          <div
            className="flex items-center gap-2 px-5 py-3"
            style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-elevated)' }}
          >
            <span style={{ ...fieldLabel, marginBottom: 0 }}>File Browser</span>
            <span className="font-mono flex-1 truncate" style={{ fontSize: '11.5px', color: 'var(--text-4)' }}>
              {imWorkspacePath}/
            </span>
          </div>
          <div className="p-4">
            <FileManager rootPath={imWorkspacePath} />
          </div>
        </div>
      )}

      {/* ── Sandbox type ─────────────────────────────────────────────────────── */}
      <div
        className="rounded-2xl p-5"
        style={{ border: '1px solid var(--border)', background: 'var(--bg-card)' }}
      >
        <label style={fieldLabel}>Sandbox Type</label>
        <div className="flex gap-3">
          {(['local', 'remote'] as const).map((type) => (
            <button
              key={type}
              onClick={() => setSandboxType(type)}
              className="flex-1 py-2.5 rounded-xl font-medium transition-all duration-150"
              style={{
                fontSize: '14px',
                ...(sandboxType === type
                  ? { background: 'var(--accent-bg)', color: 'var(--accent)', border: '1px solid var(--accent-ring)' }
                  : { background: 'var(--bg-elevated)', color: 'var(--text-2)', border: '1px solid var(--border)' }),
              }}
            >
              {type === 'local' ? 'Local' : 'Remote'}
            </button>
          ))}
        </div>
        <p className="mt-3" style={{ fontSize: '12.5px', color: 'var(--text-3)', lineHeight: 1.6 }}>
          {sandboxType === 'local'
            ? 'Claude Code runs tools directly on this machine. Workspace is under AGENT_HOME.'
            : 'Claude Code connects to a remote sandbox API. Tools execute on the remote server.'}
        </p>

        {sandboxType === 'remote' && (
          <div className="mt-4">
            <label style={fieldLabel}>Remote URL</label>
            <div className="flex gap-2">
              <div
                className="flex-1 flex items-center gap-2 rounded-xl px-3.5 py-2.5"
                style={{ border: '1px solid var(--border)', background: 'var(--bg-base)' }}
              >
                <Globe size={15} style={{ color: 'var(--text-4)', flexShrink: 0 }} />
                <input
                  type="text"
                  value={urlInput}
                  onChange={(e) => setUrlInput(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') applyUrl() }}
                  placeholder="https://sandbox.example.com"
                  className="flex-1 min-w-0"
                  style={{ border: 'none', background: 'transparent', outline: 'none', fontSize: '14px', color: 'var(--text-1)' }}
                />
              </div>
              <button
                onClick={applyUrl}
                className="px-4 py-2.5 rounded-xl font-medium transition-colors shrink-0"
                style={{ fontSize: '14px', background: 'var(--accent)', color: '#fff' }}
              >
                Set
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Gateway sub-page ─────────────────────────────────────────────────────────

function GatewayPage() {
  const t = useT()
  const { lang, fontSize, setFontSize, thinkingPostStream, setThinkingPostStream } = useUIStore()
  const agentId = useSlotsStore((s) => s.agentId)
  const setAgentId = useSlotsStore((s) => s.setAgentId)

  const [agentIdDraft, setAgentIdDraft] = useState(agentId)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  useEffect(() => { setAgentIdDraft(agentId) }, [agentId])

  const saveAgentId = () => {
    const val = agentIdDraft.trim()
    if (!val || val === agentId) return
    setSaving(true)
    setSaved(false)
    api.updateGatewayConfig({ agent_id: val })
      .then(() => {
        setAgentId(val)
        setSaved(true)
        setSaving(false)
        setTimeout(() => setSaved(false), 2000)
      })
      .catch(console.error)
  }

  const fieldLabel: React.CSSProperties = {
    fontSize: 11, color: 'var(--text-3)', fontWeight: 600,
    textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 4,
  }

  return (
    <div className="space-y-8 max-w-lg">
      {/* Agent ID */}
      <div>
        <label style={fieldLabel}>{t('gw.settings.agent_id')}</label>
        <p style={{ fontSize: 13, color: 'var(--text-4)', marginBottom: 10 }}>
          {t('gw.settings.agent_id_hint')}
        </p>
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={agentIdDraft}
            onChange={(e) => { setAgentIdDraft(e.target.value); setSaved(false) }}
            onKeyDown={(e) => { if (e.key === 'Enter') saveAgentId() }}
            style={{
              fontSize: 13, padding: '7px 10px', borderRadius: 8,
              border: '1px solid var(--border)', background: 'var(--bg-elevated)',
              color: 'var(--text-1)', outline: 'none', width: 200,
            }}
            onFocus={(e) => { e.currentTarget.style.borderColor = 'var(--accent-ring)' }}
            onBlur={(e) => { e.currentTarget.style.borderColor = 'var(--border)' }}
          />
          <button
            onClick={saveAgentId}
            disabled={saving || !agentIdDraft.trim() || agentIdDraft.trim() === agentId}
            className="px-3 py-1.5 rounded-lg font-medium"
            style={{
              fontSize: 12,
              background: 'var(--accent-bg)', color: 'var(--accent)',
              border: '1px solid var(--accent-ring)',
              opacity: (saving || !agentIdDraft.trim() || agentIdDraft.trim() === agentId) ? 0.5 : 1,
            }}
          >
            {saving ? '…' : t('gw.settings.save')}
          </button>
          {saved && <span style={{ fontSize: 12, color: '#34c759' }}>{t('gw.settings.saved')}</span>}
        </div>
      </div>

      {/* Font size */}
      <div>
        <label style={fieldLabel}>{t('settings.font_size')}</label>
        <div className="flex items-center gap-3 mt-2">
          <input
            type="range" min={0.80} max={1.30} step={0.05}
            value={fontSize ?? 1}
            onChange={(e) => setFontSize(parseFloat(e.target.value))}
            className="range-slider"
            style={{ width: 160 }}
          />
          <span style={{ fontSize: 12, color: 'var(--accent)', fontFamily: 'JetBrains Mono, monospace' }}>
            {Math.round((fontSize ?? 1) * 100)}%
          </span>
        </div>
        <div className="flex gap-1 mt-1" style={{ fontSize: 10, color: 'var(--text-4)', fontFamily: 'JetBrains Mono, monospace', width: 160 }}>
          <span>80%</span><span className="flex-1" /><span>130%</span>
        </div>
      </div>

      {/* Thinking post-stream */}
      <div>
        <label style={fieldLabel}>{t('settings.thinking_post_stream')}</label>
        <div
          className="flex items-center gap-0.5 rounded-lg p-0.5 mt-2"
          style={{ border: '1px solid var(--border)', background: 'var(--bg-elevated)', width: 'fit-content' }}
        >
          {(['collapse', 'keep'] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => setThinkingPostStream(mode)}
              className="px-3 py-1 rounded-md transition-all duration-150"
              style={{
                fontSize: 12,
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
  )
}

// ── Main SettingsPage ────────────────────────────────────────────────────────

export function SettingsPage() {
  const t = useT()
  const { lang } = useUIStore()
  const agentId = useSlotsStore((s) => s.agentId)

  const [tab, setTab] = useState<SettingsTab>('models')
  const [envPage, setEnvPage] = useState<EnvPage>('gateway')

  // ── Harness tab state ──────────────────────────────────────────────────────
  const [harnessConfig, setHarnessConfig] = useState<HarnessConfig>(DEFAULT_HARNESS_CONFIG)
  const [harnessLoaded, setHarnessLoaded] = useState(false)

  useEffect(() => {
    if (tab !== 'harness' || harnessLoaded) return
    api.getGwHarnessConfig(agentId, 'default')
      .then((r) => { setHarnessConfig(r.harness_config); setHarnessLoaded(true) })
      .catch(console.error)
  }, [tab, agentId, harnessLoaded])

  useEffect(() => { setHarnessLoaded(false) }, [agentId])

  const TABS: { id: SettingsTab; label: string }[] = [
    { id: 'models',  label: t('gw.settings.tab.models') },
    { id: 'harness', label: t('gw.settings.tab.harness') },
    { id: 'env',     label: t('gw.settings.tab.env') },
  ]

  const pageInfo = ENV_PAGE_INFO[envPage]

  return (
    <div className="flex flex-col flex-1 min-h-0 overflow-hidden">
      {/* Tab bar */}
      <div
        className="flex items-center px-6 shrink-0"
        style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-card)' }}
      >
        {TABS.map(({ id, label }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className="relative px-4 py-2.5 font-medium transition-colors"
            style={{
              fontSize: 13,
              color: tab === id ? 'var(--text-1)' : 'var(--text-3)',
              background: 'transparent',
              border: 'none',
            }}
          >
            {label}
            {tab === id && (
              <span
                className="absolute bottom-0 left-3 right-3 h-0.5 rounded-t-full"
                style={{ background: 'var(--accent)' }}
              />
            )}
          </button>
        ))}
      </div>

      {/* Tab content */}
      <div className="flex-1 min-h-0 overflow-hidden">

        {/* Models tab */}
        {tab === 'models' && (
          <div className="h-full overflow-y-auto">
            <div className="px-8 py-7 max-w-4xl">
              <ModelPage />
            </div>
          </div>
        )}

        {/* Harness tab */}
        {tab === 'harness' && (
          <div className="h-full overflow-y-auto">
            <div className="px-8 py-7 max-w-lg">
              <HarnessReadonlySection
                harnessConfig={harnessConfig}
                hint={t('gw.settings.harness_hint')}
                onImport={async (cfg) => {
                  await api.saveGwHarnessConfig(agentId, 'default', cfg)
                  setHarnessConfig(cfg)
                }}
              />
            </div>
          </div>
        )}

        {/* Environment tab — sidebar layout identical to Lab UI SettingsSheet */}
        {tab === 'env' && (
          <div className="flex flex-1 min-h-0 h-full">
            {/* Left sidebar */}
            <aside
              className="shrink-0 flex flex-col py-4"
              style={{ width: 220, borderRight: '1px solid var(--border)', background: 'var(--bg-card)' }}
            >
              <nav className="flex-1 px-3 space-y-0.5">
                {ENV_NAV.map(({ id, icon, labelKey }) => {
                  const active = envPage === id
                  return (
                    <button
                      key={id}
                      onClick={() => setEnvPage(id)}
                      className="w-full flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-100 text-left"
                      style={{
                        fontSize: 14,
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
            </aside>

            {/* Right content */}
            <main className="flex-1 min-w-0 overflow-y-auto">
              <div className="px-8 py-7">
                <div className="mb-7">
                  <h1 className="font-bold" style={{ fontSize: 22, color: 'var(--text-1)', letterSpacing: '-0.03em' }}>
                    {lang === 'zh' ? pageInfo.zh : pageInfo.en}
                  </h1>
                  <p className="mt-1.5" style={{ fontSize: 14, color: 'var(--text-3)', lineHeight: 1.6 }}>
                    {pageInfo.desc}
                  </p>
                </div>

                {envPage === 'gateway'   && <GatewayPage />}
                {envPage === 'workspace' && <GwWorkspacePage />}
                {envPage === 'tools'     && <ToolsPage />}
                {envPage === 'skills'    && <SkillsPage />}
                {envPage === 'plugins'   && <PluginsPage />}
              </div>
            </main>
          </div>
        )}
      </div>
    </div>
  )
}
