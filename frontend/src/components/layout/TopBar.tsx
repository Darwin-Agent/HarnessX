import { Activity, Sun, Moon, Settings, Cpu, Clock, BookOpen, User } from 'lucide-react'
import { NavLink } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { HistoryDrawer } from './HistoryDrawer'
import { useLabStore } from '../../store/lab'
import { useRunsStore } from '../../store/runs'
import { useSlotsStore } from '../../store/slots'
import { useUIStore } from '../../store/ui'
import { useDocsStore } from '../../store/docs'
import { useT } from '../../i18n'
import { api } from '../../api/client'
import { DEFAULT_HARNESS_CONFIG } from '../../api/types'
import { resolveBuilderHarnessName, workspaceFromHarnessName } from '../../lib/labWorkspace'
import type { ModelCapability } from '../../api/types'

export function TopBar() {
  const {
    setDimensions, setExamples, replaceCurrentConfig,
    selectedCustomId, selectedExampleKey, customHarnesses, examples, savedWorkspaceConfig,
  } = useLabStore()
  const columns = useRunsStore((s) => s.columns)
  const {
    setVendors,
    setToolInfos,
    setSkillInfos,
    setMcpServers,
    importModelConfig,
    setAgentId,
    setCurrentProject,
    agentId,
    currentProject,
  } = useSlotsStore()
  const { theme, lang, toggleTheme, setLang, setSettingsOpen, setModelOpen } = useUIStore()
  const openDocs = useDocsStore((s) => s.open)
  const t = useT()
  const [historyOpen, setHistoryOpen] = useState(false)
  const [initNotice, setInitNotice] = useState<string | null>(null)

  useEffect(() => {
    api.schema().then((r) => setDimensions(r.dimensions)).catch(console.error)
    api.examples()
      .then((items) => setExamples(items.filter((e) => e.key !== 'strict')))
      .catch(console.error)
    api.vendors().then(setVendors).catch(console.error)
    api.tools().then(setToolInfos).catch(console.error)
    api.skills().then(setSkillInfos).catch(console.error)
    api.mcpServers().then(setMcpServers).catch(console.error)
    api.getHome().then((h) => {
      setAgentId(h.default_agent_id)
      setCurrentProject(h.default_project)
    }).catch(console.error)
  }, [setDimensions, setExamples, setVendors, setToolInfos, setSkillInfos, setMcpServers, setAgentId, setCurrentProject])

  // Keep the "CLI Agent" config synced with the agent-shared harness_config.yaml.
  useEffect(() => {
    if (selectedCustomId !== null || selectedExampleKey !== null) return
    let cancelled = false
    api.getAgentHarnessConfig(
      agentId,
      currentProject,
      agentId === 'hxagent',
    )
      .then((res) => {
        if (!cancelled) {
          replaceCurrentConfig(res.harness_config, {
            agent_id: agentId,
            project: currentProject,
          })
        }
        if (!cancelled && res.used_default) {
          setInitNotice(
            res.persisted_default
              ? `Initialized agent ${res.agent_id} with CLI default harness config`
              : `Using CLI default harness config for agent ${res.agent_id}`,
          )
        }
      })
      .catch((err) => {
        console.error(err)
        if (!cancelled) {
          replaceCurrentConfig({ ...DEFAULT_HARNESS_CONFIG }, {
            agent_id: agentId,
            project: currentProject,
          })
        }
      })
    return () => { cancelled = true }
  }, [agentId, currentProject, selectedCustomId, selectedExampleKey, replaceCurrentConfig])

  useEffect(() => {
    if (!initNotice) return
    const timer = window.setTimeout(() => setInitNotice(null), 6000)
    return () => window.clearTimeout(timer)
  }, [initNotice])

  // Auto-sync ~/.harnessx/model_config.yaml on every page load.
  useEffect(() => {
    api.modelConfig().then((cfg) => {
      if (!cfg || cfg.registry.length === 0) return
      importModelConfig(
        cfg.registry.map((m) => ({
          id:           m.id,
          display_name: m.display_name,
          vendor:       m.vendor,
          model_id:     m.model_id,
          api_key:      m.api_key,
          base_url:     m.base_url,
          extra_headers: m.extra_headers,
          capabilities: m.capabilities as ModelCapability[],
          extended_thinking: m.extended_thinking,
          thinking_budget_tokens: m.thinking_budget_tokens,
          reasoning_effort: m.reasoning_effort,
          reasoning_summary: m.reasoning_summary,
        })),
        cfg.slots.map((s) => ({
          slot_name: s.slot_name,
          model_ids: s.model_ids,
          strategy:  s.strategy as 'primary' | 'fallback' | 'round_robin',
        })),
      )
    }).catch(console.error)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const currentHarnessName = resolveBuilderHarnessName({
    selectedCustomId,
    selectedExampleKey,
    customHarnesses,
    examples,
  })
  const derivedWs = workspaceFromHarnessName(currentHarnessName)
  const preferStoreWorkspace = selectedCustomId !== null || selectedExampleKey !== null || columns.length <= 1
  const activeAgentId = preferStoreWorkspace
    ? savedWorkspaceConfig.agent_id
    : (columns[0]?.workspaceAgentId ?? derivedWs.agentId)
  const activeProject = preferStoreWorkspace
    ? savedWorkspaceConfig.project
    : (columns[0]?.workspaceProject ?? derivedWs.project)

  const iconBtn = {
    base:  { color: 'var(--text-3)', background: 'transparent' } as React.CSSProperties,
    hover: { color: 'var(--text-1)', background: 'var(--bg-elevated)' } as React.CSSProperties,
  }

  return (
    <header
      className="flex items-stretch shrink-0"
      style={{
        height: '48px',
        background: 'var(--bg-card)',
        borderBottom: '1px solid var(--border)',
        paddingLeft: '1rem',
        paddingRight: '0.75rem',
      }}
    >
      {/* Logo */}
      <div
        className="flex items-center gap-2 mr-5 pr-5"
        style={{ borderRight: '1px solid var(--border)' }}
      >
        <Activity
          size={17}
          strokeWidth={2.5}
          style={{ color: 'var(--accent)', filter: 'drop-shadow(0 0 5px var(--accent-ring))' }}
        />
        <span className="font-semibold" style={{ fontSize: '15px', color: 'var(--text-1)', letterSpacing: '-0.02em' }}>
          Harness Lab
        </span>
      </div>

      {/* Navigation */}
      <nav className="flex items-stretch gap-0">
        {[
          { to: '/builder', key: 'nav.builder' },
          { to: '/compare', key: 'nav.compare' },
        ].map(({ to, key }) => (
          <NavLink
            key={to}
            to={to}
            className="relative flex items-center px-4 font-medium transition-colors duration-150"
            style={({ isActive }) => ({
              fontSize: '14px',
              color: isActive ? 'var(--text-1)' : 'var(--text-3)',
            })}
          >
            {({ isActive }) => (
              <>
                {t(key)}
                {isActive && (
                  <span
                    className="absolute bottom-0 left-3 right-3 h-0.5 rounded-t-full"
                    style={{ background: 'var(--accent)', boxShadow: 'var(--accent-glow)' }}
                  />
                )}
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Agent / Project breadcrumb */}
      <div
        className="flex items-center gap-1 ml-3 px-2.5 py-1 rounded-lg cursor-pointer"
        style={{ border: '1px solid var(--border)', background: 'transparent' }}
        onClick={() => setSettingsOpen(true)}
        title="Switch agent / project"
      >
        <User size={11} style={{ color: 'var(--text-4)', flexShrink: 0 }} />
        <span className="font-mono" style={{ fontSize: '11.5px', color: 'var(--text-3)', maxWidth: '120px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {activeAgentId}/{activeProject}
        </span>
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Controls */}
      <div
        className="flex items-center gap-2"
        style={{ borderLeft: '1px solid var(--border)', paddingLeft: '0.75rem', marginLeft: '0.5rem' }}
      >
        {initNotice && (
          <span
            className="px-2.5 py-1 rounded-lg"
            style={{
              fontSize: '12px',
              color: 'var(--accent)',
              background: 'var(--accent-bg)',
              border: '1px solid var(--accent-ring)',
              maxWidth: '340px',
              whiteSpace: 'nowrap',
              overflow: 'hidden',
              textOverflow: 'ellipsis',
            }}
            title={initNotice}
          >
            {initNotice}
          </span>
        )}

        {/* Docs button */}
        <button
          onClick={() => openDocs()}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all duration-150"
          style={{ fontSize: '13px', color: 'var(--text-2)', border: '1px solid var(--border)', background: 'transparent' }}
          onMouseEnter={(e) => {
            e.currentTarget.style.color = 'var(--accent)'
            e.currentTarget.style.background = 'var(--accent-bg)'
            e.currentTarget.style.borderColor = 'var(--accent-ring)'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.color = 'var(--text-2)'
            e.currentTarget.style.background = 'transparent'
            e.currentTarget.style.borderColor = 'var(--border)'
          }}
        >
          <BookOpen size={13} />
          Docs
        </button>

        {/* History button */}
        <button
          onClick={() => setHistoryOpen(true)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all duration-150"
          style={{ fontSize: '13px', color: 'var(--text-2)', border: '1px solid var(--border)', background: 'transparent' }}
          onMouseEnter={(e) => {
            e.currentTarget.style.color = 'var(--accent)'
            e.currentTarget.style.background = 'var(--accent-bg)'
            e.currentTarget.style.borderColor = 'var(--accent-ring)'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.color = 'var(--text-2)'
            e.currentTarget.style.background = 'transparent'
            e.currentTarget.style.borderColor = 'var(--border)'
          }}
          title="History"
        >
          <Clock size={13} />
          {t('topbar.history')}
        </button>

        {/* Model button */}
        <button
          onClick={() => setModelOpen(true)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all duration-150"
          style={{ fontSize: '13px', color: 'var(--text-2)', border: '1px solid var(--border)', background: 'transparent' }}
          onMouseEnter={(e) => {
            e.currentTarget.style.color = 'var(--accent)'
            e.currentTarget.style.background = 'var(--accent-bg)'
            e.currentTarget.style.borderColor = 'var(--accent-ring)'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.color = 'var(--text-2)'
            e.currentTarget.style.background = 'transparent'
            e.currentTarget.style.borderColor = 'var(--border)'
          }}
        >
          <Cpu size={13} />
          {t('topbar.model')}
        </button>

        {/* Environment settings button */}
        <button
          onClick={() => setSettingsOpen(true)}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-all duration-150"
          style={{ fontSize: '13px', color: 'var(--text-2)', border: '1px solid var(--border)', background: 'transparent' }}
          onMouseEnter={(e) => {
            e.currentTarget.style.color = 'var(--accent)'
            e.currentTarget.style.background = 'var(--accent-bg)'
            e.currentTarget.style.borderColor = 'var(--accent-ring)'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.color = 'var(--text-2)'
            e.currentTarget.style.background = 'transparent'
            e.currentTarget.style.borderColor = 'var(--border)'
          }}
        >
          <Settings size={13} />
          {t('settings.env')}
        </button>

        <span style={{ color: 'var(--border)', fontSize: '14px', margin: '0 1px' }}>|</span>

        {/* Language toggle */}
        {(['en', 'zh'] as const).map((l) => (
          <button
            key={l}
            onClick={() => setLang(l)}
            className="px-1.5 py-1 rounded-lg transition-all duration-150"
            style={{
              fontSize: '12px',
              fontWeight: 500,
              color: lang === l ? 'var(--accent)' : 'var(--text-3)',
              background: lang === l ? 'var(--accent-bg)' : 'transparent',
              fontFamily: l === 'zh' ? 'Outfit, system-ui, sans-serif' : 'JetBrains Mono, monospace',
            }}
          >
            {l === 'en' ? 'EN' : '中'}
          </button>
        ))}

        <span style={{ color: 'var(--border)', fontSize: '14px', margin: '0 1px' }}>|</span>

        {/* Theme toggle */}
        <button
          onClick={toggleTheme}
          title={theme === 'dark' ? t('settings.light') : t('settings.dark')}
          className="p-2 rounded-xl transition-all duration-150"
          style={iconBtn.base}
          onMouseEnter={(e) => Object.assign(e.currentTarget.style, iconBtn.hover)}
          onMouseLeave={(e) => Object.assign(e.currentTarget.style, iconBtn.base)}
        >
          {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
        </button>
      </div>

      <HistoryDrawer isOpen={historyOpen} onClose={() => setHistoryOpen(false)} />
    </header>
  )
}
