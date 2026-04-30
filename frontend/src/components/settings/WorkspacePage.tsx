import { useState, useEffect } from 'react'
import { Globe, Home, User, Layers, RefreshCw, ChevronRight } from 'lucide-react'
import { useSlotsStore } from '../../store/slots'
import { useT } from '../../i18n'
import { FileManager } from './FileManager'
import { DocLink } from '../docs/DocLink'
import { api } from '../../api/client'
import type { HomeInfo, AgentEntry } from '../../api/types'

// ── Inline agent/project editor ───────────────────────────────────────────────

interface NameInputProps {
  value: string
  onApply: (v: string) => void
  placeholder: string
}

function NameInput({ value, onApply, placeholder }: NameInputProps) {
  const [draft, setDraft] = useState(value)
  const dirty = draft.trim() !== value

  useEffect(() => { setDraft(value) }, [value])

  return (
    <div className="flex gap-1.5 items-center">
      <input
        type="text"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && draft.trim()) onApply(draft.trim())
          if (e.key === 'Escape') setDraft(value)
        }}
        placeholder={placeholder}
        className="rounded-lg px-2.5 py-1"
        style={{ fontSize: '13px', width: '120px', fontFamily: 'JetBrains Mono, monospace' }}
      />
      {dirty && (
        <button
          onClick={() => { if (draft.trim()) onApply(draft.trim()) }}
          className="px-2 py-1 rounded-lg text-xs font-medium"
          style={{ background: 'var(--accent)', color: '#fff' }}
        >
          Set
        </button>
      )}
    </div>
  )
}

// ── Agent entry row ───────────────────────────────────────────────────────────

interface AgentRowProps {
  entry:          AgentEntry
  activeAgentId:  string
  activeProject:  string
  onActivate:     (agentId: string, project: string) => void
}

function AgentRow({ entry, activeAgentId, activeProject, onActivate }: AgentRowProps) {
  const [open, setOpen] = useState(entry.id === activeAgentId)
  const isActiveAgent = entry.id === activeAgentId

  return (
    <div>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 w-full px-3 py-2 rounded-xl transition-colors text-left"
        style={{
          background: isActiveAgent ? 'var(--accent-bg)' : 'transparent',
          border: `1px solid ${isActiveAgent ? 'var(--accent-ring)' : 'transparent'}`,
        }}
      >
        <User size={13} style={{ color: isActiveAgent ? 'var(--accent)' : 'var(--text-3)', flexShrink: 0 }} />
        <span className="font-mono flex-1" style={{ fontSize: '13px', color: isActiveAgent ? 'var(--accent)' : 'var(--text-1)', fontWeight: isActiveAgent ? 600 : 400 }}>
          {entry.id}
        </span>
        <span style={{ fontSize: '11px', color: 'var(--text-4)', fontFamily: 'JetBrains Mono, monospace' }}>
          {entry.projects.length} project{entry.projects.length !== 1 ? 's' : ''}
        </span>
        <ChevronRight size={12} style={{ color: 'var(--text-4)', transform: open ? 'rotate(90deg)' : 'none', transition: 'transform 0.15s' }} />
      </button>

      {open && (
        <div className="pl-6 pt-1 pb-1 space-y-0.5">
          {entry.projects.map((p) => {
            const isActive = isActiveAgent && p === activeProject
            return (
              <button
                key={p}
                onClick={() => onActivate(entry.id, p)}
                className="flex items-center gap-2 w-full px-3 py-1.5 rounded-xl transition-colors text-left"
                style={{
                  background: isActive ? 'var(--accent-bg)' : 'transparent',
                  border: `1px solid ${isActive ? 'var(--accent-ring)' : 'transparent'}`,
                }}
                onMouseEnter={(e) => { if (!isActive) e.currentTarget.style.background = 'var(--bg-elevated)' }}
                onMouseLeave={(e) => { if (!isActive) e.currentTarget.style.background = 'transparent' }}
              >
                <Layers size={12} style={{ color: isActive ? 'var(--accent)' : 'var(--text-4)', flexShrink: 0 }} />
                <span className="font-mono flex-1" style={{ fontSize: '12.5px', color: isActive ? 'var(--accent)' : 'var(--text-2)', fontWeight: isActive ? 600 : 400 }}>
                  {p}
                </span>
                {isActive && (
                  <span
                    className="px-1.5 py-0.5 rounded-md"
                    style={{ fontSize: '10px', background: 'var(--accent)', color: '#fff', fontFamily: 'JetBrains Mono, monospace' }}
                  >
                    active
                  </span>
                )}
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────

export function WorkspacePage() {
  const t = useT()
  const {
    sandboxType, setSandboxType,
    sandboxUrl, setSandboxUrl,
    agentId, setAgentId,
    currentProject, setCurrentProject,
  } = useSlotsStore()

  const [homeInfo, setHomeInfo]   = useState<HomeInfo | null>(null)
  const [homeLoading, setHomeLoading] = useState(false)
  const [urlInput, setUrlInput]   = useState(sandboxUrl)

  function loadHome() {
    setHomeLoading(true)
    api.getHome()
      .then(setHomeInfo)
      .catch(() => setHomeInfo(null))
      .finally(() => setHomeLoading(false))
  }

  useEffect(() => { loadHome() }, [])

  function handleActivate(newAgent: string, newProject: string) {
    setAgentId(newAgent)
    setCurrentProject(newProject)
  }

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

  // The resolved home path to use for the file tree root
  const homePath = homeInfo?.home ?? null
  // Derived workspace path
  const derivedWorkspace = homeInfo
    ? (homeInfo.agents_tree.find((a) => a.id === agentId)?.project_paths[currentProject]
       ?? `${homeInfo.home}/workspaces/${agentId}/${currentProject}`)
    : `~/.harnessx/workspaces/${agentId}/${currentProject}`

  return (
    <div className="space-y-5">

      {/* ── AGENT_HOME context ───────────────────────────────────────────────── */}
      <div
        className="rounded-2xl overflow-hidden"
        style={{ border: '1px solid var(--border)', background: 'var(--bg-card)' }}
      >
        {/* Header */}
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
          <div className="flex items-center gap-2 shrink-0">
            <DocLink path="feats/workspace" label="Docs" />
            <button
              onClick={loadHome}
              className="p-1.5 rounded-lg transition-colors"
              style={{ color: 'var(--text-3)', border: '1px solid var(--border)' }}
              title="Refresh"
            >
              <RefreshCw size={12} className={homeLoading ? 'animate-spin' : ''} />
            </button>
          </div>
        </div>

        {/* Agent / Project context row */}
        <div
          className="flex items-center gap-4 px-5 py-3"
          style={{ borderBottom: '1px solid var(--border)' }}
        >
          {/* Agent */}
          <div className="flex items-center gap-2">
            <User size={12} style={{ color: 'var(--text-4)' }} />
            <span style={{ fontSize: '11px', color: 'var(--text-4)', fontFamily: 'JetBrains Mono, monospace', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Agent</span>
            <NameInput
              value={agentId}
              onApply={(v) => { setAgentId(v); loadHome() }}
              placeholder="default"
            />
          </div>

          <span style={{ color: 'var(--border)' }}>/</span>

          {/* Project */}
          <div className="flex items-center gap-2">
            <Layers size={12} style={{ color: 'var(--text-4)' }} />
            <span style={{ fontSize: '11px', color: 'var(--text-4)', fontFamily: 'JetBrains Mono, monospace', textTransform: 'uppercase', letterSpacing: '0.06em' }}>Project</span>
            <NameInput
              value={currentProject}
              onApply={(v) => { setCurrentProject(v); loadHome() }}
              placeholder="default"
            />
          </div>
        </div>

        {/* Active workspace path */}
        <div
          className="flex items-center gap-2 px-5 py-2.5"
          style={{ borderBottom: '1px solid var(--border)', background: 'rgba(79,70,229,0.04)' }}
        >
          <span style={{ fontSize: '11px', color: 'var(--accent)', fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.06em', textTransform: 'uppercase', flexShrink: 0 }}>
            Workspace →
          </span>
          <span className="font-mono truncate" style={{ fontSize: '12px', color: 'var(--text-2)' }}>
            {derivedWorkspace}
          </span>
        </div>

        {/* Known agents tree (from API) */}
        {homeInfo && homeInfo.agents_tree.length > 0 && (
          <div className="px-4 py-3 space-y-1">
            <div style={{ fontSize: '11px', color: 'var(--text-4)', fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: '6px', paddingLeft: '4px' }}>
              Known agents
            </div>
            {homeInfo.agents_tree.map((entry) => (
              <AgentRow
                key={entry.id}
                entry={entry}
                activeAgentId={agentId}
                activeProject={currentProject}
                onActivate={handleActivate}
              />
            ))}
          </div>
        )}
      </div>

      {/* ── AGENT_HOME file tree ─────────────────────────────────────────────── */}
      {homePath && (
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
              {homePath}
            </span>
          </div>
          <div className="p-4">
            <FileManager rootPath={homePath} />
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

        {/* Remote URL */}
        {sandboxType === 'remote' && (
          <div className="mt-4">
            <label style={fieldLabel}>{t('workspace.remote_url')}</label>
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
