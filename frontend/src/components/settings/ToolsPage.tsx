import { useState } from 'react'
import { ChevronDown, ChevronRight, Plus, Trash2, Loader, Terminal, Globe, Cpu, Edit2 } from 'lucide-react'
import { useSlotsStore } from '../../store/slots'
import { api } from '../../api/client'
import type { McpServerConfig, McpToolInfo } from '../../api/types'
import { useT } from '../../i18n'
import { DocLink } from '../docs/DocLink'

const GROUP_ICON: Record<string, React.ReactNode> = {
  filesystem: <Terminal size={13} />,
  web:        <Globe size={13} />,
}

// ── MCP Server card ───────────────────────────────────────────────────────────

interface McpCardProps {
  server: McpServerConfig
  onUpdate: (patch: Partial<McpServerConfig>) => void
  onDelete: () => void
}

function McpCard({ server, onUpdate, onDelete }: McpCardProps) {
  const t = useT()
  const [expanded, setExpanded] = useState(false)
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(server)
  const [tools, setTools] = useState<McpToolInfo[] | null>(null)
  const [loadingTools, setLoadingTools] = useState(false)
  const [toolError, setToolError] = useState<string | null>(null)

  async function handlePreview() {
    if (tools) { setExpanded((o) => !o); return }
    setExpanded(true)
    setLoadingTools(true)
    setToolError(null)
    try {
      const result = await api.mcpServerTools(server.id)
      setTools(result)
    } catch (e) {
      setToolError(String(e))
    }
    setLoadingTools(false)
  }

  function handleSaveEdit() {
    onUpdate(draft)
    setEditing(false)
  }

  const fieldLabel: React.CSSProperties = {
    fontSize: '11.5px', color: 'var(--text-3)', fontFamily: 'JetBrains Mono, monospace',
    letterSpacing: '0.05em', textTransform: 'uppercase', display: 'block', marginBottom: '5px',
  }

  return (
    <div
      className="rounded-2xl overflow-hidden"
      style={{ border: '1px solid var(--border)', background: 'var(--bg-card)' }}
    >
      <div className="flex items-center gap-3 px-4 py-3">
        <Cpu size={14} style={{ color: 'var(--accent)', flexShrink: 0 }} />
        <span className="flex-1 font-medium truncate" style={{ fontSize: '14px', color: 'var(--text-1)' }}>
          {server.name}
        </span>
        <span
          className="px-2 py-0.5 rounded-md font-mono"
          style={{ fontSize: '11px', background: 'var(--bg-elevated)', color: 'var(--text-3)', border: '1px solid var(--border)' }}
        >
          {server.transport}
        </span>

        {/* Enabled toggle */}
        <button
          onClick={() => onUpdate({ enabled: !server.enabled })}
          className="px-2.5 py-1 rounded-lg font-medium transition-colors"
          style={{
            fontSize: '12px',
            ...(server.enabled
              ? { background: 'rgba(16,185,129,0.08)', color: '#10b981', border: '1px solid rgba(16,185,129,0.2)' }
              : { background: 'var(--bg-elevated)', color: 'var(--text-3)', border: '1px solid var(--border)' }),
          }}
        >
          {server.enabled ? 'ON' : 'OFF'}
        </button>

        <button
          onClick={() => setEditing((o) => !o)}
          className="p-1.5 rounded-lg transition-colors"
          style={{ color: 'var(--text-3)' }}
          title={t('mcp.edit')}
          onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--text-1)')}
          onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-3)')}
        >
          <Edit2 size={13} />
        </button>

        <button
          onClick={onDelete}
          className="p-1.5 rounded-lg transition-colors"
          style={{ color: 'var(--text-3)' }}
          title={t('mcp.delete')}
          onMouseEnter={(e) => (e.currentTarget.style.color = '#ef4444')}
          onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-3)')}
        >
          <Trash2 size={13} />
        </button>

        <button
          onClick={handlePreview}
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg transition-colors"
          style={{ fontSize: '12px', color: 'var(--accent)', background: 'var(--accent-bg)', border: '1px solid var(--accent-ring)' }}
        >
          {loadingTools ? <Loader size={11} className="animate-spin" /> : (expanded ? <ChevronDown size={11} /> : <ChevronRight size={11} />)}
          {t('mcp.preview')}
        </button>
      </div>

      {/* Edit form */}
      {editing && (
        <div className="px-4 pb-4 pt-1 space-y-3" style={{ borderTop: '1px solid var(--border)' }}>
          <div className="grid gap-3" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <div>
              <label style={fieldLabel}>Name</label>
              <input type="text" value={draft.name} onChange={(e) => setDraft({ ...draft, name: e.target.value })} className="w-full rounded-xl px-3 py-2" style={{ fontSize: '13px' }} />
            </div>
            <div>
              <label style={fieldLabel}>{t('mcp.transport')}</label>
              <select value={draft.transport} onChange={(e) => setDraft({ ...draft, transport: e.target.value as 'stdio' | 'http' })} className="w-full rounded-xl px-3 py-2" style={{ fontSize: '13px' }}>
                <option value="stdio">stdio</option>
                <option value="http">http</option>
              </select>
            </div>
          </div>
          {draft.transport === 'stdio' && (
            <div>
              <label style={fieldLabel}>{t('mcp.command')}</label>
              <input type="text" value={draft.command} onChange={(e) => setDraft({ ...draft, command: e.target.value })} className="w-full rounded-xl px-3 py-2 font-mono" style={{ fontSize: '13px' }} placeholder="uvx mcp-server-sqlite --db ./data.db" />
            </div>
          )}
          {draft.transport === 'http' && (
            <div>
              <label style={fieldLabel}>{t('mcp.url')}</label>
              <input type="text" value={draft.url} onChange={(e) => setDraft({ ...draft, url: e.target.value })} className="w-full rounded-xl px-3 py-2" style={{ fontSize: '13px' }} placeholder="https://..." />
            </div>
          )}
          <div className="flex gap-2 justify-end">
            <button onClick={() => setEditing(false)} className="px-3 py-1.5 rounded-lg" style={{ fontSize: '13px', color: 'var(--text-3)', border: '1px solid var(--border)' }}>
              {t('fs.cancel')}
            </button>
            <button onClick={handleSaveEdit} className="px-3 py-1.5 rounded-lg font-medium" style={{ fontSize: '13px', background: 'var(--accent)', color: '#fff' }}>
              Save
            </button>
          </div>
        </div>
      )}

      {/* Tools preview */}
      {expanded && (
        <div className="px-4 pb-4 pt-1 space-y-2" style={{ borderTop: '1px solid var(--border)' }}>
          {loadingTools && (
            <div className="flex items-center gap-2 py-2" style={{ color: 'var(--text-4)', fontSize: '13px' }}>
              <Loader size={14} className="animate-spin" /> {t('mcp.connecting')}
            </div>
          )}
          {toolError && (
            <p style={{ fontSize: '13px', color: '#ef4444' }}>{toolError}</p>
          )}
          {tools && tools.length === 0 && (
            <p style={{ fontSize: '13px', color: 'var(--text-4)' }}>{t('mcp.tools_empty')}</p>
          )}
          {tools && tools.map((tool) => (
            <div key={tool.name} className="flex items-start gap-2 rounded-xl px-3 py-2.5" style={{ background: 'var(--bg-elevated)' }}>
              <span className="font-mono font-medium shrink-0 mt-px" style={{ fontSize: '13px', color: 'var(--accent)' }}>{tool.name}</span>
              <span style={{ fontSize: '12px', color: 'var(--text-3)', lineHeight: 1.5 }}>{tool.description}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── Add MCP Server form ───────────────────────────────────────────────────────

interface AddMcpFormProps {
  onAdd: (cfg: Omit<McpServerConfig, 'id'>) => void
  onCancel: () => void
}

function AddMcpForm({ onAdd, onCancel }: AddMcpFormProps) {
  const t = useT()
  const [name, setName] = useState('')
  const [transport, setTransport] = useState<'stdio' | 'http'>('stdio')
  const [command, setCommand] = useState('')
  const [url, setUrl] = useState('')

  function handleAdd() {
    if (!name.trim()) return
    onAdd({ name: name.trim(), transport, command: command.trim(), url: url.trim(), env: {}, enabled: true })
  }

  const fieldLabel: React.CSSProperties = {
    fontSize: '11.5px', color: 'var(--text-3)', fontFamily: 'JetBrains Mono, monospace',
    letterSpacing: '0.05em', textTransform: 'uppercase', display: 'block', marginBottom: '5px',
  }

  return (
    <div className="rounded-2xl p-4 space-y-3" style={{ border: '1px solid var(--accent-ring)', background: 'var(--accent-bg)' }}>
      <div className="grid gap-3" style={{ gridTemplateColumns: '1fr 1fr' }}>
        <div>
          <label style={fieldLabel}>Name</label>
          <input autoFocus type="text" value={name} onChange={(e) => setName(e.target.value)} className="w-full rounded-xl px-3 py-2" style={{ fontSize: '13px' }} placeholder="my-mcp-server" />
        </div>
        <div>
          <label style={fieldLabel}>{t('mcp.transport')}</label>
          <select value={transport} onChange={(e) => setTransport(e.target.value as 'stdio' | 'http')} className="w-full rounded-xl px-3 py-2" style={{ fontSize: '13px' }}>
            <option value="stdio">stdio</option>
            <option value="http">http</option>
          </select>
        </div>
      </div>

      {transport === 'stdio' && (
        <div>
          <label style={fieldLabel}>{t('mcp.command')}</label>
          <input
            type="text"
            value={command}
            onChange={(e) => setCommand(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') handleAdd() }}
            className="w-full rounded-xl px-3 py-2 font-mono"
            style={{ fontSize: '13px' }}
            placeholder="uvx mcp-server-sqlite --db ./data.db"
          />
        </div>
      )}
      {transport === 'http' && (
        <div>
          <label style={fieldLabel}>{t('mcp.url')}</label>
          <input
            type="text"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') handleAdd() }}
            className="w-full rounded-xl px-3 py-2"
            style={{ fontSize: '13px' }}
            placeholder="https://..."
          />
        </div>
      )}

      <div className="flex gap-2 justify-end">
        <button onClick={onCancel} className="px-3.5 py-1.5 rounded-lg" style={{ fontSize: '13px', color: 'var(--text-3)', border: '1px solid var(--border)' }}>
          {t('fs.cancel')}
        </button>
        <button onClick={handleAdd} disabled={!name.trim()} className="px-3.5 py-1.5 rounded-lg font-medium disabled:opacity-40" style={{ fontSize: '13px', background: 'var(--accent)', color: '#fff' }}>
          {t('mcp.add')}
        </button>
      </div>
    </div>
  )
}

// ── Page ─────────────────────────────────────────────────────────────────────

export function ToolsPage() {
  const t = useT()
  const { toolInfos, enabledTools, setEnabledTools, mcpServers, addMcpServer, removeMcpServer, updateMcpServer } = useSlotsStore()
  const [showAddMcp, setShowAddMcp] = useState(false)

  function isToolEnabled(name: string) {
    return enabledTools === null || enabledTools.includes(name)
  }

  function toggleTool(name: string) {
    if (enabledTools === null) {
      setEnabledTools(toolInfos.map((t) => t.name).filter((n) => n !== name))
    } else {
      const next = isToolEnabled(name)
        ? enabledTools.filter((n) => n !== name)
        : [...enabledTools, name]
      setEnabledTools(next.length === toolInfos.length ? null : next)
    }
  }

  async function handleAddMcp(cfg: Omit<McpServerConfig, 'id'>) {
    try {
      const saved = await api.mcpAddServer(cfg)
      addMcpServer(saved)
      setShowAddMcp(false)
    } catch (e) {
      alert(String(e))
    }
  }

  async function handleDeleteMcp(id: string) {
    try {
      await api.mcpDeleteServer(id)
      removeMcpServer(id)
    } catch { /* ignore */ }
  }

  async function handleUpdateMcp(id: string, patch: Partial<McpServerConfig>) {
    try {
      const updated = await api.mcpUpdateServer(id, patch)
      updateMcpServer(id, updated)
    } catch { /* ignore */ }
  }

  const groups = toolInfos.reduce<Record<string, typeof toolInfos>>((acc, t) => {
    const g = t.group ?? 'other'
    ;(acc[g] ??= []).push(t)
    return acc
  }, {})

  return (
    <div className="space-y-6">
      {/* Local tools */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-1)' }}>Local Tools</h3>
          <div className="flex gap-2">
            <button onClick={() => setEnabledTools(null)} style={{ fontSize: '12px', color: 'var(--accent)' }}>{t('tools.enable_all')}</button>
            <span style={{ color: 'var(--border)' }}>·</span>
            <button onClick={() => setEnabledTools([])} style={{ fontSize: '12px', color: 'var(--text-3)' }}>{t('tools.disable_all')}</button>
          </div>
        </div>

        <div className="space-y-3">
          {Object.entries(groups).map(([group, tools]) => (
            <div key={group}>
              <div className="flex items-center gap-2 mb-2">
                <span style={{ color: 'var(--text-3)' }}>{GROUP_ICON[group]}</span>
                <span className="label-mono">{group}</span>
              </div>
              <div className="grid gap-2" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))' }}>
                {tools.map((tool) => {
                  const on = isToolEnabled(tool.name)
                  return (
                    <div
                      key={tool.name}
                      className="flex items-start gap-3 rounded-xl p-3.5 cursor-pointer transition-all duration-150"
                      style={{
                        border: on ? '1px solid var(--accent-ring)' : '1px solid var(--border)',
                        background: on ? 'var(--accent-bg)' : 'var(--bg-elevated)',
                      }}
                      onClick={() => toggleTool(tool.name)}
                    >
                      <input
                        type="checkbox"
                        checked={on}
                        onChange={() => toggleTool(tool.name)}
                        className="mt-0.5 shrink-0"
                        onClick={(e) => e.stopPropagation()}
                      />
                      <div className="min-w-0">
                        <div className="font-medium font-mono" style={{ fontSize: '13px', color: on ? 'var(--accent)' : 'var(--text-1)' }}>{tool.name}</div>
                        <div className="mt-0.5 line-clamp-2" style={{ fontSize: '12px', color: 'var(--text-3)', lineHeight: 1.4 }}>{tool.description}</div>
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* MCP Servers */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 style={{ fontSize: '16px', fontWeight: 600, color: 'var(--text-1)' }}>{t('mcp.section')}</h3>
          <DocLink path="feats/mcp" label="Docs" />
          {!showAddMcp && (
            <button
              onClick={() => setShowAddMcp(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-colors"
              style={{ fontSize: '13px', color: 'var(--accent)', background: 'var(--accent-bg)', border: '1px solid var(--accent-ring)' }}
            >
              <Plus size={13} />
              {t('mcp.add')}
            </button>
          )}
        </div>

        <div className="space-y-3">
          {showAddMcp && (
            <AddMcpForm onAdd={handleAddMcp} onCancel={() => setShowAddMcp(false)} />
          )}

          {mcpServers.length === 0 && !showAddMcp && (
            <p style={{ fontSize: '14px', color: 'var(--text-4)' }}>{t('mcp.no_servers')}</p>
          )}

          {mcpServers.map((server) => (
            <McpCard
              key={server.id}
              server={server}
              onUpdate={(patch) => handleUpdateMcp(server.id, patch)}
              onDelete={() => handleDeleteMcp(server.id)}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
