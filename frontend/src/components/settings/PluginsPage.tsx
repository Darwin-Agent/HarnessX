import { useState, useEffect } from 'react'
import {
  Puzzle, FolderOpen, Trash2, Plus, Loader, Wrench, BookOpen, Cpu,
  FolderSearch, Info, X,
} from 'lucide-react'
import { api } from '../../api/client'
import type { PluginInfo } from '../../api/types'
import { useT } from '../../i18n'
import { FileManager } from './FileManager'
import { DocLink } from '../docs/DocLink'

// ── Plugin card ───────────────────────────────────────────────────────────────

interface PluginCardProps {
  plugin:   PluginInfo
  onToggle: () => void
  onRemove: () => void
}

function PluginCard({ plugin, onToggle, onRemove }: PluginCardProps) {
  const t = useT()
  const [showFiles, setShowFiles] = useState(false)

  return (
    <div
      className="rounded-2xl overflow-hidden transition-all duration-150"
      style={{ border: '1px solid var(--border)', background: 'var(--bg-card)' }}
    >
      <div className="p-4">
        <div className="flex items-start gap-3">
          <div
            className="rounded-xl p-2.5 shrink-0"
            style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)' }}
          >
            <Puzzle size={16} style={{ color: plugin.enabled ? 'var(--accent)' : 'var(--text-3)' }} />
          </div>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-semibold truncate" style={{ fontSize: '15px', color: 'var(--text-1)' }}>
                {plugin.name}
              </span>
              <span
                className="px-1.5 py-0.5 rounded-md font-mono"
                style={{ fontSize: '10px', background: 'var(--bg-elevated)', color: 'var(--text-4)', border: '1px solid var(--border)', flexShrink: 0 }}
              >
                v{plugin.version}
              </span>
            </div>

            <p className="mt-1 line-clamp-2" style={{ fontSize: '13px', color: 'var(--text-3)', lineHeight: 1.5 }}>
              {plugin.description || 'No description'}
            </p>

            {/* Stats row */}
            <div className="flex items-center gap-3 mt-2">
              {plugin.tool_count > 0 && (
                <span className="flex items-center gap-1" style={{ fontSize: '12px', color: 'var(--text-3)' }}>
                  <Wrench size={11} /> {plugin.tool_count} {t('plugin.tools')}
                </span>
              )}
              {plugin.skill_count > 0 && (
                <span className="flex items-center gap-1" style={{ fontSize: '12px', color: 'var(--text-3)' }}>
                  <BookOpen size={11} /> {plugin.skill_count} {t('plugin.skills')}
                </span>
              )}
              {plugin.mcp_count > 0 && (
                <span className="flex items-center gap-1" style={{ fontSize: '12px', color: 'var(--text-3)' }}>
                  <Cpu size={11} /> {plugin.mcp_count} {t('plugin.mcp')}
                </span>
              )}
            </div>

            {/* Path (dim, truncated) */}
            {plugin.path && (
              <p className="mt-1 truncate font-mono" style={{ fontSize: '11px', color: 'var(--text-4)' }}>
                {plugin.path}
              </p>
            )}
          </div>

          {/* Controls */}
          <div className="flex items-center gap-1.5 shrink-0">
            <button
              onClick={onToggle}
              className="px-3 py-1.5 rounded-lg font-medium transition-all duration-150"
              style={{
                fontSize: '12px',
                ...(plugin.enabled
                  ? { background: 'rgba(16,185,129,0.08)', color: '#10b981', border: '1px solid rgba(16,185,129,0.2)' }
                  : { background: 'var(--bg-elevated)', color: 'var(--text-3)', border: '1px solid var(--border)' }),
              }}
            >
              {plugin.enabled ? 'ON' : 'OFF'}
            </button>
            <button
              onClick={() => setShowFiles((o) => !o)}
              className="p-1.5 rounded-lg transition-colors"
              style={{ color: showFiles ? 'var(--accent)' : 'var(--text-3)', border: '1px solid var(--border)' }}
              title={t('plugin.browse')}
              onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--accent)')}
              onMouseLeave={(e) => (e.currentTarget.style.color = showFiles ? 'var(--accent)' : 'var(--text-3)')}
            >
              <FolderOpen size={14} />
            </button>
            <button
              onClick={onRemove}
              className="p-1.5 rounded-lg transition-colors"
              style={{ color: 'var(--text-3)', border: '1px solid var(--border)' }}
              title={t('plugin.remove')}
              onMouseEnter={(e) => { e.currentTarget.style.color = '#ef4444'; e.currentTarget.style.borderColor = '#ef444440' }}
              onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-3)'; e.currentTarget.style.borderColor = 'var(--border)' }}
            >
              <Trash2 size={14} />
            </button>
          </div>
        </div>
      </div>

      {/* File manager */}
      {showFiles && plugin.path && (
        <div className="px-4 pb-4" style={{ borderTop: '1px solid var(--border)' }}>
          <div className="pt-3">
            <FileManager rootPath={plugin.path} />
          </div>
        </div>
      )}
    </div>
  )
}

// ── Scan directories section ──────────────────────────────────────────────────

interface ScanDirsSectionProps {
  onRefreshPlugins: () => void
}

function ScanDirsSection({ onRefreshPlugins }: ScanDirsSectionProps) {
  const t = useT()
  const [scanDirs, setScanDirs] = useState<string[]>([])
  const [loading, setLoading]   = useState(true)
  const [showAdd, setShowAdd]   = useState(false)
  const [addPath, setAddPath]   = useState('')
  const [adding, setAdding]     = useState(false)
  const [error, setError]       = useState<string | null>(null)

  useEffect(() => {
    api.pluginScanDirs()
      .then((r) => setScanDirs(r.scan_dirs))
      .catch(() => setScanDirs([]))
      .finally(() => setLoading(false))
  }, [])

  async function handleAdd() {
    const p = addPath.trim()
    if (!p) return
    setAdding(true); setError(null)
    try {
      const r = await api.pluginAddScanDir(p)
      setScanDirs(r.scan_dirs)
      setAddPath(''); setShowAdd(false)
      onRefreshPlugins()
    } catch (e) {
      setError(String(e))
    } finally {
      setAdding(false)
    }
  }

  async function handleRemove(dir: string) {
    try {
      const r = await api.pluginRemoveScanDir(dir)
      setScanDirs(r.scan_dirs)
      onRefreshPlugins()
    } catch { /* ignore */ }
  }

  const fieldLabel: React.CSSProperties = {
    fontSize: '11px', color: 'var(--text-3)',
    fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.06em',
    textTransform: 'uppercase', display: 'block', marginBottom: '8px',
  }

  return (
    <div
      className="rounded-2xl p-5 space-y-3"
      style={{ border: '1px solid var(--border)', background: 'var(--bg-card)' }}
    >
      <div className="flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <FolderSearch size={15} style={{ color: 'var(--accent)' }} />
            <label style={{ ...fieldLabel, marginBottom: 0 }}>{t('plugin.scan_dirs')}</label>
          </div>
          <p className="mt-1" style={{ fontSize: '12.5px', color: 'var(--text-3)' }}>
            {t('plugin.scan_dirs_desc')}
          </p>
        </div>
        {!showAdd && (
          <button
            onClick={() => setShowAdd(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-colors shrink-0"
            style={{ fontSize: '12.5px', color: 'var(--accent)', background: 'var(--accent-bg)', border: '1px solid var(--accent-ring)' }}
          >
            <Plus size={12} />{t('plugin.scan_add')}
          </button>
        )}
      </div>

      {loading ? (
        <div style={{ color: 'var(--text-4)', fontSize: '13px' }}>Loading…</div>
      ) : scanDirs.length === 0 ? (
        <div
          className="px-3 py-2.5 rounded-xl"
          style={{ fontSize: '12.5px', color: 'var(--text-4)', background: 'var(--bg-elevated)', fontStyle: 'italic' }}
        >
          {t('plugin.scan_empty')}
        </div>
      ) : (
        <div className="space-y-1.5">
          {scanDirs.map((dir, idx) => {
            const isBuiltin = idx === 0
            return (
              <div
                key={dir}
                className="flex items-center gap-3 px-3 py-2 rounded-xl"
                style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)' }}
              >
                <FolderSearch size={13} style={{ color: 'var(--text-4)', flexShrink: 0 }} />
                <span className="flex-1 font-mono truncate" style={{ fontSize: '12.5px', color: 'var(--text-2)' }}>
                  {dir}
                </span>
                {isBuiltin ? (
                  <span style={{ fontSize: '10px', color: 'var(--text-4)', flexShrink: 0 }}>built-in</span>
                ) : (
                  <button
                    onClick={() => handleRemove(dir)}
                    className="p-1 rounded-lg transition-colors shrink-0"
                    style={{ color: 'var(--text-4)' }}
                    onMouseEnter={(e) => (e.currentTarget.style.color = '#ef4444')}
                    onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-4)')}
                  >
                    <X size={12} />
                  </button>
                )}
              </div>
            )
          })}
        </div>
      )}

      {showAdd && (
        <div className="space-y-2">
          <input
            autoFocus
            type="text"
            value={addPath}
            onChange={(e) => setAddPath(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') handleAdd(); if (e.key === 'Escape') { setShowAdd(false); setAddPath('') } }}
            placeholder={t('plugin.scan_add_hint')}
            className="w-full rounded-xl px-3.5 py-2.5"
            style={{ fontSize: '13px', border: `1px solid ${error ? '#ef4444' : 'var(--accent-ring)'}` }}
          />
          {error && <p style={{ fontSize: '12px', color: '#ef4444' }}>{error}</p>}
          <div className="flex gap-2">
            <button
              onClick={handleAdd}
              disabled={!addPath.trim() || adding}
              className="px-4 py-2 rounded-xl text-sm font-medium disabled:opacity-40"
              style={{ background: 'var(--accent)', color: '#fff' }}
            >
              {adding ? 'Adding…' : 'Add'}
            </button>
            <button
              onClick={() => { setShowAdd(false); setAddPath(''); setError(null) }}
              className="px-3.5 py-2 rounded-xl text-sm"
              style={{ color: 'var(--text-3)', border: '1px solid var(--border)' }}
            >
              {t('fs.cancel')}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Import form ───────────────────────────────────────────────────────────────

interface ImportFormProps {
  onImport: (path: string) => void
  onCancel: () => void
}

function ImportForm({ onImport, onCancel }: ImportFormProps) {
  const t = useT()
  const [path, setPath] = useState('')

  return (
    <div className="rounded-2xl p-4 space-y-3" style={{ border: '1px solid var(--accent-ring)', background: 'var(--accent-bg)' }}>
      <p style={{ fontSize: '12.5px', color: 'var(--text-2)', lineHeight: 1.5 }}>
        Import an individual plugin directory. The directory must contain a <code style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11.5px' }}>plugin.json</code> file.
      </p>
      <div>
        <input
          autoFocus
          type="text"
          value={path}
          onChange={(e) => setPath(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') onImport(path.trim()); if (e.key === 'Escape') onCancel() }}
          placeholder={t('plugin.import_hint')}
          className="w-full rounded-xl px-3.5 py-2.5"
          style={{ fontSize: '13px' }}
        />
      </div>
      <div className="flex gap-2 justify-end">
        <button
          onClick={onCancel}
          className="px-3.5 py-1.5 rounded-lg"
          style={{ fontSize: '13px', color: 'var(--text-3)', border: '1px solid var(--border)' }}
        >
          {t('fs.cancel')}
        </button>
        <button
          onClick={() => onImport(path.trim())}
          disabled={!path.trim()}
          className="px-3.5 py-1.5 rounded-lg font-medium disabled:opacity-40"
          style={{ fontSize: '13px', background: 'var(--accent)', color: '#fff' }}
        >
          {t('plugin.import')}
        </button>
      </div>
    </div>
  )
}

// ── Page ─────────────────────────────────────────────────────────────────────

export function PluginsPage() {
  const t = useT()
  const [plugins, setPlugins]   = useState<PluginInfo[]>([])
  const [loading, setLoading]   = useState(true)
  const [showImport, setShowImport] = useState(false)

  function loadPlugins() {
    setLoading(true)
    api.plugins()
      .then(setPlugins)
      .catch(() => setPlugins([]))
      .finally(() => setLoading(false))
  }

  useEffect(() => { loadPlugins() }, [])

  async function handleToggle(plugin: PluginInfo) {
    try {
      const updated = await api.pluginPatch(plugin.id, { enabled: !plugin.enabled })
      setPlugins((prev) => prev.map((p) => (p.id === plugin.id ? updated : p)))
    } catch { /* ignore */ }
  }

  async function handleRemove(plugin: PluginInfo) {
    if (!confirm(`Remove plugin "${plugin.name}"? (Files will not be deleted)`)) return
    try {
      await api.pluginRemove(plugin.id)
      setPlugins((prev) => prev.filter((p) => p.id !== plugin.id))
    } catch { /* ignore */ }
  }

  async function handleImport(path: string) {
    try {
      const newPlugin = await api.pluginImport(path)
      setPlugins((prev) => [...prev, newPlugin])
      setShowImport(false)
    } catch (e) {
      alert(String(e))
    }
  }

  return (
    <div className="space-y-5">

      {/* Install hint */}
      <div
        className="flex items-start gap-3 px-4 py-3 rounded-2xl"
        style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)' }}
      >
        <Info size={14} style={{ color: 'var(--text-4)', flexShrink: 0, marginTop: '2px' }} />
        <p style={{ fontSize: '12.5px', color: 'var(--text-3)', lineHeight: 1.6 }}>
          {t('plugin.install_hint')}
          <br />
          <code style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11.5px', color: 'var(--text-2)' }}>
            claude plugin install &lt;name&gt;
          </code>
          {' '}or{' '}
          <code style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11.5px', color: 'var(--text-2)' }}>
            harnessx plugin install &lt;name&gt;
          </code>
        </p>
      </div>

      {/* Scan directories */}
      <ScanDirsSection onRefreshPlugins={loadPlugins} />

      {/* Plugins list header */}
      <div className="flex items-center justify-between">
        <p style={{ fontSize: '13.5px', color: 'var(--text-3)', lineHeight: 1.6 }}>
          {loading ? '' : `${plugins.length} plugin${plugins.length !== 1 ? 's' : ''} discovered`}
        </p>
        <div className="flex gap-2">
          <DocLink path="feats/plugins" label="Docs" />
          {!showImport && (
            <button
              onClick={() => setShowImport(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition-colors"
              style={{ fontSize: '13px', color: 'var(--accent)', background: 'var(--accent-bg)', border: '1px solid var(--accent-ring)' }}
            >
              <Plus size={13} />
              {t('plugin.import')}
            </button>
          )}
        </div>
      </div>

      {showImport && (
        <ImportForm onImport={handleImport} onCancel={() => setShowImport(false)} />
      )}

      {loading && (
        <div className="flex items-center justify-center py-12" style={{ color: 'var(--text-4)' }}>
          <Loader size={20} className="animate-spin" />
        </div>
      )}

      {!loading && plugins.length === 0 && !showImport && (
        <div
          className="text-center py-12 rounded-2xl"
          style={{ border: '1px dashed var(--border)', color: 'var(--text-4)', fontSize: '14px' }}
        >
          {t('plugin.no_plugins')}
        </div>
      )}

      <div className="space-y-3">
        {plugins.map((plugin) => (
          <PluginCard
            key={plugin.id}
            plugin={plugin}
            onToggle={() => handleToggle(plugin)}
            onRemove={() => handleRemove(plugin)}
          />
        ))}
      </div>
    </div>
  )
}
