import { useState, useCallback, useEffect } from 'react'
import { Folder, FolderOpen, File, FileText, ChevronRight, ChevronDown, Save, X, Loader } from 'lucide-react'
import { api } from '../../api/client'
import type { FsEntry } from '../../api/types'
import { useT } from '../../i18n'

const EDITABLE_EXTS = new Set(['.md', '.txt', '.yaml', '.yml', '.json', '.toml'])

function ext(name: string) {
  const i = name.lastIndexOf('.')
  return i >= 0 ? name.slice(i).toLowerCase() : ''
}

function FileIcon({ entry }: { entry: FsEntry }) {
  if (entry.type === 'dir') return <Folder size={14} style={{ color: 'var(--accent)' }} />
  const e = ext(entry.name)
  if (e === '.md' || e === '.txt') return <FileText size={14} style={{ color: 'var(--text-3)' }} />
  return <File size={14} style={{ color: 'var(--text-3)' }} />
}

// ── Directory row (expandable) ────────────────────────────────────────────────

interface DirRowProps {
  path: string
  name: string
  depth: number
  onSelectFile: (path: string) => void
  selectedPath: string | null
}

function DirRow({ path, name, depth, onSelectFile, selectedPath }: DirRowProps) {
  const [open, setOpen] = useState(false)
  const [entries, setEntries] = useState<FsEntry[]>([])
  const [loading, setLoading] = useState(false)

  async function toggle() {
    if (!open && entries.length === 0) {
      setLoading(true)
      try {
        const res = await api.fsList(path)
        setEntries(res.entries)
      } catch { /* ignore */ }
      setLoading(false)
    }
    setOpen((o) => !o)
  }

  return (
    <div>
      <button
        onClick={toggle}
        className="flex items-center gap-1.5 w-full rounded-lg px-2 py-1.5 transition-colors text-left"
        style={{
          paddingLeft: `${8 + depth * 16}px`,
          background: 'transparent',
          color: 'var(--text-1)',
          fontSize: '13px',
        }}
        onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-elevated)')}
        onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
      >
        <span style={{ color: 'var(--text-4)', flexShrink: 0 }}>
          {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        </span>
        {open ? <FolderOpen size={14} style={{ color: 'var(--accent)', flexShrink: 0 }} />
               : <Folder size={14} style={{ color: 'var(--accent)', flexShrink: 0 }} />}
        <span className="truncate">{name}</span>
        {loading && <Loader size={11} className="animate-spin ml-auto" style={{ color: 'var(--text-4)' }} />}
      </button>

      {open && (
        <div>
          {entries.length === 0 && !loading && (
            <div style={{ paddingLeft: `${8 + (depth + 1) * 16}px`, fontSize: '12px', color: 'var(--text-4)', padding: '4px 8px 4px ' + (8 + (depth + 1) * 16) + 'px' }}>
              empty
            </div>
          )}
          {entries.map((e) => (
            e.type === 'dir' ? (
              <DirRow
                key={e.name}
                path={path + '/' + e.name}
                name={e.name}
                depth={depth + 1}
                onSelectFile={onSelectFile}
                selectedPath={selectedPath}
              />
            ) : (
              <FileRow
                key={e.name}
                entry={e}
                path={path + '/' + e.name}
                depth={depth + 1}
                onSelect={onSelectFile}
                selected={selectedPath === path + '/' + e.name}
              />
            )
          ))}
        </div>
      )}
    </div>
  )
}

// ── File row ──────────────────────────────────────────────────────────────────

interface FileRowProps {
  entry:    FsEntry
  path:     string
  depth:    number
  onSelect: (path: string) => void
  selected: boolean
}

function FileRow({ entry, path, depth, onSelect, selected }: FileRowProps) {
  return (
    <button
      onClick={() => onSelect(path)}
      className="flex items-center gap-1.5 w-full rounded-lg px-2 py-1.5 transition-colors text-left"
      style={{
        paddingLeft: `${8 + depth * 16}px`,
        background: selected ? 'var(--accent-bg)' : 'transparent',
        color: selected ? 'var(--accent)' : 'var(--text-1)',
        fontSize: '13px',
      }}
      onMouseEnter={(e) => {
        if (!selected) e.currentTarget.style.background = 'var(--bg-elevated)'
      }}
      onMouseLeave={(e) => {
        if (!selected) e.currentTarget.style.background = 'transparent'
      }}
    >
      <span style={{ width: '12px', flexShrink: 0 }} />
      <FileIcon entry={entry} />
      <span className="truncate flex-1">{entry.name}</span>
      {ext(entry.name) === '.md' && (
        <span style={{ fontSize: '10px', color: 'var(--text-4)', fontFamily: 'monospace', flexShrink: 0 }}>md</span>
      )}
    </button>
  )
}

// ── Editor panel ──────────────────────────────────────────────────────────────

interface EditorProps {
  path: string
  onClose: () => void
}

function Editor({ path, onClose }: EditorProps) {
  const t = useT()
  const [content, setContent] = useState('')
  const [original, setOriginal] = useState('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)

  const parts0 = path.split('/')
  const canEdit = EDITABLE_EXTS.has(ext(parts0[parts0.length - 1] ?? ''))

  useEffect(() => {
    setLoading(true)
    setError(null)
    api.fsReadFile(path)
      .then((res) => { setContent(res.content); setOriginal(res.content) })
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false))
  }, [path])

  async function handleSave() {
    setSaving(true)
    setError(null)
    try {
      await api.fsWriteFile(path, content)
      setOriginal(content)
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } catch (e) {
      setError(String(e))
    }
    setSaving(false)
  }

  const parts1 = path.split('/')
  const fileName = parts1[parts1.length - 1] ?? ''
  const dirty = content !== original

  return (
    <div className="flex flex-col h-full">
      {/* Editor header */}
      <div
        className="flex items-center gap-2 px-4 py-2.5 shrink-0"
        style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-elevated)' }}
      >
        <FileText size={14} style={{ color: 'var(--accent)' }} />
        <span className="flex-1 font-mono truncate" style={{ fontSize: '13px', color: 'var(--text-1)' }}>
          {fileName}
          {dirty && <span style={{ color: 'var(--accent)', marginLeft: '6px' }}>●</span>}
        </span>
        {canEdit && dirty && (
          <button
            onClick={handleSave}
            disabled={saving}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium transition-colors disabled:opacity-50"
            style={{ fontSize: '12px', background: 'var(--accent)', color: '#fff' }}
          >
            {saving ? <Loader size={11} className="animate-spin" /> : <Save size={11} />}
            {t('fs.save')}
          </button>
        )}
        {saved && !dirty && (
          <span style={{ fontSize: '12px', color: '#10b981' }}>✓ Saved</span>
        )}
        <button
          onClick={onClose}
          className="p-1 rounded-lg transition-colors"
          style={{ color: 'var(--text-4)' }}
          onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--text-1)')}
          onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-4)')}
        >
          <X size={14} />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto">
        {loading && (
          <div className="flex items-center justify-center h-full" style={{ color: 'var(--text-4)' }}>
            <Loader size={20} className="animate-spin" />
          </div>
        )}
        {error && (
          <div className="p-4" style={{ color: '#ef4444', fontSize: '13px' }}>{error}</div>
        )}
        {!loading && !error && canEdit && (
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            className="w-full h-full resize-none p-4 font-mono"
            style={{ fontSize: '13px', lineHeight: 1.6, background: 'var(--bg-base)', border: 'none', outline: 'none', color: 'var(--text-1)' }}
            spellCheck={false}
          />
        )}
        {!loading && !error && !canEdit && (
          <div className="p-4">
            <div
              className="p-3 rounded-xl mb-3"
              style={{ background: 'var(--bg-elevated)', fontSize: '12px', color: 'var(--text-3)' }}
            >
              {t('fs.read_only')}
            </div>
            <pre className="font-mono whitespace-pre-wrap" style={{ fontSize: '12px', color: 'var(--text-2)', lineHeight: 1.6 }}>
              {content}
            </pre>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Main FileManager ──────────────────────────────────────────────────────────

interface FileManagerProps {
  rootPath: string
}

export function FileManager({ rootPath }: FileManagerProps) {
  const t = useT()
  const [rootEntries, setRootEntries] = useState<FsEntry[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedFile, setSelectedFile] = useState<string | null>(null)

  const loadRoot = useCallback(async () => {
    if (!rootPath) return
    setLoading(true)
    setError(null)
    try {
      const res = await api.fsList(rootPath)
      setRootEntries(res.entries)
    } catch (e) {
      setError(String(e))
    }
    setLoading(false)
  }, [rootPath])

  useEffect(() => {
    setSelectedFile(null)
    loadRoot()
  }, [loadRoot])

  if (!rootPath) {
    return (
      <div className="flex items-center justify-center h-40" style={{ color: 'var(--text-4)', fontSize: '14px' }}>
        {t('fs.enter_path')}
      </div>
    )
  }

  return (
    <div
      className="flex rounded-2xl overflow-hidden"
      style={{ border: '1px solid var(--border)', height: '400px', background: 'var(--bg-card)' }}
    >
      {/* File tree */}
      <div
        className="overflow-y-auto shrink-0 py-2"
        style={{ width: '220px', borderRight: '1px solid var(--border)', background: 'var(--bg-elevated)' }}
      >
        {loading && (
          <div className="flex items-center justify-center py-6" style={{ color: 'var(--text-4)' }}>
            <Loader size={16} className="animate-spin" />
          </div>
        )}
        {error && (
          <div className="px-3 py-2" style={{ fontSize: '12px', color: '#ef4444' }}>{error}</div>
        )}
        {!loading && !error && rootEntries.length === 0 && (
          <div className="px-3 py-2" style={{ fontSize: '12px', color: 'var(--text-4)' }}>{t('fs.empty_dir')}</div>
        )}
        {!loading && !error && rootEntries.map((e) => (
          e.type === 'dir' ? (
            <DirRow
              key={e.name}
              path={rootPath + '/' + e.name}
              name={e.name}
              depth={0}
              onSelectFile={setSelectedFile}
              selectedPath={selectedFile}
            />
          ) : (
            <FileRow
              key={e.name}
              entry={e}
              path={rootPath + '/' + e.name}
              depth={0}
              onSelect={setSelectedFile}
              selected={selectedFile === rootPath + '/' + e.name}
            />
          )
        ))}
      </div>

      {/* Editor / placeholder */}
      <div className="flex-1 min-w-0">
        {selectedFile ? (
          <Editor path={selectedFile} onClose={() => setSelectedFile(null)} />
        ) : (
          <div className="flex items-center justify-center h-full" style={{ color: 'var(--text-4)', fontSize: '13px' }}>
            Select a file to view or edit
          </div>
        )}
      </div>
    </div>
  )
}
