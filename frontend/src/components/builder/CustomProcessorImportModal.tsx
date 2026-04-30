import { useMemo, useState } from 'react'
import { FlaskConical, Upload, FolderSearch, CheckCircle2, AlertCircle, X } from 'lucide-react'
import { api } from '../../api/client'
import type { CustomProcessorCandidate } from '../../api/types'

interface Props {
  open: boolean
  onClose: () => void
  onImported: () => Promise<void> | void
}

type Mode = 'path' | 'file'

type TestResult = {
  ok: boolean
  instantiable: boolean
  required_args: string[]
  message: string
}

export function CustomProcessorImportModal({ open, onClose, onImported }: Props) {
  const [mode, setMode] = useState<Mode>('path')
  const [path, setPath] = useState('')
  const [filename, setFilename] = useState('')
  const [content, setContent] = useState('')
  const [loading, setLoading] = useState(false)
  const [candidates, setCandidates] = useState<CustomProcessorCandidate[]>([])
  const [error, setError] = useState<string | null>(null)
  const [tests, setTests] = useState<Record<string, TestResult>>({})
  const [importingKey, setImportingKey] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)

  const canScan = useMemo(() => {
    if (mode === 'path') return path.trim().length > 0
    return filename.trim().length > 0 && content.trim().length > 0
  }, [mode, path, filename, content])

  if (!open) return null

  function keyOf(c: CustomProcessorCandidate): string {
    return `${c.file_path}::${c.class_name}`
  }

  async function scan() {
    setLoading(true)
    setError(null)
    setNotice(null)
    setTests({})
    try {
      if (mode === 'path') {
        const res = await api.customProcessorScanPath(path.trim())
        setCandidates(res.candidates)
      } else {
        const res = await api.customProcessorScanFile(filename.trim(), content)
        setCandidates(res.candidates)
      }
    } catch (e) {
      setCandidates([])
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  async function testCandidate(c: CustomProcessorCandidate) {
    const k = keyOf(c)
    setError(null)
    try {
      const res = await api.customProcessorTest(
        mode === 'path'
          ? {
              mode: 'path',
              class_name: c.class_name,
              path: path.trim(),
              file_path: c.file_path,
            }
          : {
              mode: 'file',
              class_name: c.class_name,
              filename: filename.trim() || c.file_path || 'uploaded_processor.py',
              content,
            },
      )
      setTests((prev) => ({ ...prev, [k]: res }))
    } catch (e) {
      setTests((prev) => ({
        ...prev,
        [k]: { ok: false, instantiable: false, required_args: [], message: String(e) },
      }))
    }
  }

  async function importCandidate(c: CustomProcessorCandidate) {
    const k = keyOf(c)
    setImportingKey(k)
    setError(null)
    setNotice(null)
    try {
      await api.customProcessorImport(
        mode === 'path'
          ? {
              mode: 'path',
              class_name: c.class_name,
              label: c.label,
              path: path.trim(),
              file_path: c.file_path,
            }
          : {
              mode: 'file',
              class_name: c.class_name,
              label: c.label,
              filename: filename.trim() || c.file_path || 'uploaded_processor.py',
              content,
            },
      )
      setNotice(`Imported: ${c.class_name}`)
      await onImported()
    } catch (e) {
      setError(String(e))
    } finally {
      setImportingKey(null)
    }
  }

  async function onFileSelected(file: File | null) {
    if (!file) return
    const text = await file.text()
    setFilename(file.name)
    setContent(text)
    setCandidates([])
    setTests({})
    setError(null)
    setNotice(null)
  }

  return (
    <>
      <div
        className="fixed inset-0 z-40"
        style={{ background: 'rgba(0,0,0,0.3)' }}
        onClick={onClose}
      />
      <div
        className="fixed z-50 left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 w-[min(960px,96vw)] max-h-[88vh] flex flex-col overflow-hidden rounded-2xl"
        style={{ background: 'var(--bg-base)', border: '1px solid var(--border)', boxShadow: 'var(--shadow-card-drag)' }}
      >
        <div className="flex items-center gap-2 px-4 py-3 shrink-0" style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-card)' }}>
          <FlaskConical size={15} style={{ color: 'var(--accent)' }} />
          <span className="font-semibold text-sm" style={{ color: 'var(--text-1)' }}>Import Custom Processor</span>
          <div className="flex-1" />
          <button
            onClick={onClose}
            className="p-1 rounded transition-colors"
            style={{ color: 'var(--text-3)' }}
          >
            <X size={14} />
          </button>
        </div>

        <div className="px-4 py-3 shrink-0 space-y-3" style={{ borderBottom: '1px solid var(--border-sub)' }}>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setMode('path')}
              className="px-2.5 py-1 rounded text-xs"
              style={{
                color: mode === 'path' ? 'var(--accent)' : 'var(--text-3)',
                background: mode === 'path' ? 'var(--accent-bg)' : 'transparent',
                border: '1px solid var(--border)',
              }}
            >
              Path
            </button>
            <button
              onClick={() => setMode('file')}
              className="px-2.5 py-1 rounded text-xs"
              style={{
                color: mode === 'file' ? 'var(--accent)' : 'var(--text-3)',
                background: mode === 'file' ? 'var(--accent-bg)' : 'transparent',
                border: '1px solid var(--border)',
              }}
            >
              File Upload
            </button>
          </div>

          {mode === 'path' ? (
            <div className="flex items-center gap-2">
              <input
                value={path}
                onChange={(e) => setPath(e.target.value)}
                placeholder="Path to .py file or directory"
                className="flex-1 rounded px-3 py-2 text-sm"
                style={{ border: '1px solid var(--border)' }}
              />
              <button
                onClick={scan}
                disabled={!canScan || loading}
                className="px-3 py-2 rounded text-sm disabled:opacity-50"
                style={{ border: '1px solid var(--border)', color: 'var(--text-2)' }}
              >
                <span className="inline-flex items-center gap-1.5">
                  <FolderSearch size={13} />
                  Scan
                </span>
              </button>
            </div>
          ) : (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <label
                  className="px-3 py-2 rounded text-sm cursor-pointer"
                  style={{ border: '1px solid var(--border)', color: 'var(--text-2)' }}
                >
                  <span className="inline-flex items-center gap-1.5">
                    <Upload size={13} />
                    Choose .py
                  </span>
                  <input
                    type="file"
                    accept=".py,text/x-python"
                    className="hidden"
                    onChange={(e) => onFileSelected(e.target.files?.[0] ?? null)}
                  />
                </label>
                <span className="text-xs font-mono truncate" style={{ color: 'var(--text-4)' }}>
                  {filename || 'No file selected'}
                </span>
                <div className="flex-1" />
                <button
                  onClick={scan}
                  disabled={!canScan || loading}
                  className="px-3 py-2 rounded text-sm disabled:opacity-50"
                  style={{ border: '1px solid var(--border)', color: 'var(--text-2)' }}
                >
                  Scan
                </button>
              </div>
            </div>
          )}

          {error && (
            <div className="text-xs px-2 py-1 rounded" style={{ color: '#ef4444', background: 'rgba(239,68,68,0.08)' }}>
              {error}
            </div>
          )}
          {notice && (
            <div className="text-xs px-2 py-1 rounded" style={{ color: '#10b981', background: 'rgba(16,185,129,0.08)' }}>
              {notice}
            </div>
          )}
        </div>

        <div className="flex-1 overflow-auto p-4">
          {loading ? (
            <div className="text-sm" style={{ color: 'var(--text-4)' }}>Scanning…</div>
          ) : candidates.length === 0 ? (
            <div className="text-sm" style={{ color: 'var(--text-4)' }}>
              No processor class found. The class must subclass <code>MultiHookProcessor</code>.
            </div>
          ) : (
            <div className="space-y-2">
              {candidates.map((c) => {
                const k = keyOf(c)
                const tr = tests[k]
                return (
                  <div
                    key={k}
                    className="rounded-xl p-3"
                    style={{ border: '1px solid var(--border)', background: 'var(--bg-card)' }}
                  >
                    <div className="flex items-start gap-3">
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-semibold" style={{ color: 'var(--text-1)' }}>
                          {c.class_name}
                        </div>
                        <div className="text-xs font-mono truncate mt-0.5" style={{ color: 'var(--text-4)' }}>
                          {c.file_path}
                        </div>
                        {c.doc && (
                          <div className="text-xs mt-1" style={{ color: 'var(--text-3)' }}>
                            {c.doc}
                          </div>
                        )}
                        {tr && (
                          <div
                            className="mt-2 text-xs px-2 py-1 rounded inline-flex items-center gap-1.5"
                            style={{
                              color: tr.ok ? '#10b981' : '#ef4444',
                              background: tr.ok ? 'rgba(16,185,129,0.08)' : 'rgba(239,68,68,0.08)',
                            }}
                          >
                            {tr.ok ? <CheckCircle2 size={12} /> : <AlertCircle size={12} />}
                            {tr.message}
                          </div>
                        )}
                      </div>
                      <div className="flex items-center gap-1 shrink-0">
                        <button
                          onClick={() => testCandidate(c)}
                          className="px-2 py-1 rounded text-xs"
                          style={{ border: '1px solid var(--border)', color: 'var(--text-2)' }}
                        >
                          Test
                        </button>
                        <button
                          onClick={() => importCandidate(c)}
                          disabled={importingKey === k}
                          className="px-2 py-1 rounded text-xs disabled:opacity-50"
                          style={{ border: '1px solid var(--accent-ring)', color: 'var(--accent)', background: 'var(--accent-bg)' }}
                        >
                          {importingKey === k ? 'Importing…' : 'Import'}
                        </button>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </div>
    </>
  )
}
