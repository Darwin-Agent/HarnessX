import { useRef, useState } from 'react'
import { Download, Upload } from 'lucide-react'
import type { HarnessConfig } from '@gw/api/types'
import { useT } from '@gw/i18n'

interface Props {
  harnessConfig: HarnessConfig
  onImport: (cfg: HarnessConfig) => Promise<void>
  title?: string
  hint?: string
}

export function HarnessReadonlySection({ harnessConfig, onImport, title, hint }: Props) {
  const t = useT()
  const fileRef = useRef<HTMLInputElement>(null)
  const [importError, setImportError] = useState<string | null>(null)
  const [importing, setImporting] = useState(false)
  const [imported, setImported] = useState(false)

  const handleExport = () => {
    const blob = new Blob([JSON.stringify(harnessConfig, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'harness_config.json'
    a.click()
    URL.revokeObjectURL(url)
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    setImportError(null)
    setImporting(true)
    setImported(false)
    try {
      const text = await file.text()
      const parsed = JSON.parse(text) as HarnessConfig
      if (!Array.isArray(parsed?.processors)) throw new Error('Invalid harness_config.json: missing processors array')
      await onImport(parsed)
      setImported(true)
      setTimeout(() => setImported(false), 2500)
    } catch (e) {
      setImportError(String(e))
    } finally {
      setImporting(false)
    }
  }

  const processors = harnessConfig.processors ?? []

  return (
    <div className="flex flex-col gap-3 max-w-lg">
      <div
        className="flex items-start justify-between gap-4 p-4 rounded-xl"
        style={{ border: '1px solid var(--border)', background: 'var(--bg-card)' }}
      >
        <div className="flex flex-col gap-0.5">
          <span className="font-semibold" style={{ fontSize: 13, color: 'var(--text-1)' }}>
            {title ?? t('gw.settings.harness_title')}
          </span>
          {hint && (
            <span style={{ fontSize: 12, color: 'var(--text-4)' }}>{hint}</span>
          )}
          <span style={{ fontSize: 11, color: 'var(--text-4)', marginTop: 2 }}>
            {processors.length} processors · {t('gw.harness.readonly_hint')}
          </span>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={handleExport}
            disabled={processors.length === 0}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg font-medium"
            style={{
              fontSize: 12,
              background: 'var(--bg-elevated)',
              color: processors.length === 0 ? 'var(--text-4)' : 'var(--text-2)',
              border: '1px solid var(--border)',
              opacity: processors.length === 0 ? 0.5 : 1,
            }}
            title={t('gw.harness.export')}
          >
            <Download size={11} />
            {t('gw.harness.export')}
          </button>

          <button
            onClick={() => fileRef.current?.click()}
            disabled={importing}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg font-medium"
            style={{
              fontSize: 12,
              background: imported ? 'rgba(52,199,89,0.12)' : 'var(--accent-bg)',
              color: imported ? '#34c759' : 'var(--accent)',
              border: `1px solid ${imported ? 'rgba(52,199,89,0.3)' : 'var(--accent-ring)'}`,
              opacity: importing ? 0.6 : 1,
            }}
            title={t('gw.harness.import')}
          >
            <Upload size={11} />
            {importing ? '…' : imported ? t('gw.settings.saved') : t('gw.harness.import')}
          </button>

          <input
            ref={fileRef}
            type="file"
            accept=".json"
            style={{ display: 'none' }}
            onChange={handleFileChange}
          />
        </div>
      </div>

      {importError && (
        <p style={{ fontSize: 12, color: '#ff3b30' }}>{importError}</p>
      )}

      {processors.length > 0 && (
        <div className="flex flex-col gap-1">
          {processors.map((p, i) => (
            <div
              key={i}
              className="px-3 py-2 rounded-lg font-mono"
              style={{ fontSize: 11, background: 'var(--bg-elevated)', color: 'var(--text-3)', border: '1px solid var(--border)' }}
            >
              {String((p as Record<string, unknown>)._target_ ?? '(unknown)').split('.').pop()}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
