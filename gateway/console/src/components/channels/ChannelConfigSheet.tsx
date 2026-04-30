import { useState } from 'react'
import { api } from '@gw/api/client'
import type { ChannelConfigResponse } from '@gw/api/types'
import { useT } from '@gw/i18n'

interface Props {
  cfg: ChannelConfigResponse
  onSaved: () => void
}

export function ChannelConfigSheet({ cfg, onSaved }: Props) {
  const t = useT()
  const [local, setLocal] = useState<Record<string, unknown>>(() => ({ ...cfg.config }))
  const [status, setStatus] = useState<'idle' | 'saving' | 'saved'>('idle')
  const [error, setError] = useState<string | null>(null)

  const properties = cfg.schema?.properties ?? {}
  const required = new Set(cfg.schema?.required ?? [])

  const set = (key: string, value: unknown) =>
    setLocal((prev) => ({ ...prev, [key]: value }))

  const handleSave = async () => {
    setStatus('saving')
    setError(null)
    try {
      await api.updateChannelConfig(cfg.name, local)
      setStatus('saved')
      onSaved()
      setTimeout(() => setStatus('idle'), 2000)
    } catch (e) {
      setError(String(e))
      setStatus('idle')
    }
  }

  return (
    <div className="flex flex-col gap-4">
      {Object.entries(properties).map(([key, prop]) => {
        const isPassword = prop.format === 'password'
        const isBoolean  = prop.type === 'boolean'
        const isArray    = prop.type === 'array'
        const title      = prop.title ?? key
        const isReq      = required.has(key)
        const val        = local[key]

        return (
          <div key={key} className="flex flex-col gap-1.5">
            <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-2)' }}>
              {title}
              {isReq && <span style={{ color: 'var(--accent)', marginLeft: 3 }}>*</span>}
            </label>

            {isBoolean ? (
              <button
                className="flex items-center gap-2 w-fit px-2 py-1 rounded"
                style={{
                  fontSize: 12,
                  background: val ? 'var(--accent-bg)' : 'var(--bg-elevated)',
                  color: val ? 'var(--accent)' : 'var(--text-3)',
                  border: `1px solid ${val ? 'var(--accent-ring)' : 'var(--border)'}`,
                }}
                onClick={() => set(key, !val)}
              >
                <span>{val ? '✓' : '○'}</span>
                <span>{val ? 'Enabled' : 'Disabled'}</span>
              </button>
            ) : isArray ? (
              <textarea
                rows={3}
                className="rounded px-2.5 py-1.5 w-full resize-none"
                style={{
                  fontSize: 12,
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-1)',
                  outline: 'none',
                  fontFamily: 'monospace',
                }}
                placeholder="One item per line"
                value={Array.isArray(val) ? (val as string[]).join('\n') : String(val ?? '')}
                onChange={(e) => set(key, e.target.value.split('\n').filter(Boolean))}
              />
            ) : (
              <input
                type={isPassword ? 'password' : prop.type === 'integer' ? 'number' : 'text'}
                className="rounded px-2.5 py-1.5 w-full"
                style={{
                  fontSize: 12,
                  background: 'var(--bg-elevated)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-1)',
                  outline: 'none',
                  fontFamily: isPassword ? 'monospace' : undefined,
                }}
                value={isPassword && val === '***' ? '' : String(val ?? '')}
                placeholder={isPassword ? '(unchanged)' : String(prop.default ?? '')}
                onChange={(e) =>
                  set(
                    key,
                    prop.type === 'integer' ? Number(e.target.value) : e.target.value,
                  )
                }
              />
            )}
          </div>
        )
      })}

      {error && (
        <p style={{ fontSize: 12, color: '#ff3b30' }}>{error}</p>
      )}

      <div className="flex items-center gap-3 pt-1">
        <button
          onClick={handleSave}
          disabled={status === 'saving'}
          className="px-3 py-1.5 rounded-lg font-medium transition-all"
          style={{
            fontSize: 13,
            background: status === 'saved' ? 'rgba(52,199,89,0.15)' : 'var(--accent-bg)',
            color: status === 'saved' ? '#34c759' : 'var(--accent)',
            border: `1px solid ${status === 'saved' ? 'rgba(52,199,89,0.3)' : 'var(--accent-ring)'}`,
            opacity: status === 'saving' ? 0.6 : 1,
          }}
        >
          {status === 'saving' ? '…' : status === 'saved' ? t('ch.config.saved') : t('ch.config.save')}
        </button>
        <span style={{ fontSize: 11, color: 'var(--text-4)' }}>{t('ch.config.note')}</span>
      </div>
    </div>
  )
}
