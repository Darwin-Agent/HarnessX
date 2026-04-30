import { useEffect, useState } from 'react'
import { RefreshCw } from 'lucide-react'
import { api } from '@gw/api/client'
import type { GatewaySessionMeta } from '@gw/api/types'
import { useT } from '@gw/i18n'

const STATE_COLORS: Record<string, string> = {
  online:     '#34c759',
  connecting: '#ff9f0a',
  offline:    '#8e8e93',
  error:      '#ff3b30',
}

export function SessionsPage() {
  const t = useT()
  const [sessions, setSessions] = useState<GatewaySessionMeta[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [filter, setFilter] = useState('')

  const load = () => {
    setLoading(true)
    setError(null)
    api.listGatewaySessions()
      .then((s) => { setSessions(s); setLoading(false) })
      .catch((e) => { setError(String(e)); setLoading(false) })
  }

  useEffect(() => { load() }, [])

  const filtered = filter
    ? sessions.filter(
        (s) =>
          s.session_id.toLowerCase().includes(filter.toLowerCase()) ||
          s.channel.toLowerCase().includes(filter.toLowerCase()) ||
          (s.first_query ?? '').toLowerCase().includes(filter.toLowerCase()),
      )
    : sessions

  // Group by channel
  const byChannel = filtered.reduce<Record<string, GatewaySessionMeta[]>>((acc, s) => {
    ;(acc[s.channel] ??= []).push(s)
    return acc
  }, {})

  return (
    <div className="flex-1 overflow-y-auto p-6">
      {/* Toolbar */}
      <div className="flex items-center gap-3 mb-5 max-w-3xl">
        <h2 className="font-semibold shrink-0" style={{ fontSize: 16, color: 'var(--text-1)' }}>
          {t('gw.sessions.title')}
        </h2>
        <input
          type="text"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter…"
          style={{
            fontSize: 12,
            padding: '5px 10px',
            borderRadius: 8,
            border: '1px solid var(--border)',
            background: 'var(--bg-elevated)',
            color: 'var(--text-1)',
            outline: 'none',
            flex: 1,
            minWidth: 0,
          }}
        />
        <button
          onClick={load}
          className="p-1.5 rounded-lg"
          style={{ color: 'var(--text-4)', background: 'transparent' }}
          onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--text-1)'; e.currentTarget.style.background = 'var(--bg-elevated)' }}
          onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-4)'; e.currentTarget.style.background = 'transparent' }}
        >
          <RefreshCw size={13} />
        </button>
      </div>

      {loading && <p style={{ fontSize: 13, color: 'var(--text-4)' }}>Loading…</p>}
      {!loading && error && <p style={{ fontSize: 13, color: '#ff3b30' }}>{error}</p>}
      {!loading && !error && sessions.length === 0 && (
        <p style={{ fontSize: 13, color: 'var(--text-4)' }}>{t('gw.sessions.empty')}</p>
      )}

      <div className="flex flex-col gap-6 max-w-3xl">
        {Object.entries(byChannel).map(([ch, items]) => (
          <div key={ch}>
            <div className="flex items-center gap-2 mb-2">
              <span
                className="px-2 py-0.5 rounded font-mono font-semibold"
                style={{ fontSize: 11, background: 'var(--bg-elevated)', color: 'var(--text-2)', border: '1px solid var(--border)' }}
              >
                {ch}
              </span>
              <span style={{ fontSize: 11, color: 'var(--text-4)' }}>{items.length} sessions</span>
            </div>

            <div className="flex flex-col gap-1.5">
              {items.map((s) => (
                <div
                  key={s.session_id}
                  className="flex items-start gap-3 px-4 py-3 rounded-xl"
                  style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}
                >
                  <div className="flex flex-col min-w-0 flex-1 gap-0.5">
                    <span className="font-mono truncate" style={{ fontSize: 11, color: 'var(--text-4)' }}>
                      {s.session_id}
                    </span>
                    {s.first_query && s.first_query !== s.session_id && (
                      <span className="truncate" style={{ fontSize: 13, color: 'var(--text-1)' }}>
                        {s.first_query}
                      </span>
                    )}
                    <span style={{ fontSize: 11, color: 'var(--text-4)' }}>
                      {s.updated_at ? new Date(s.updated_at).toLocaleString() : '—'}
                      {' · '}
                      {s.run_count} {t('gw.sessions.runs')}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
