import { useState } from 'react'
import type { QueryContext, ProcessorTrigger } from '../../api/types'

interface Props {
  ctx: QueryContext
}

const MONO: React.CSSProperties = {
  fontFamily: 'JetBrains Mono, monospace',
  fontSize: '11px',
}

const PROC_BG = 'rgba(234,179,8,0.08)'

function TriggerRow({ trigger }: { trigger: ProcessorTrigger }) {
  const { processor, hook, action, detail } = trigger
  const detailPairs = Object.entries(detail ?? {}).slice(0, 6)
  return (
    <div
      className="rounded px-2 py-1 my-0.5"
      style={{ background: PROC_BG, border: '1px solid rgba(234,179,8,0.18)' }}
    >
      <div className="flex items-baseline gap-1.5" style={MONO}>
        <span style={{ color: '#ca8a04' }}>⚡</span>
        <span style={{ color: '#b45309', opacity: 0.9 }}>[{hook}]</span>
        <span style={{ color: 'var(--text-2)', fontWeight: 600 }}>{processor}</span>
        <span style={{ color: 'var(--text-3)' }}>·</span>
        <span style={{ color: 'var(--text-3)' }}>{action}</span>
      </div>
      {detailPairs.length > 0 && (
        <div className="mt-0.5 pl-4" style={{ ...MONO, color: 'var(--text-4)' }}>
          {detailPairs.map(([k, v]) => (
            <span key={k} className="mr-3">
              {k}: {typeof v === 'object' ? JSON.stringify(v) : String(v)}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

export function QueryContextPanel({ ctx }: Props) {
  const [systemOpen, setSystemOpen] = useState(false)

  return (
    <div
      className="mt-2 rounded-xl px-3.5 py-3 overflow-x-auto"
      style={{ background: 'var(--bg-base)', border: '1px solid var(--border)' }}
    >
      {/* Tools */}
      {ctx.tool_names.length > 0 && (
        <div className="mb-2.5">
          <div style={{ ...MONO, color: 'var(--text-3)', fontWeight: 700, marginBottom: '4px' }}>
            tools ({ctx.tool_names.length})
          </div>
          <div className="flex flex-wrap gap-1 pl-1">
            {ctx.tool_names.map((name) => (
              <span
                key={name}
                className="rounded px-1.5 py-0.5"
                style={{ ...MONO, background: 'rgba(0,153,192,0.08)', color: '#0099c0', border: '1px solid rgba(0,153,192,0.18)' }}
              >
                {name}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* System prompt */}
      {ctx.system && (
        <div className="mb-2.5">
          <button
            onClick={() => setSystemOpen((v) => !v)}
            className="flex items-center gap-1.5"
            style={{ ...MONO, color: 'var(--text-3)', background: 'transparent', padding: 0, cursor: 'pointer', fontWeight: 700 }}
          >
            <span style={{ opacity: 0.6, fontSize: '10px' }}>{systemOpen ? '▾' : '▸'}</span>
            system prompt
          </button>
          {systemOpen && (
            <pre
              className="mt-1.5 px-2.5 py-2 rounded whitespace-pre-wrap break-words overflow-auto leading-relaxed"
              style={{
                ...MONO,
                color: 'var(--text-3)',
                background: 'var(--bg-elevated)',
                border: '1px solid var(--border)',
                maxHeight: '260px',
              }}
            >
              {ctx.system}
            </pre>
          )}
        </div>
      )}

      {/* on_task_start processor triggers */}
      {ctx.on_task_start_triggers.length > 0 && (
        <div>
          <div style={{ ...MONO, color: 'var(--text-3)', fontWeight: 700, marginBottom: '4px' }}>
            on_task_start
          </div>
          <div className="pl-1">
            {ctx.on_task_start_triggers.map((t, i) => (
              <TriggerRow key={i} trigger={t} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
