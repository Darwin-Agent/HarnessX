import type { RunInstance } from '../../store/runs'
import type { MessageBlock } from '../../api/types'

interface Props {
  run: RunInstance
}

function fmt_tok(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return String(n)
}

function fmt_usd(n: number): string {
  if (n >= 0.01) return `$${n.toFixed(4)}`
  return `$${n.toFixed(6)}`
}

// Aggregate tool_result blocks from messages across root + children
interface ToolRow { name: string; calls: number; totalMs: number }

function collectToolStats(run: RunInstance): ToolRow[] {
  const map = new Map<string, ToolRow>()

  const allMessages = [
    ...run.messages,
    ...Object.values(run.children).flatMap((c) => c.messages),
  ]

  for (const msg of allMessages) {
    for (const block of msg.blocks ?? []) {
      if (block.type === 'tool_result') {
        const b = block as Extract<MessageBlock, { type: 'tool_result' }>
        const row = map.get(b.name) ?? { name: b.name, calls: 0, totalMs: 0 }
        row.calls++
        row.totalMs += b.duration_ms
        map.set(b.name, row)
      }
    }
  }

  return [...map.values()].sort((a, b) => b.totalMs - a.totalMs)
}

export function StatsPanel({ run }: Props) {
  // Aggregate across root + all children
  const allSteps = [
    ...run.steps,
    ...Object.values(run.children).flatMap((c) => c.steps),
  ]

  const totalCost = allSteps.reduce((s, r) => s + r.cost_usd, 0)
  const totalIn   = run.result?.total_input_tokens  ?? allSteps.reduce((s, r) => s + r.input_tokens,  0)
  const totalOut  = run.result?.total_output_tokens ?? allSteps.reduce((s, r) => s + r.output_tokens, 0)
  const stepCount = allSteps.length

  const maxTok = Math.max(...allSteps.map((s) => s.input_tokens + s.output_tokens), 1)

  const tools = collectToolStats(run)

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 min-h-0 flex flex-col gap-5">

      {/* ── Summary ─────────────────────────────────────────────────────── */}
      <div
        className="grid gap-3 rounded-xl p-3"
        style={{ gridTemplateColumns: 'repeat(4, 1fr)', background: 'var(--bg-elevated)', border: '1px solid var(--border)' }}
      >
        {[
          { label: 'Total cost',   value: fmt_usd(totalCost) },
          { label: 'Input tokens', value: fmt_tok(totalIn)   },
          { label: 'Output tokens', value: fmt_tok(totalOut)  },
          { label: 'Steps',        value: String(stepCount)  },
        ].map(({ label, value }) => (
          <div key={label} className="flex flex-col gap-0.5 items-center">
            <span style={{ fontSize: 18, fontWeight: 600, color: 'var(--text-1)', fontFamily: 'JetBrains Mono, monospace' }}>
              {value}
            </span>
            <span style={{ fontSize: 11, color: 'var(--text-4)' }}>{label}</span>
          </div>
        ))}
      </div>

      {/* ── Per-step token breakdown ─────────────────────────────────────── */}
      {allSteps.length > 0 && (
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-4)', marginBottom: 8, fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.04em', textTransform: 'uppercase' }}>
            Per-step tokens
          </div>
          <div className="flex flex-col gap-1.5">
            {allSteps.map((step, i) => {
              const total  = step.input_tokens + step.output_tokens
              const inPct  = total > 0 ? (step.input_tokens  / maxTok) * 100 : 0
              const outPct = total > 0 ? (step.output_tokens / maxTok) * 100 : 0
              const isChild = step.run_id !== run.run_id
              return (
                <div key={i} className="flex items-center gap-2">
                  <div
                    className="shrink-0 font-mono text-right"
                    style={{ width: 70, fontSize: 10, color: isChild ? 'var(--text-4)' : 'var(--text-3)' }}
                  >
                    {isChild ? `↳ S${step.step}` : `S${step.step}`}
                  </div>
                  <div className="flex-1 flex gap-0.5" style={{ height: 12 }}>
                    <div style={{ width: `${inPct}%`,  background: '#3b82f6', borderRadius: '2px 0 0 2px', minWidth: inPct  > 0 ? 2 : 0 }} />
                    <div style={{ width: `${outPct}%`, background: '#22c55e', borderRadius: '0 2px 2px 0', minWidth: outPct > 0 ? 2 : 0 }} />
                  </div>
                  <div className="shrink-0 font-mono text-right" style={{ width: 52, fontSize: 10, color: 'var(--text-4)' }}>
                    {fmt_tok(total)}
                  </div>
                  <div className="shrink-0 font-mono text-right" style={{ width: 60, fontSize: 10, color: 'var(--text-4)' }}>
                    {fmt_usd(step.cost_usd)}
                  </div>
                </div>
              )
            })}
          </div>
          {/* Token legend */}
          <div className="flex items-center gap-3 mt-2">
            <div className="flex items-center gap-1.5">
              <div className="w-2.5 h-2 rounded-sm" style={{ background: '#3b82f6' }} />
              <span style={{ fontSize: 10, color: 'var(--text-4)' }}>input</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-2.5 h-2 rounded-sm" style={{ background: '#22c55e' }} />
              <span style={{ fontSize: 10, color: 'var(--text-4)' }}>output</span>
            </div>
          </div>
        </div>
      )}

      {/* ── Tool usage table ─────────────────────────────────────────────── */}
      {tools.length > 0 && (
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-4)', marginBottom: 8, fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.04em', textTransform: 'uppercase' }}>
            Tool calls
          </div>
          <div
            className="rounded-xl overflow-hidden"
            style={{ border: '1px solid var(--border)' }}
          >
            <table className="w-full" style={{ borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr style={{ background: 'var(--bg-elevated)', borderBottom: '1px solid var(--border)' }}>
                  {['Tool', 'Calls', 'Avg', 'Total'].map((h) => (
                    <th
                      key={h}
                      style={{
                        padding: '6px 12px',
                        textAlign: h === 'Tool' ? 'left' : 'right',
                        fontSize: 10,
                        fontWeight: 500,
                        color: 'var(--text-4)',
                        fontFamily: 'JetBrains Mono, monospace',
                        letterSpacing: '0.04em',
                        textTransform: 'uppercase',
                      }}
                    >{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {tools.map((t, i) => (
                  <tr
                    key={t.name}
                    style={{ borderBottom: i < tools.length - 1 ? '1px solid var(--border)' : 'none' }}
                  >
                    <td style={{ padding: '6px 12px', color: 'var(--text-1)', fontFamily: 'JetBrains Mono, monospace', fontSize: 12 }}>
                      {t.name}
                    </td>
                    <td style={{ padding: '6px 12px', textAlign: 'right', color: 'var(--text-3)', fontFamily: 'JetBrains Mono, monospace' }}>
                      {t.calls}
                    </td>
                    <td style={{ padding: '6px 12px', textAlign: 'right', color: 'var(--text-3)', fontFamily: 'JetBrains Mono, monospace' }}>
                      {t.calls > 0 ? (t.totalMs / t.calls < 1000 ? `${Math.round(t.totalMs / t.calls)}ms` : `${(t.totalMs / t.calls / 1000).toFixed(1)}s`) : '—'}
                    </td>
                    <td style={{ padding: '6px 12px', textAlign: 'right', color: 'var(--text-3)', fontFamily: 'JetBrains Mono, monospace' }}>
                      {t.totalMs < 1000 ? `${Math.round(t.totalMs)}ms` : `${(t.totalMs / 1000).toFixed(1)}s`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {allSteps.length === 0 && tools.length === 0 && (
        <div className="flex-1 flex items-center justify-center text-xs font-mono" style={{ color: 'var(--text-4)' }}>
          no data yet
        </div>
      )}
    </div>
  )
}
