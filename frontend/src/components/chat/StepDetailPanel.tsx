import type { StepTrace, TimelineItem } from '../../api/types'

interface Props {
  traces: StepTrace[]
}

const PROC_BG = 'rgba(234,179,8,0.08)'   // yellow-500/8
const MONO: React.CSSProperties = {
  fontFamily: 'JetBrains Mono, monospace',
  fontSize: '11px',
}

function fmt(n: number): string {
  return n.toLocaleString()
}

function fmtDur(ms: number): string {
  return ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`
}

function fmtCost(usd: number): string {
  if (usd === 0) return ''
  if (usd < 0.001) return `$${(usd * 1000).toFixed(3)}m`
  return `$${usd.toFixed(4)}`
}

function isPreHook(hook: string): boolean {
  return (
    hook === 'step_start'
    || hook === 'on_step_start'
    || hook === 'before_model'
    || hook === 'on_before_model'
    || hook === 'on_model_start'
  )
}

function TimelineRow({ item }: { item: TimelineItem }) {
  if (item.kind === 'processor') {
    const { processor, hook, action, detail } = item.trigger
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

  const { block } = item

  if (block.type === 'thinking') {
    const est = Math.ceil(block.content.length / 4)
    return (
      <div className="flex items-baseline gap-1.5 py-0.5" style={{ ...MONO, color: 'var(--text-4)' }}>
        <span>💭</span>
        <span>thinking</span>
        <span>·</span>
        <span>~{fmt(est)} tokens</span>
      </div>
    )
  }

  if (block.type === 'text') {
    const preview = block.content.slice(0, 140).replace(/\n/g, ' ')
    return (
      <div className="flex items-baseline gap-1.5 py-0.5" style={MONO}>
        <span style={{ color: 'var(--text-4)' }}>📝</span>
        <span style={{ color: 'var(--text-3)' }}>
          "{preview}{block.content.length > 140 ? '…' : ''}"
        </span>
      </div>
    )
  }

  if (block.type === 'tool_use') {
    return (
      <div className="flex items-baseline gap-1.5 py-0.5" style={MONO}>
        <span style={{ color: '#0099c0', opacity: 0.7 }}>🔧</span>
        <span style={{ color: 'var(--text-3)' }}>tool_use:</span>
        <span style={{ color: 'var(--text-2)', fontWeight: 600 }}>{block.name}</span>
      </div>
    )
  }

  if (block.type === 'tool_result') {
    const hasError = !!block.error
    const icon = hasError ? '✗' : '✓'
    const accent = hasError ? '#ef4444' : '#059669'
    return (
      <div className="flex items-baseline gap-1.5 py-0.5" style={MONO}>
        <span style={{ color: accent }}>{icon}</span>
        <span style={{ color: 'var(--text-3)' }}>tool_result:</span>
        <span style={{ color: 'var(--text-2)', fontWeight: 600 }}>{block.name}</span>
        {block.duration_ms > 0 && (
          <>
            <span style={{ color: 'var(--text-4)' }}>·</span>
            <span style={{ color: 'var(--text-4)' }}>{fmtDur(block.duration_ms)}</span>
          </>
        )}
      </div>
    )
  }

  return null
}

function StepSection({ trace }: { trace: StepTrace }) {
  const cost = fmtCost(trace.cost_usd)
  const inp = trace.input
  const hasInputMeta = !!inp && (inp.tool_names.length > 0 || inp.message_count > 0)
  const outputTimeline = trace.timeline.filter((item) => item.kind !== 'processor' || !isPreHook(item.trigger.hook))
  return (
    <div className="mb-3 last:mb-0">
      {/* Step header */}
      <div
        className="flex items-center gap-2 mb-1.5 pb-1"
        style={{ ...MONO, color: 'var(--text-3)', borderBottom: '1px solid var(--border)' }}
      >
        <span style={{ color: 'var(--text-2)', fontWeight: 700 }}>Step {trace.step}</span>
        {trace.model && (
          <>
            <span style={{ opacity: 0.4 }}>·</span>
            <span style={{ color: 'var(--text-3)' }}>{trace.model}</span>
          </>
        )}
        <span style={{ opacity: 0.4 }}>·</span>
        <span>↑{fmt(trace.input_tokens)}</span>
        <span>↓{fmt(trace.output_tokens)}</span>
        <span style={{ opacity: 0.4 }}>·</span>
        <span>{fmtDur(trace.duration_ms)}</span>
        {cost && (
          <>
            <span style={{ opacity: 0.4 }}>·</span>
            <span style={{ color: 'var(--text-4)' }}>{cost}</span>
          </>
        )}
      </div>

      {/* Input metadata (counts only). Pre-step triggers are shown above reply bubble, not repeated here. */}
      {hasInputMeta && (
        <div className="mb-2 pl-1">
          <div style={{ ...MONO, color: 'var(--text-4)', marginBottom: '3px', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            input
            {inp!.message_count > 0 && <span className="ml-2">msgs:{inp!.message_count}</span>}
            {inp!.tool_names.length > 0 && <span className="ml-2">tools:{inp!.tool_names.length}</span>}
          </div>
        </div>
      )}

      {/* Output section: timeline */}
      {hasInputMeta && (
        <div style={{ ...MONO, color: 'var(--text-4)', marginBottom: '3px', marginLeft: '4px', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
          output
        </div>
      )}
      <div className="pl-1">
        {outputTimeline.length === 0 ? (
          <span style={{ ...MONO, color: 'var(--text-4)' }}>(no output events)</span>
        ) : (
          outputTimeline.map((item, i) => <TimelineRow key={i} item={item} />)
        )}
      </div>
    </div>
  )
}

export function StepDetailPanel({ traces }: Props) {
  return (
    <div
      className="mt-2 rounded-xl px-3.5 py-3 overflow-x-auto"
      style={{
        background: 'var(--bg-base)',
        border: '1px solid var(--border)',
      }}
    >
      {traces.map((trace, i) => (
        <StepSection key={i} trace={trace} />
      ))}
    </div>
  )
}
