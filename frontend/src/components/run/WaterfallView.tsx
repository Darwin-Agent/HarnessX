import { useState } from 'react'
import type { RunInstance, RunStep } from '../../store/runs'

interface Props {
  run: RunInstance
}

interface FlatStep extends RunStep {
  label:      string    // "root" or child task prefix
  isChild:    boolean
  childTask?: string
}

function fmt_ms(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function fmt_tok(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return String(n)
}

interface TooltipProps {
  step: FlatStep
  x: number
  y: number
}

function Tooltip({ step, x, y }: TooltipProps) {
  const modelMs = Math.max(0, step.duration_ms - step.tool_ms)
  return (
    <div
      style={{
        position:    'fixed',
        left:        x + 12,
        top:         y - 10,
        background:  'var(--bg-elevated)',
        border:      '1px solid var(--border)',
        borderRadius: 8,
        padding:     '8px 12px',
        fontSize:    11,
        fontFamily:  'JetBrains Mono, monospace',
        color:       'var(--text-2)',
        pointerEvents: 'none',
        zIndex:      9999,
        minWidth:    160,
        boxShadow:   '0 4px 12px rgba(0,0,0,0.15)',
      }}
    >
      <div style={{ marginBottom: 4, color: 'var(--text-1)', fontWeight: 600 }}>
        Step {step.step}
        {step.isChild && <span style={{ color: 'var(--text-4)', fontWeight: 400 }}> · {step.label}</span>}
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '2px 8px' }}>
        <span style={{ color: 'var(--text-4)' }}>total</span>
        <span>{fmt_ms(step.duration_ms)}</span>
        <span style={{ color: '#3b82f6' }}>model</span>
        <span>{fmt_ms(modelMs)}</span>
        {step.tool_ms > 0 && <>
          <span style={{ color: '#f97316' }}>tools</span>
          <span>{fmt_ms(step.tool_ms)}</span>
        </>}
        <span style={{ color: 'var(--text-4)' }}>in tok</span>
        <span>{fmt_tok(step.input_tokens)}</span>
        <span style={{ color: 'var(--text-4)' }}>out tok</span>
        <span>{fmt_tok(step.output_tokens)}</span>
        {step.cost_usd > 0 && <>
          <span style={{ color: 'var(--text-4)' }}>cost</span>
          <span>${step.cost_usd.toFixed(5)}</span>
        </>}
      </div>
    </div>
  )
}

export function WaterfallView({ run }: Props) {
  const [tooltip, setTooltip] = useState<{ step: FlatStep; x: number; y: number } | null>(null)

  // Collect all steps into a flat sorted list
  const flat: FlatStep[] = []

  for (const s of run.steps) {
    flat.push({ ...s, label: 'root', isChild: false })
  }
  for (const child of Object.values(run.children)) {
    for (const s of child.steps) {
      flat.push({
        ...s,
        label:     child.run_id.slice(0, 8),
        isChild:   true,
        childTask: child.task,
      })
    }
  }

  // Sort by ts_ms (step start time) so waterfall is chronological
  flat.sort((a, b) => a.ts_ms - b.ts_ms)

  if (flat.length === 0) {
    return (
      <div
        className="flex-1 flex items-center justify-center text-xs font-mono"
        style={{ color: 'var(--text-4)' }}
      >
        no steps yet
      </div>
    )
  }

  const maxEnd = Math.max(...flat.map((s) => s.ts_ms + s.duration_ms))
  const scale  = (ms: number) => maxEnd > 0 ? (ms / maxEnd) * 100 : 0

  return (
    <div className="flex-1 overflow-y-auto px-3 py-3 min-h-0">
      {/* Legend */}
      <div className="flex items-center gap-4 mb-3 px-1">
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-2 rounded-sm" style={{ background: '#3b82f6' }} />
          <span style={{ fontSize: 11, color: 'var(--text-4)', fontFamily: 'JetBrains Mono, monospace' }}>model</span>
        </div>
        <div className="flex items-center gap-1.5">
          <div className="w-3 h-2 rounded-sm" style={{ background: '#f97316' }} />
          <span style={{ fontSize: 11, color: 'var(--text-4)', fontFamily: 'JetBrains Mono, monospace' }}>tools</span>
        </div>
        <div className="ml-auto" style={{ fontSize: 11, color: 'var(--text-4)', fontFamily: 'JetBrains Mono, monospace' }}>
          total: {fmt_ms(maxEnd)}
        </div>
      </div>

      {/* Rows */}
      <div className="flex flex-col gap-1">
        {flat.map((step, i) => {
          const barLeft   = scale(step.ts_ms)
          const barWidth  = Math.max(scale(step.duration_ms), 0.4)  // min 0.4% for visibility
          const modelMs   = Math.max(0, step.duration_ms - step.tool_ms)
          const modelPct  = step.duration_ms > 0 ? (modelMs / step.duration_ms) * 100 : 100
          const toolPct   = 100 - modelPct

          return (
            <div key={i} className="flex items-center gap-2" style={{ height: 28 }}>
              {/* Row label */}
              <div
                className="shrink-0 text-right font-mono truncate"
                style={{
                  width: 90,
                  fontSize: 11,
                  color: step.isChild ? 'var(--text-4)' : 'var(--text-3)',
                  paddingLeft: step.isChild ? 12 : 0,
                }}
                title={step.isChild ? step.childTask : undefined}
              >
                {step.isChild ? `↳${step.label}` : 'root'} S{step.step}
              </div>

              {/* Bar track */}
              <div className="flex-1 relative" style={{ height: 16 }}>
                <div
                  className="absolute top-0 rounded"
                  style={{
                    left:   `${barLeft}%`,
                    width:  `${barWidth}%`,
                    height: '100%',
                    display: 'flex',
                    overflow: 'hidden',
                    cursor: 'pointer',
                    borderRadius: 4,
                  }}
                  onMouseEnter={(e) => setTooltip({ step, x: e.clientX, y: e.clientY })}
                  onMouseMove={(e)  => setTooltip({ step, x: e.clientX, y: e.clientY })}
                  onMouseLeave={()  => setTooltip(null)}
                >
                  {/* Model segment */}
                  <div style={{ width: `${modelPct}%`, background: step.isChild ? '#6366f1' : '#3b82f6', minWidth: modelPct > 0 ? 2 : 0 }} />
                  {/* Tool segment */}
                  {toolPct > 0 && (
                    <div style={{ width: `${toolPct}%`, background: '#f97316', minWidth: 2 }} />
                  )}
                </div>
              </div>

              {/* Duration label */}
              <div
                className="shrink-0 font-mono text-right"
                style={{ width: 48, fontSize: 11, color: 'var(--text-4)' }}
              >
                {fmt_ms(step.duration_ms)}
              </div>
            </div>
          )
        })}
      </div>

      {tooltip && <Tooltip step={tooltip.step} x={tooltip.x} y={tooltip.y} />}
    </div>
  )
}
