import { useState } from 'react'
import { ChevronDown, ChevronRight, Bot } from 'lucide-react'
import type { RunInstance, ChildRun } from '../../store/runs'
import { CostChip } from '../shared/CostChip'
import { ChatPanel } from '../chat/ChatPanel'

interface Props {
  run: RunInstance
}

const STATUS_DOT: Record<string, string> = {
  running: '#3b82f6',
  done:    '#22c55e',
  error:   '#ef4444',
}

function ChildSection({ child }: { child: ChildRun }) {
  const [open, setOpen] = useState(true)
  const cost = child.steps.reduce((s, r) => s + r.cost_usd, 0)
  const totalIn  = child.steps.reduce((s, r) => s + r.input_tokens,  0)
  const totalOut = child.steps.reduce((s, r) => s + r.output_tokens, 0)
  const dot = STATUS_DOT[child.status] ?? STATUS_DOT.done

  return (
    <div
      className="rounded-xl overflow-hidden"
      style={{ border: '1px solid var(--border)', marginBottom: '10px' }}
    >
      {/* Header */}
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2.5 px-3.5 py-2.5 text-left"
        style={{ background: 'var(--bg-elevated)', borderBottom: open ? '1px solid var(--border)' : 'none' }}
      >
        <span style={{ color: 'var(--text-4)', fontSize: '12px' }}>
          {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        </span>
        <Bot size={13} style={{ color: 'var(--accent)', flexShrink: 0 }} />
        <span
          className="font-mono"
          style={{ fontSize: '11px', color: 'var(--text-3)', letterSpacing: '0.04em' }}
        >
          {child.run_id.slice(0, 8)}
        </span>
        <span
          className="flex-1 truncate"
          style={{ fontSize: '12px', color: 'var(--text-2)' }}
        >
          {child.task.length > 80 ? child.task.slice(0, 80) + '…' : child.task}
        </span>
        <div className="flex items-center gap-2 shrink-0">
          {(totalIn + totalOut) > 0 && (
            <span className="font-mono" style={{ fontSize: '11px', color: 'var(--text-4)' }}>
              {((totalIn + totalOut) / 1000).toFixed(1)}k tok
            </span>
          )}
          {cost > 0 && <CostChip usd={cost} />}
          <span
            className="w-2 h-2 rounded-full"
            style={{ background: dot, flexShrink: 0 }}
          />
        </div>
      </button>

      {/* Chat */}
      {open && (
        <div style={{ height: '300px', display: 'flex', flexDirection: 'column' }}>
          <ChatPanel messages={child.messages} autoScroll={child.status === 'running'} />
        </div>
      )}
    </div>
  )
}

export function TraceView({ run }: Props) {
  const rootCost  = run.steps.reduce((s, r) => s + r.cost_usd, 0)
  const rootIn    = run.steps.reduce((s, r) => s + r.input_tokens,  0)
  const rootOut   = run.steps.reduce((s, r) => s + r.output_tokens, 0)
  const children  = Object.values(run.children)

  return (
    <div className="flex-1 overflow-y-auto px-3 py-3 min-h-0">
      {/* Root agent */}
      <div
        className="rounded-xl overflow-hidden mb-3"
        style={{ border: '1px solid var(--accent-ring)' }}
      >
        <div
          className="flex items-center gap-2.5 px-3.5 py-2.5"
          style={{ background: 'var(--accent-bg)', borderBottom: '1px solid var(--accent-ring)' }}
        >
          <Bot size={13} style={{ color: 'var(--accent)', flexShrink: 0 }} />
          <span
            className="font-mono"
            style={{ fontSize: '11px', color: 'var(--accent)', letterSpacing: '0.04em', fontWeight: 600 }}
          >
            {run.run_id?.slice(0, 8) ?? 'root'}
          </span>
          <span style={{ fontSize: '12px', color: 'var(--text-3)' }}>root agent</span>
          <div className="flex items-center gap-2 ml-auto shrink-0">
            {(rootIn + rootOut) > 0 && (
              <span className="font-mono" style={{ fontSize: '11px', color: 'var(--text-4)' }}>
                {((rootIn + rootOut) / 1000).toFixed(1)}k tok
              </span>
            )}
            {rootCost > 0 && <CostChip usd={rootCost} />}
            <span
              className="w-2 h-2 rounded-full"
              style={{ background: STATUS_DOT[run.status] ?? STATUS_DOT.done }}
            />
          </div>
        </div>
        <div style={{ height: '320px', display: 'flex', flexDirection: 'column' }}>
          <ChatPanel messages={run.messages} autoScroll={run.status === 'running'} />
        </div>
      </div>

      {/* Children */}
      {children.length === 0 && (
        <div
          className="flex items-center justify-center py-6 text-xs font-mono"
          style={{ color: 'var(--text-4)' }}
        >
          no sub-agents spawned
        </div>
      )}
      {children.map((child) => (
        <ChildSection key={child.run_id} child={child} />
      ))}
    </div>
  )
}
