import { useState } from 'react'
import type { RunInstance } from '../../store/runs'
import { Badge } from '../shared/Badge'
import { CostChip } from '../shared/CostChip'
import { ChatPanel } from '../chat/ChatPanel'
import { TraceView } from './TraceView'
import { WaterfallView } from './WaterfallView'
import { StatsPanel } from './StatsPanel'

interface RunCardProps {
  run: RunInstance
  onStop?: () => void
}

const STATUS_BADGE: Record<string, { label: string; variant: 'gray' | 'blue' | 'green' | 'red' }> = {
  idle:    { label: 'Idle',    variant: 'gray' },
  running: { label: 'Running', variant: 'blue' },
  done:    { label: 'Done',    variant: 'green' },
  error:   { label: 'Error',   variant: 'red' },
}

type Tab = 'chat' | 'trace' | 'waterfall' | 'stats'

export function RunCard({ run, onStop }: RunCardProps) {
  const [tab, setTab] = useState<Tab>('chat')
  const { label, variant } = STATUS_BADGE[run.status] ?? STATUS_BADGE.idle
  const totalCost   = run.steps.reduce((s, r) => s + r.cost_usd, 0)
  const hasChildren = Object.keys(run.children).length > 0
  const hasSteps    = run.steps.length > 0 || hasChildren

  // Tabs available: chat always; trace when children; waterfall + stats when steps exist
  const tabs: { id: Tab; label: string }[] = [
    { id: 'chat',      label: 'Chat' },
    ...(hasChildren ? [{ id: 'trace' as Tab, label: 'Trace' }] : []),
    ...(hasSteps    ? [{ id: 'waterfall' as Tab, label: 'Waterfall' }] : []),
    ...(hasSteps    ? [{ id: 'stats' as Tab, label: 'Stats' }] : []),
  ]

  // Keep tab valid when available tabs change (e.g. steps arrive)
  const activeTab = tabs.some((t) => t.id === tab) ? tab : 'chat'

  return (
    <div className="flex flex-col h-full gap-2">
      {/* Header row */}
      <div className="flex items-center gap-2 shrink-0">
        <Badge variant={variant}>{label}</Badge>
        {run.steps.length > 0 && (
          <span className="text-xs text-gray-500">{run.steps.length} step{run.steps.length !== 1 ? 's' : ''}</span>
        )}
        {totalCost > 0 && <CostChip usd={totalCost} />}
        {run.status === 'running' && onStop && (
          <button onClick={onStop} className="btn btn-ghost ml-auto text-xs px-2 py-0.5">
            Stop
          </button>
        )}
      </div>

      {/* Result summary */}
      {run.result && (
        <div className={`text-xs rounded px-2 py-1 shrink-0 ${
          run.result.passed === true  ? 'bg-green-50 text-green-700' :
          run.result.passed === false ? 'bg-red-50 text-red-700' :
          'bg-gray-50 text-gray-600'
        }`}>
          {run.result.passed === true ? '✓ Passed' : run.result.passed === false ? '✗ Failed' : '—'}{' '}
          <span className="text-gray-500">· {run.result.exit_reason}</span>
        </div>
      )}

      {run.error && (
        <div className="text-xs rounded px-2 py-1 bg-red-50 text-red-700 shrink-0">
          {run.error}
        </div>
      )}

      {/* Tab bar — only show when there are extra tabs */}
      {tabs.length > 1 && (
        <div
          className="flex shrink-0"
          style={{ borderBottom: '1px solid var(--border)', gap: 0 }}
        >
          {tabs.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                padding:      '5px 14px',
                fontSize:     12,
                fontFamily:   'JetBrains Mono, monospace',
                cursor:       'pointer',
                background:   'transparent',
                border:       'none',
                borderBottom: activeTab === t.id ? '2px solid var(--accent)' : '2px solid transparent',
                color:        activeTab === t.id ? 'var(--accent)' : 'var(--text-3)',
                marginBottom: -1,
                transition:   'color 0.15s, border-color 0.15s',
              }}
            >
              {t.label}
            </button>
          ))}
        </div>
      )}

      {/* Tab content */}
      {activeTab === 'chat' && (
        <ChatPanel messages={run.messages} autoScroll={run.status === 'running'} />
      )}
      {activeTab === 'trace' && (
        <TraceView run={run} />
      )}
      {activeTab === 'waterfall' && (
        <WaterfallView run={run} />
      )}
      {activeTab === 'stats' && (
        <StatsPanel run={run} />
      )}
    </div>
  )
}
