import { RotateCcw, X } from 'lucide-react'
import type { RunInstance } from '../../store/runs'
import type { CustomHarness, HarnessConfig } from '../../api/types'
import { useRunsStore } from '../../store/runs'
import { Badge } from '../shared/Badge'
import { CostChip } from '../shared/CostChip'
import { ChatPanel } from '../chat/ChatPanel'
import { workspaceFromHarnessName } from '../../lib/labWorkspace'
import { useT } from '../../i18n'

type ProcessorList = Record<string, unknown>[]

interface CompareColumnProps {
  run:      RunInstance
  customHarnesses: CustomHarness[]
  onRemove: () => void
}

const STATUS_BADGE: Record<string, { labelKey: string; variant: 'gray' | 'blue' | 'green' | 'red' }> = {
  idle:    { labelKey: 'status.idle',    variant: 'gray' },
  running: { labelKey: 'status.running', variant: 'blue' },
  done:    { labelKey: 'status.done',    variant: 'green' },
  error:   { labelKey: 'status.error',   variant: 'red' },
}

/** Stable stringify with sorted object keys so config matching ignores key insertion order. */
function stableStringify(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map((v) => stableStringify(v)).join(',')}]`
  }
  if (value && typeof value === 'object') {
    const obj = value as Record<string, unknown>
    const entries = Object.entries(obj).sort(([a], [b]) => a.localeCompare(b))
    return `{${entries.map(([k, v]) => `${JSON.stringify(k)}:${stableStringify(v)}`).join(',')}}`
  }
  return JSON.stringify(value)
}

function processorSetSignature(processors: ProcessorList): string {
  return processors.map((p) => stableStringify(p)).sort().join('|')
}

function cloneHarnessConfig(cfg: HarnessConfig): HarnessConfig {
  if (typeof structuredClone === 'function') return structuredClone(cfg)
  return JSON.parse(JSON.stringify(cfg)) as HarnessConfig
}

export function CompareColumn({ run, customHarnesses, onRemove }: CompareColumnProps) {
  const t = useT()
  const { stopRun, updateColumn, resetSession } = useRunsStore()
  const { labelKey, variant } = STATUS_BADGE[run.status] ?? STATUS_BADGE.idle
  const totalCost = run.steps.reduce((s, r) => s + r.cost_usd, 0)
  const hasSession = run.messages.length > 0 || !!run.session_id || run.isReadOnly

  function applyCustomHarness(harnessId: string) {
    const ch = customHarnesses.find((c) => c.id === harnessId)
    if (!ch) return
    const fallbackWs = workspaceFromHarnessName(ch.name)
    const ws = {
      agentId: (ch.workspace?.agent_id || '').trim() || fallbackWs.agentId,
      project: (ch.workspace?.project || '').trim() || fallbackWs.project,
    }
    updateColumn(run.id, {
      harnessConfig: cloneHarnessConfig(ch.harness_config),
      harnessName: ch.name,
      workspaceAgentId: ws.agentId,
      workspaceProject: ws.project,
    })
  }

  const runSignature = processorSetSignature(run.harnessConfig.processors)
  const currentHarnessId = customHarnesses.find((c) => {
    if (c.name === run.harnessName) return true
    const sig = processorSetSignature(c.harness_config.processors)
    return sig === runSignature
  })?.id ?? ''

  return (
    <div
      className="relative flex flex-col rounded-xl overflow-hidden min-h-0 h-full transition-all duration-150"
      style={{
        background: 'var(--bg-card)',
        border: '1px solid var(--border)',
        boxShadow: 'var(--shadow-card)',
      }}
    >
      <div
        className="flex items-center gap-1.5 px-3 py-2 shrink-0"
        style={{ borderBottom: '1px solid var(--border)' }}
      >
        <select
          value={currentHarnessId}
          onChange={(e) => applyCustomHarness(e.target.value)}
          disabled={run.status === 'running' || customHarnesses.length === 0}
          className="text-xs font-medium flex-1 min-w-0 bg-transparent border-0 focus:outline-none cursor-pointer"
          style={{ color: 'var(--text-1)' }}
        >
          {customHarnesses.length === 0 ? (
            <option value="">{t('compare.no_custom_harness')}</option>
          ) : (
            customHarnesses.map((ch) => (
              <option key={ch.id} value={ch.id}>{ch.name}</option>
            ))
          )}
        </select>

        <div className="flex items-center gap-1 shrink-0">
          <Badge variant={variant} size="xs">{t(labelKey)}</Badge>
          {totalCost > 0 && <CostChip usd={totalCost} />}
        </div>

        {run.status === 'running' ? (
          <button
            onClick={() => stopRun(run.id)}
            className="text-xs px-1.5 py-0.5 rounded transition-colors"
            style={{ color: '#ef4444', background: 'rgba(239,68,68,0.08)' }}
          >
            {t('status.stop')}
          </button>
        ) : hasSession ? (
          <button
            onClick={() => resetSession(run.id)}
            className="flex items-center gap-1 text-xs px-2 py-0.5 rounded border transition-colors"
            style={{
              color: 'var(--text-3)',
              borderColor: 'var(--border)',
              background: 'transparent',
            }}
            title={t('status.new_chat')}
          >
            <RotateCcw size={11} />
            {t('status.new_chat')}
          </button>
        ) : null}

        <button
          onClick={onRemove}
          className="transition-colors ml-0.5"
          style={{ color: 'var(--text-4)' }}
          onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--text-2)')}
          onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-4)')}
        >
          <X size={13} />
        </button>
      </div>

      {run.result && (
        <div
          className="text-xs px-3 py-1 shrink-0 font-mono"
          style={{
            background: run.result.passed === true
              ? 'rgba(16,185,129,0.08)'
              : run.result.passed === false
              ? 'rgba(239,68,68,0.08)'
              : 'var(--bg-elevated)',
            borderBottom: '1px solid var(--border)',
            color: run.result.passed === true ? '#10b981'
              : run.result.passed === false ? '#ef4444'
              : 'var(--text-3)',
          }}
        >
          {run.result.passed === true ? t('status.passed')
            : run.result.passed === false ? t('status.failed')
            : '— '}
          {' · '}{run.result.exit_reason}{' · '}{run.result.steps} steps
        </div>
      )}

      {run.error && (
        <div
          className="text-xs px-3 py-1 shrink-0"
          style={{
            background: 'rgba(239,68,68,0.08)',
            color: '#ef4444',
            borderBottom: '1px solid rgba(239,68,68,0.2)',
          }}
        >
          {run.error}
        </div>
      )}

      <div className="flex-1 min-h-0">
        <ChatPanel messages={run.messages} autoScroll={run.status === 'running'} />
      </div>
    </div>
  )
}
