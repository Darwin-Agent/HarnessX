import { useEffect, useCallback } from 'react'
import { Plus, RotateCcw, Settings } from 'lucide-react'
import { useLabStore } from '../store/lab'
import { useRunsStore } from '../store/runs'
import { useSlotsStore } from '../store/slots'
import { useUIStore } from '../store/ui'
import { CompareColumn } from '../components/compare/CompareColumn'
import { ChatInput } from '../components/chat/ChatInput'
import { Badge } from '../components/shared/Badge'
import { CostChip } from '../components/shared/CostChip'
import type { Attachment, TaskBlock } from '../api/types'
import { useT } from '../i18n'

const MAX_COLUMNS = 4

const STATUS_BADGE: Record<string, { labelKey: string; variant: 'gray' | 'blue' | 'green' | 'red' }> = {
  idle:    { labelKey: 'status.idle',    variant: 'gray' },
  running: { labelKey: 'status.running', variant: 'blue' },
  done:    { labelKey: 'status.done',    variant: 'green' },
  error:   { labelKey: 'status.error',   variant: 'red' },
}

export function ComparePage() {
  const t = useT()
  const { successCriteria, customHarnesses } = useLabStore()
  const { columns, addColumn, removeColumn, startRun, resetSession } = useRunsStore()
  const { toSlotConfig, toModelConfigPayload, validateLaunch } = useSlotsStore()
  const setSettingsOpen = useUIStore((s) => s.setSettingsOpen)

  const noCustomHarness = customHarnesses.length === 0

  useEffect(() => {
    if (noCustomHarness) return
    const current = useRunsStore.getState().columns
    const needed = Math.max(0, Math.min(2, customHarnesses.length) - current.length)
    for (let i = 0; i < needed; i++) {
      const ch = customHarnesses[current.length + i] ?? customHarnesses[0]
      if (!ch) break
      addColumn(ch.harness_config, ch.name, ch.workspace)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [customHarnesses, noCustomHarness, addColumn])

  const anyRunning = columns.some((c) => c.status === 'running')
  const hasAnySession = columns.some((c) =>
    c.messages.length > 0 || !!c.session_id || c.isReadOnly || !!c.result || !!c.error,
  )

  const handleSend = useCallback((text: string, attachments: Attachment[]) => {
    const modelError = validateLaunch()
    if (modelError) return
    const slotConfig     = toSlotConfig()
    const providerConfig = toModelConfigPayload()
    const task: string | TaskBlock[] = attachments.length > 0
      ? [
          ...(text ? [{ type: 'text' as const, text }] : []),
          ...attachments.map((a) => ({
            type:   'image' as const,
            source: { type: 'base64' as const, media_type: a.media_type, data: a.data },
          })),
        ]
      : text
    columns.forEach((col) => {
      if (col.status !== 'running') {
        startRun(col.id, task, successCriteria, providerConfig, slotConfig, {
          harnessConfig: col.harnessConfig,
          harnessName: col.harnessName,
          agentId: col.workspaceAgentId,
          project: col.workspaceProject,
        })
      }
    })
  }, [columns, successCriteria, startRun, toSlotConfig, toModelConfigPayload, validateLaunch])

  function handleAddColumn() {
    if (columns.length >= MAX_COLUMNS || noCustomHarness) return
    const ch = customHarnesses[columns.length % customHarnesses.length]
    if (!ch) return
    addColumn(ch.harness_config, ch.name, ch.workspace)
  }

  const handleResetAllSessions = useCallback(() => {
    columns.forEach((col) => resetSession(col.id))
  }, [columns, resetSession])

  return (
    <div className="flex flex-col flex-1 min-h-0 overflow-hidden">
      <div
        className="flex items-center gap-2 px-4 py-2.5 shrink-0"
        style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-card)' }}
      >
        <span className="font-mono" style={{ fontSize: '12px', color: 'var(--text-3)' }}>
          {columns.length} {t('compare.columns_of')} {MAX_COLUMNS}
        </span>

        {columns.length < MAX_COLUMNS && (
          <button
            onClick={handleAddColumn}
            disabled={noCustomHarness}
            className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg transition-all duration-150 disabled:opacity-35 disabled:cursor-not-allowed"
            style={{ fontSize: '13px', color: 'var(--text-2)' }}
            onMouseEnter={(e) => {
              if (e.currentTarget.disabled) return
              e.currentTarget.style.color = 'var(--accent)'
              e.currentTarget.style.background = 'var(--accent-bg)'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = 'var(--text-2)'
              e.currentTarget.style.background = 'transparent'
            }}
            title={noCustomHarness ? t('compare.no_custom_harness') : undefined}
          >
            <Plus size={13} />
            {t('compare.add_column')}
          </button>
        )}

        <button
          onClick={handleResetAllSessions}
          disabled={anyRunning || !hasAnySession}
          className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg transition-all duration-150 disabled:opacity-35 disabled:cursor-not-allowed"
          style={{ fontSize: '13px', color: 'var(--text-2)' }}
          onMouseEnter={(e) => {
            if (e.currentTarget.disabled) return
            e.currentTarget.style.color = 'var(--accent)'
            e.currentTarget.style.background = 'var(--accent-bg)'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.color = 'var(--text-2)'
            e.currentTarget.style.background = 'transparent'
          }}
          title={t('status.new_chat')}
        >
          <RotateCcw size={13} />
          {t('status.new_chat')}
        </button>

        <div className="ml-auto">
          <button
            onClick={() => setSettingsOpen(true)}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg transition-all duration-150"
            style={{ fontSize: '13px', color: 'var(--text-2)', border: '1px solid transparent' }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = 'var(--accent)'
              e.currentTarget.style.background = 'var(--accent-bg)'
              e.currentTarget.style.borderColor = 'var(--accent-ring)'
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = 'var(--text-2)'
              e.currentTarget.style.background = 'transparent'
              e.currentTarget.style.borderColor = 'transparent'
            }}
          >
            <Settings size={13} />
            <span className="hidden sm:inline">{t('compare.config')}</span>
          </button>
        </div>
      </div>

      {columns.length > 0 && (
        <div
          className="shrink-0 px-3 py-1.5 gap-2.5"
          style={{
            display: 'grid',
            gridTemplateColumns: `repeat(${columns.length}, minmax(0, 1fr))`,
            borderBottom: '1px solid var(--border)',
            background: 'var(--bg-elevated)',
          }}
        >
          {columns.map((col) => {
            const sb = STATUS_BADGE[col.status] ?? STATUS_BADGE.idle
            const cost = col.steps.reduce((s, r) => s + r.cost_usd, 0)
            const steps = col.steps.length
            const result = col.result
            return (
              <div key={col.id} className="flex items-center gap-1.5 min-w-0">
                <Badge variant={sb.variant} size="xs">{t(sb.labelKey)}</Badge>
                {cost > 0 && <CostChip usd={cost} />}
                {steps > 0 && (
                  <span className="font-mono" style={{ fontSize: '10px', color: 'var(--text-4)' }}>
                    {steps}s
                  </span>
                )}
                {result && (
                  <span
                    className="font-mono ml-auto"
                    style={{
                      fontSize: '10px',
                      color: result.passed === true ? '#10b981' : result.passed === false ? '#ef4444' : 'var(--text-4)',
                    }}
                  >
                    {result.passed === true ? '✓' : result.passed === false ? '✗' : '—'}
                    {result.exit_reason ? ` ${result.exit_reason}` : ''}
                  </span>
                )}
              </div>
            )
          })}
        </div>
      )}

      <div
        className="flex-1 min-h-0 overflow-hidden p-3 gap-2.5"
        style={{ display: 'grid', gridTemplateColumns: `repeat(${columns.length || 1}, minmax(0, 1fr))` }}
      >
        {columns.map((col) => (
          <CompareColumn
            key={col.id}
            run={col}
            customHarnesses={customHarnesses}
            onRemove={() => removeColumn(col.id)}
          />
        ))}

        {columns.length === 0 && (
          <div
            className="flex items-center justify-center text-sm col-span-full text-center px-6"
            style={{ color: 'var(--text-4)' }}
          >
            {noCustomHarness ? t('compare.no_custom_harness_hint') : t('compare.empty')}
          </div>
        )}
      </div>

      <ChatInput
        onSend={handleSend}
        disabled={anyRunning || columns.length === 0 || noCustomHarness}
        placeholder={noCustomHarness ? t('compare.no_custom_harness') : t('chat.compare_placeholder')}
      />
    </div>
  )
}
