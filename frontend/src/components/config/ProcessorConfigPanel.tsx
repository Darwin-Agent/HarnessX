import { useEffect } from 'react'
import { X } from 'lucide-react'
import type { HarnessConfig } from '../../api/types'
import { useT } from '../../i18n'

interface Props {
  open:         boolean
  onClose:      () => void
  harnessConfig: HarnessConfig
  onChange:     (c: HarnessConfig) => void
}

type ProcessorList = Record<string, unknown>[]

// ── Known processor _target_ strings ─────────────────────────────────────────

const T = {
  // Context
  SYSTEM_PROMPT:    'harnessx.processors.context.system_prompt.SystemPromptProcessor',
  USER_WRAPPER:     'harnessx.processors.context.user_wrapper.UserWrapperProcessor',
  ENV_INJECTION:    'harnessx.processors.context.env_context_injector.EnvironmentContextInjector',
  COMPACTION:       'harnessx.processors.control.compaction.CompactionProcessor',
  TOKEN_BUDGET:     'harnessx.processors.control.token_budget.TokenBudgetProcessor',
  END_NUDGE:        'harnessx.processors.control.end_nudge.EndNudgeProcessor',
  // Memory
  MEM_EXTRACTION:   'harnessx.processors.memory.memory_extraction.MemoryExtractionProcessor',
  MEM_RETRIEVAL:    'harnessx.processors.memory.memory_retrieval.MemoryRetrievalProcessor',
  SLIDING_WIN:      'harnessx.processors.memory.strategies.sliding_window.SlidingWindowMemory',
  // Control
  LOOP_DETECT:      'harnessx.processors.control.loop_detection.LoopDetectionProcessor',
  TOOL_CORRECT:     'harnessx.processors.control.tool_call_correction.ToolCallCorrectionLayer',
  PARSE_RETRY:      'harnessx.processors.control.parse_retry.ParseRetryProcessor',
  TOOL_FAIL_GUARD:  'harnessx.processors.control.tool_failure_guard.ToolFailureGuard',
  REPEATED_EDIT:    'harnessx.processors.control.repeated_edit_detector.RepeatedEditDetector',
  BG_GUARD:         'harnessx.processors.control.bg_install_guard.BackgroundInstallGuard',
  SKILL_LOAD:       'harnessx.processors.context.progressive_skill_loader.ProgressiveSkillLoader',
  COST_GUARD:       'harnessx.processors.control.cost_guard.CostGuardProcessor',
  // Observability
  OBS_JSONL:        'harnessx.processors.observability.jsonl_obs.JsonlObsProcessor',
  OBS_OTEL:         'harnessx.processors.observability.otel_proc.OTelProcessor',
  OBS_CHECKPOINT:   'harnessx.processors.observability.checkpoint.CheckpointProcessor',
  // Evaluation
  EVALUATION:       'harnessx.processors.evaluation.evaluation.EvaluationProcessor',
}

// ── Processor list helpers ────────────────────────────────────────────────────

function hasProc(processors: ProcessorList, target: string): boolean {
  return processors.some((p) => p._target_ === target)
}

function addProc(processors: ProcessorList, entry: Record<string, unknown>): ProcessorList {
  if (hasProc(processors, entry._target_ as string)) return processors
  return [...processors, entry]
}

function removeProc(processors: ProcessorList, target: string): ProcessorList {
  return processors.filter((p) => p._target_ !== target)
}

function getProcProp(processors: ProcessorList, target: string, path: string): unknown {
  const proc = processors.find((p) => p._target_ === target)
  if (!proc) return undefined
  const parts = path.split('.')
  let val: unknown = proc
  for (const part of parts) {
    val = (val as Record<string, unknown>)?.[part]
  }
  return val
}

function setProcProp(processors: ProcessorList, target: string, path: string, value: unknown): ProcessorList {
  return processors.map((proc) => {
    if (proc._target_ !== target) return proc
    const parts = path.split('.')
    const updated = { ...proc }
    if (parts.length === 1) {
      updated[parts[0]] = value
    } else {
      updated[parts[0]] = { ...(updated[parts[0]] as Record<string, unknown> ?? {}), [parts[1]]: value }
    }
    return updated
  })
}

// ── Re-usable micro widgets ───────────────────────────────────────────────────

function Toggle({
  label, checked, disabled, hint,
  onChange,
}: {
  label:    string
  checked:  boolean
  disabled?: boolean
  hint?:    string
  onChange: (v: boolean) => void
}) {
  return (
    <label
      className="flex items-start gap-2"
      style={{ opacity: disabled ? 0.6 : 1, cursor: disabled ? 'default' : 'pointer' }}
    >
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-0.5"
      />
      <span>
        <span className="text-xs font-medium" style={{ color: 'var(--text-1)' }}>{label}</span>
        {hint && <span className="text-xs block mt-0" style={{ color: 'var(--text-4)' }}>{hint}</span>}
      </span>
    </label>
  )
}

function NumberField({
  label, value, min, max, step, placeholder,
  onChange,
}: {
  label:        string
  value:        number | null
  min?:         number
  max?:         number
  step?:        number
  placeholder?: string
  onChange:     (v: number | null) => void
}) {
  return (
    <div>
      <label className="block mb-1 label-mono">{label}</label>
      <input
        type="number"
        value={value ?? ''}
        min={min}
        max={max}
        step={step ?? 1}
        placeholder={placeholder}
        onChange={(e) => {
          const v = e.target.value === '' ? null : Number(e.target.value)
          onChange(v)
        }}
        className="w-full text-xs rounded px-2 py-1.5"
      />
    </div>
  )
}

function DimSection({
  title, accent, children,
}: {
  title:    string
  accent:   string
  children: React.ReactNode
}) {
  return (
    <div className="mb-0">
      <div
        className="flex items-center gap-2 px-4 py-2 sticky top-0 z-10"
        style={{
          background:   'var(--bg-card)',
          borderBottom: '1px solid var(--border)',
          borderLeft:   `3px solid ${accent}`,
        }}
      >
        <span className="text-xs font-semibold tracking-wider uppercase" style={{ color: accent }}>
          {title}
        </span>
      </div>
      <div className="px-4 py-3 space-y-2.5">
        {children}
      </div>
    </div>
  )
}

// ── Main panel ────────────────────────────────────────────────────────────────

export function ProcessorConfigPanel({ open, onClose, harnessConfig, onChange }: Props) {
  const t = useT()
  const processors = harnessConfig.processors

  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open, onClose])

  if (!open) return null

  function setProcessors(next: ProcessorList) {
    onChange({ ...harnessConfig, processors: next })
  }

  function toggle(target: string, checked: boolean, defaultEntry?: Record<string, unknown>) {
    setProcessors(checked
      ? addProc(processors, defaultEntry ?? { _target_: target })
      : removeProc(processors, target))
  }

  const hasCompaction = hasProc(processors, T.COMPACTION)
  const compactionThreshold = (getProcProp(processors, T.COMPACTION, 'token_threshold') as number | undefined) ?? 80_000

  const hasMem = hasProc(processors, T.MEM_EXTRACTION)
  const memN   = (getProcProp(processors, T.MEM_EXTRACTION, 'memory.n') as number | undefined) ?? 40

  const hasCostGuard = hasProc(processors, T.COST_GUARD)
  const maxCostUsd   = (getProcProp(processors, T.COST_GUARD, 'max_usd') as number | undefined) ?? null

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40"
        style={{ background: 'rgba(0,0,0,0.25)' }}
        onClick={onClose}
      />

      {/* Drawer */}
      <div
        className="fixed top-0 right-0 h-full z-50 flex flex-col overflow-hidden"
        style={{
          width:      'min(480px, 100vw)',
          background: 'var(--bg-base)',
          borderLeft: '1px solid var(--border)',
          boxShadow:  '-8px 0 32px rgba(0,0,0,0.15)',
        }}
      >
        {/* Header */}
        <div
          className="flex items-center gap-3 px-4 py-3 shrink-0"
          style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-card)' }}
        >
          <span
            className="font-semibold text-sm flex-1"
            style={{ color: 'var(--text-1)', letterSpacing: '-0.02em' }}
          >
            {t('processor.title')}
          </span>
          <button
            onClick={onClose}
            className="p-1 rounded transition-colors"
            style={{ color: 'var(--text-3)' }}
            onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--text-1)')}
            onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-3)')}
          >
            <X size={15} />
          </button>
        </div>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto">

          {/* Context */}
          <DimSection title={t('processor.dim.context')} accent="#0099c0">
            <Toggle
              label="System Prompt Builder"
              hint="Build context-aware system prompts"
              checked={hasProc(processors, T.SYSTEM_PROMPT)}
              onChange={(v) => toggle(T.SYSTEM_PROMPT, v)}
            />
            <Toggle
              label="User Wrapper"
              hint="Wrap user messages (XML / CoT)"
              checked={hasProc(processors, T.USER_WRAPPER)}
              onChange={(v) => toggle(T.USER_WRAPPER, v)}
            />
            <Toggle
              label="Environment Injection"
              hint="Inject env variables into context"
              checked={hasProc(processors, T.ENV_INJECTION)}
              onChange={(v) => toggle(T.ENV_INJECTION, v)}
            />
            <Toggle
              label="Compaction"
              hint="Summarize history when context grows large"
              checked={hasCompaction}
              onChange={(v) => toggle(T.COMPACTION, v, { _target_: T.COMPACTION, token_threshold: 80_000 })}
            />
            {hasCompaction && (
              <NumberField
                label="Token Threshold"
                value={compactionThreshold}
                min={10000}
                max={500000}
                step={5000}
                onChange={(v) => setProcessors(setProcProp(processors, T.COMPACTION, 'token_threshold', v ?? 80_000))}
              />
            )}
          </DimSection>

          {/* Memory */}
          <DimSection title={t('processor.dim.memory')} accent="#7c3aed">
            <Toggle
              label="Sliding Window Memory"
              hint="Keep recent messages in context window"
              checked={hasMem}
              onChange={(v) => toggle(T.MEM_EXTRACTION, v, {
                _target_: T.MEM_EXTRACTION,
                memory: { _target_: T.SLIDING_WIN, n: 40 },
                threshold: 140_000,
              })}
            />
            {hasMem && (
              <NumberField
                label="Window Size"
                value={memN}
                min={1}
                max={200}
                onChange={(v) => setProcessors(setProcProp(processors, T.MEM_EXTRACTION, 'memory.n', v ?? 40))}
              />
            )}
          </DimSection>

          {/* Control */}
          <DimSection title={t('processor.dim.control')} accent="#d97706">
            <div className="grid grid-cols-1 gap-2">
              <Toggle
                label="Loop Detection"
                checked={hasProc(processors, T.LOOP_DETECT)}
                onChange={(v) => toggle(T.LOOP_DETECT, v)}
              />
              <Toggle
                label="Tool Call Correction"
                checked={hasProc(processors, T.TOOL_CORRECT)}
                onChange={(v) => toggle(T.TOOL_CORRECT, v)}
              />
              <Toggle
                label="Parse Retry"
                checked={hasProc(processors, T.PARSE_RETRY)}
                onChange={(v) => toggle(T.PARSE_RETRY, v)}
              />
              <Toggle
                label="Tool Failure Guard"
                checked={hasProc(processors, T.TOOL_FAIL_GUARD)}
                onChange={(v) => toggle(T.TOOL_FAIL_GUARD, v)}
              />
              <Toggle
                label="Repeated Edit Detector"
                checked={hasProc(processors, T.REPEATED_EDIT)}
                onChange={(v) => toggle(T.REPEATED_EDIT, v)}
              />
              <Toggle
                label="Background Install Guard"
                checked={hasProc(processors, T.BG_GUARD)}
                onChange={(v) => toggle(T.BG_GUARD, v)}
              />
              <Toggle
                label="Skill Loading"
                hint="Auto-inject relevant skills per step"
                checked={hasProc(processors, T.SKILL_LOAD)}
                onChange={(v) => toggle(T.SKILL_LOAD, v)}
              />
            </div>

            <div className="mt-1 pt-2.5" style={{ borderTop: '1px solid var(--border-sub)' }}>
              <Toggle
                label="Cost Cap"
                hint="Stop run when cost exceeds limit"
                checked={hasCostGuard}
                onChange={(v) => toggle(T.COST_GUARD, v, { _target_: T.COST_GUARD, max_usd: 5.0 })}
              />
              {hasCostGuard && (
                <div className="mt-2">
                  <NumberField
                    label="Max Cost (USD)"
                    value={maxCostUsd}
                    min={0.1}
                    step={0.5}
                    placeholder="5.0"
                    onChange={(v) => setProcessors(setProcProp(processors, T.COST_GUARD, 'max_usd', v ?? 5.0))}
                  />
                </div>
              )}
            </div>
          </DimSection>

          {/* Observability */}
          <DimSection title={t('processor.dim.observability')} accent="#059669">
            <Toggle
              label="JSONL Export"
              hint="Write session trace to .jsonl file"
              checked={hasProc(processors, T.OBS_JSONL)}
              onChange={(v) => toggle(T.OBS_JSONL, v)}
            />
            <Toggle
              label="OpenTelemetry"
              hint="Emit spans to OTEL collector"
              checked={hasProc(processors, T.OBS_OTEL)}
              onChange={(v) => toggle(T.OBS_OTEL, v)}
            />
            <Toggle
              label="Checkpoint DB"
              hint="Persist state snapshots for resume"
              checked={hasProc(processors, T.OBS_CHECKPOINT)}
              onChange={(v) => toggle(T.OBS_CHECKPOINT, v)}
            />
          </DimSection>

          {/* Evaluation */}
          <DimSection title={t('processor.dim.evaluation')} accent="#dc2626">
            <Toggle
              label="Evaluation Processor"
              hint="Evaluate agent output against success criteria"
              checked={hasProc(processors, T.EVALUATION)}
              onChange={(v) => toggle(T.EVALUATION, v)}
            />
          </DimSection>

          <div className="h-6" />
        </div>
      </div>
    </>
  )
}
