import { useState } from 'react'
import type { Dimension, DimensionOption, DimensionParam, DimensionParamTarget, Conflict } from '../../api/types'
import { OptionChip } from './OptionChip'
import { ParamSlider } from './ParamSlider'
import { DRAG_TYPE, buildDragPayload } from '../../utils/dragUtils'
import { useLabStore } from '../../store/lab'

type ProcessorList = Record<string, unknown>[]

interface DimensionCardProps {
  dimension: Dimension
  processors: ProcessorList
  onProcessorsChange: (p: ProcessorList) => void
  readOnly?: boolean
}

const ICON: Record<string, string> = {
  brain: '⬡', layers: '⧉', shield: '◈', shield_check: '◈', eye: '◎', flask: '◇',
  network: '⬡', wrench: '◆', dollar: '◇', dollar_sign: '◇',
  file_text: '▤', activity: '▦', git_fork: '⑂', check_circle: '◉',
}

const SEVERITY_CLS: Record<string, string> = {
  error:   'border-red-300 bg-red-50 text-red-600 dark:border-red-800/60 dark:bg-red-950/40 dark:text-red-400',
  warning: 'border-yellow-300 bg-yellow-50 text-yellow-700 dark:border-yellow-800/60 dark:bg-yellow-950/40 dark:text-yellow-400',
  info:    'border-blue-300 bg-blue-50 text-blue-600 dark:border-blue-800/60 dark:bg-blue-950/40 dark:text-blue-400',
}

const DIM_ACCENT: Record<string, string> = {
  context:      '#0099c0',
  control:      '#d97706',
  observability:'#059669',
  evaluation:   '#7c3aed',
  memory:       '#ea580c',
  tool_choice:  '#db2777',
  multi_model:  '#0284c7',
}

const DIM_ACCENT_DARK: Record<string, string> = {
  context:      '#00d4ff',
  control:      '#f59e0b',
  observability:'#10b981',
  evaluation:   '#8b5cf6',
  memory:       '#f97316',
  tool_choice:  '#ec4899',
  multi_model:  '#06b6d4',
}

// ── Processor list helpers ────────────────────────────────────────────────────

function hasTarget(processors: ProcessorList, target: string): boolean {
  return processors.some((p) => p._target_ === target)
}

function isOptionActive(opt: DimensionOption, processors: ProcessorList): boolean {
  return opt.processors.length > 0 &&
    opt.processors.every((p) => hasTarget(processors, p._target_ as string))
}

function applyOption(opt: DimensionOption, dimension: Dimension, processors: ProcessorList): ProcessorList {
  const allDimTargets = new Set(
    dimension.options.flatMap((o) => o.processors.map((p) => p._target_ as string)),
  )
  const kept = processors.filter((p) => !allDimTargets.has(p._target_ as string))
  return [...kept, ...opt.processors]
}

/** Extract the bare class name from any _target_ string.
 *  file:///path/to/file.py::ClassName  →  "ClassName"
 *  module.path.ClassName               →  "ClassName"
 */
function targetClassName(target: string): string | null {
  if (typeof target !== 'string' || !target.trim()) return null
  if (target.startsWith('file://')) {
    const after = target.split('::')
    const cls = after.length >= 2 ? after[after.length - 1].trim() : ''
    return cls || null
  }
  const parts = target.split('.')
  return parts[parts.length - 1].trim() || null
}

function toggleOptionProcessors(opt: DimensionOption, checked: boolean, processors: ProcessorList): ProcessorList {
  if (checked) {
    // For file:// processors, remove any existing processor with the same class
    // name before adding — this replaces stale file:// paths and module-path
    // entries that accumulated from previous imports of the same processor.
    let base = processors
    for (const p of opt.processors) {
      const t = p._target_ as string | undefined
      if (typeof t === 'string' && t.startsWith('file://')) {
        const cls = targetClassName(t)
        if (cls) base = base.filter((e) => targetClassName(e._target_ as string) !== cls)
      }
    }
    const existing = new Set(base.map((p) => p._target_ as string))
    const toAdd = opt.processors.filter((p) => !existing.has(p._target_ as string))
    return [...base, ...toAdd]
  }
  const optTargets = new Set(opt.processors.map((p) => p._target_ as string))
  return processors.filter((p) => !optTargets.has(p._target_ as string))
}

function getParamValue(
  processors: ProcessorList,
  targets: DimensionParamTarget[],
  defaultVal: number | string,
): number | string {
  for (const t of targets) {
    const proc = processors.find((p) => p._target_ === t.processor_target)
    if (!proc) continue
    const parts = t.path.split('.')
    let val: unknown = proc
    for (const part of parts) {
      val = (val as Record<string, unknown>)?.[part]
    }
    if (val !== undefined && val !== null) return val as number | string
  }
  return defaultVal
}

function setParamValue(
  processors: ProcessorList,
  targets: DimensionParamTarget[],
  value: unknown,
): ProcessorList {
  return processors.map((proc) => {
    const target = targets.find((t) => t.processor_target === proc._target_)
    if (!target) return proc
    const parts = target.path.split('.')
    const updated = { ...proc }
    if (parts.length === 1) {
      updated[parts[0]] = value
    } else {
      const nested = { ...(updated[parts[0]] as Record<string, unknown> ?? {}) }
      nested[parts[1]] = value
      updated[parts[0]] = nested
    }
    return updated
  })
}

function getProcessorArgs(proc: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const [k, v] of Object.entries(proc)) {
    if (k === '_target_') continue
    out[k] = v
  }
  return out
}

function setProcessorArgs(
  processors: ProcessorList,
  target: string,
  args: Record<string, unknown>,
): ProcessorList {
  return processors.map((proc) => {
    if (proc._target_ !== target) return proc
    return { _target_: target, ...args }
  })
}

function shortTargetName(target: string): string {
  const parts = target.split('.')
  return parts[parts.length - 1] ?? target
}

function optionActiveProcessors(opt: DimensionOption, processors: ProcessorList): ProcessorList {
  const wanted = new Set(opt.processors.map((p) => String(p._target_ ?? '')))
  return processors.filter((p) => wanted.has(String(p._target_ ?? '')))
}

function collectConflicts(
  options: DimensionOption[],
  processors: ProcessorList,
): Array<Conflict & { optionLabel: string }> {
  const triggered: Array<Conflict & { optionLabel: string }> = []
  for (const opt of options) {
    if (!opt.conflicts) continue
    for (const c of opt.conflicts) {
      const present = hasTarget(processors, c.if_processor)
      const active = c.negate ? !present : present
      if (active) triggered.push({ ...c, optionLabel: opt.label })
    }
  }
  return triggered
}

function AdvancedParamsEditor({
  options,
  processors,
  onProcessorsChange,
}: {
  options: DimensionOption[]
  processors: ProcessorList
  onProcessorsChange: (p: ProcessorList) => void
}) {
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [expanded, setExpanded] = useState(false)

  const merged = new Map<string, Record<string, unknown>>()
  for (const opt of options) {
    for (const proc of optionActiveProcessors(opt, processors)) {
      const target = String(proc._target_ ?? '')
      if (!target) continue
      if (!merged.has(target)) merged.set(target, proc)
    }
  }
  const active = [...merged.values()]
  if (active.length === 0) return null

  return (
    <div className="mt-3 pt-3" style={{ borderTop: '1px solid var(--border)' }}>
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center justify-between text-left rounded px-2 py-1.5 transition-colors"
        style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)' }}
      >
        <span
          className="text-[11px] font-semibold uppercase tracking-wider"
          style={{ color: 'var(--text-3)' }}
        >
          Advanced Params
        </span>
        <span className="text-xs font-mono" style={{ color: 'var(--text-4)' }}>
          {expanded ? '−' : '+'}
        </span>
      </button>

      {!expanded ? null : (
        <>
          <p className="text-[11px] mt-2 mb-2 leading-relaxed" style={{ color: 'var(--text-4)' }}>
            Edit processor kwargs as JSON object. Do not include <code>_target_</code>.
          </p>

          <div className="flex flex-col gap-2">
            {active.map((proc) => {
              const target = String(proc._target_ ?? '')
              const args = getProcessorArgs(proc)
              const normalized = JSON.stringify(args, null, 2)
              const draft = drafts[target] ?? normalized
              return (
                <div
                  key={target}
                  className="rounded-md p-2"
                  style={{ border: '1px solid var(--border)', background: 'var(--bg-elevated)' }}
                >
                  <div className="text-[11px] mb-1.5 font-mono" style={{ color: 'var(--text-2)' }}>
                    {shortTargetName(target)}
                  </div>
                  <textarea
                    value={draft}
                    onChange={(e) => {
                      const txt = e.target.value
                      setDrafts((s) => ({ ...s, [target]: txt }))
                      if (errors[target]) {
                        setErrors((s) => {
                          const next = { ...s }
                          delete next[target]
                          return next
                        })
                      }
                    }}
                    onBlur={() => {
                      const raw = (drafts[target] ?? normalized).trim()
                      const source = raw === '' ? '{}' : raw
                      try {
                        const parsed = JSON.parse(source)
                        if (parsed === null || Array.isArray(parsed) || typeof parsed !== 'object') {
                          throw new Error('must be a JSON object')
                        }
                        onProcessorsChange(
                          setProcessorArgs(processors, target, parsed as Record<string, unknown>),
                        )
                        const pretty = JSON.stringify(parsed, null, 2)
                        setDrafts((s) => ({ ...s, [target]: pretty }))
                        setErrors((s) => {
                          const next = { ...s }
                          delete next[target]
                          return next
                        })
                      } catch (err) {
                        const message = err instanceof Error ? err.message : 'invalid JSON'
                        setErrors((s) => ({ ...s, [target]: message }))
                      }
                    }}
                    rows={4}
                    className="w-full text-xs rounded px-2 py-1.5 font-mono"
                    style={{ resize: 'vertical' }}
                  />
                  {errors[target] && (
                    <div className="mt-1 text-[11px]" style={{ color: '#dc2626' }}>
                      {errors[target]}
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </>
      )}
    </div>
  )
}

// ── Single-select card ────────────────────────────────────────────────────────

function SingleSelectCard({ dimension, processors, onProcessorsChange }: DimensionCardProps) {
  const selectedOption = dimension.options.find((o) => isOptionActive(o, processors))

  function handleSelect(opt: DimensionOption) {
    let next = applyOption(opt, dimension, processors)
    // Apply param defaults
    for (const p of opt.params ?? []) {
      next = setParamValue(next, p.targets, p.default) as ProcessorList
    }
    onProcessorsChange(next)
  }

  const conflicts = collectConflicts(dimension.options, processors)

  return (
    <>
      <div className="flex flex-wrap gap-1.5">
        {dimension.options.map((opt) => (
          <OptionChip
            key={opt.key}
            option={opt}
            selected={opt === selectedOption}
            onSelect={() => handleSelect(opt)}
          />
        ))}
      </div>

      {selectedOption?.params && selectedOption.params.length > 0 && (
        <div className="mt-3 pt-3" style={{ borderTop: '1px solid var(--border)' }}>
          {selectedOption.params.map((p) => (
            p.type === 'select' ? (
              <div key={p.targets[0]?.path ?? p.label} className="flex items-center gap-2 mt-1.5">
                <label className="text-xs w-28 shrink-0" style={{ color: 'var(--text-3)' }}>{p.label}</label>
                <select
                  value={String(getParamValue(processors, p.targets, p.default))}
                  onChange={(e) => onProcessorsChange(setParamValue(processors, p.targets, e.target.value) as ProcessorList)}
                  className="text-xs rounded px-1.5 py-0.5"
                >
                  {p.options?.map((v) => <option key={v} value={v}>{v}</option>)}
                </select>
              </div>
            ) : (
              <ParamSlider
                key={p.targets[0]?.path ?? p.label}
                param={p as DimensionParam & { min: number; max: number; default: number }}
                value={getParamValue(processors, p.targets, p.default) as number}
                onChange={(v) => onProcessorsChange(setParamValue(processors, p.targets, v) as ProcessorList)}
              />
            )
          ))}
        </div>
      )}

      {selectedOption && (
        <AdvancedParamsEditor
          options={[selectedOption]}
          processors={processors}
          onProcessorsChange={onProcessorsChange}
        />
      )}

      {conflicts.length > 0 && (
        <div className="mt-2 flex flex-col gap-1">
          {conflicts.map((c, i) => (
            <div key={i} className={`text-xs rounded-md px-2 py-1 border ${SEVERITY_CLS[c.severity] ?? SEVERITY_CLS.info}`}>
              {c.message}
            </div>
          ))}
        </div>
      )}
    </>
  )
}

// ── Multi-select card ─────────────────────────────────────────────────────────

function MultiSelectCard({ dimension, processors, onProcessorsChange }: DimensionCardProps) {
  const conflicts = collectConflicts(dimension.options, processors)
  const activeOptions = dimension.options.filter((opt) => isOptionActive(opt, processors))

  return (
    <>
      <div className="flex flex-col gap-0.5">
        {dimension.options.map((opt) => {
          const checked = isOptionActive(opt, processors)
          return (
            <label
              key={opt.key}
              className="flex items-start gap-2.5 cursor-pointer rounded-md px-2 py-1.5 transition-colors"
              style={{ background: checked ? 'var(--accent-bg)' : 'transparent' }}
              title={opt.description}
            >
              <input
                type="checkbox"
                checked={checked}
                onChange={(e) => onProcessorsChange(toggleOptionProcessors(opt, e.target.checked, processors))}
                className="mt-0.5 h-3.5 w-3.5 shrink-0"
              />
              <div className="flex-1 min-w-0">
                <span
                  className="text-xs font-medium transition-colors"
                  style={{ color: checked ? 'var(--text-1)' : 'var(--text-2)' }}
                >
                  {opt.label}
                </span>
                {checked && opt.params && opt.params.length > 0 && (
                  <div className="mt-1 pl-0">
                    {opt.params.map((p) =>
                      p.type === 'select' ? (
                        <div key={p.targets[0]?.path ?? p.label} className="flex items-center gap-2 mt-1">
                          <label className="text-xs shrink-0" style={{ color: 'var(--text-3)' }}>{p.label}:</label>
                          <select
                            value={String(getParamValue(processors, p.targets, p.default))}
                            onChange={(e) => onProcessorsChange(setParamValue(processors, p.targets, e.target.value) as ProcessorList)}
                            className="text-xs rounded px-1.5 py-0.5"
                          >
                            {p.options?.map((v) => <option key={v} value={v}>{v}</option>)}
                          </select>
                        </div>
                      ) : (
                        <ParamSlider
                          key={p.targets[0]?.path ?? p.label}
                          param={p as DimensionParam & { min: number; max: number; default: number }}
                          value={getParamValue(processors, p.targets, p.default) as number}
                          onChange={(v) => onProcessorsChange(setParamValue(processors, p.targets, v) as ProcessorList)}
                        />
                      )
                    )}
                  </div>
                )}
              </div>
            </label>
          )
        })}
      </div>

      {activeOptions.length > 0 && (
        <AdvancedParamsEditor
          options={activeOptions}
          processors={processors}
          onProcessorsChange={onProcessorsChange}
        />
      )}

      {conflicts.length > 0 && (
        <div className="mt-2 flex flex-col gap-1">
          {conflicts.map((c, i) => (
            <div key={i} className={`text-xs rounded-md px-2 py-1 border ${SEVERITY_CLS[c.severity] ?? SEVERITY_CLS.info}`}>
              {c.message}
            </div>
          ))}
        </div>
      )}
    </>
  )
}

// ── Main card wrapper ─────────────────────────────────────────────────────────

export function DimensionCard(props: DimensionCardProps) {
  const { dimension, processors, readOnly = false } = props
  const setDraggingDimension = useLabStore((s) => s.setDraggingDimension)
  const copyDimension = useLabStore((s) => s.copyDimension)
  const theme = typeof document !== 'undefined' && document.documentElement.classList.contains('dark') ? 'dark' : 'light'
  const [copied, setCopied] = useState(false)
  const [isDragging, setIsDragging] = useState(false)

  const accent = theme === 'dark'
    ? (DIM_ACCENT_DARK[dimension.key] ?? '#00d4ff')
    : (DIM_ACCENT[dimension.key] ?? '#0099c0')

  function handleDragStart(e: React.DragEvent) {
    const payload = buildDragPayload(dimension, processors)
    e.dataTransfer.setData(DRAG_TYPE, JSON.stringify(payload))
    e.dataTransfer.effectAllowed = 'copy'
    setDraggingDimension(payload)
    setIsDragging(true)
  }

  function handleDragEnd() {
    setDraggingDimension(null)
    setIsDragging(false)
  }

  function handleCopy(e: React.MouseEvent) {
    e.stopPropagation()
    copyDimension(buildDragPayload(dimension, processors))
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return (
    <div
      className={`group/card relative rounded-xl overflow-hidden transition-all duration-200${isDragging ? ' is-dragging' : ''}`}
      style={{
        background: 'var(--bg-card)',
        border: `1px solid ${isDragging ? accent : 'var(--border)'}`,
        boxShadow: isDragging
          ? `var(--shadow-card-drag), 0 0 20px ${accent}28`
          : 'var(--shadow-card)',
        transform: isDragging ? 'scale(0.97)' : undefined,
      }}
    >
      {/* Colored header band — only this area is draggable */}
      <div
        className="flex items-center gap-2 px-3 py-2.5"
        draggable={!readOnly}
        onDragStart={handleDragStart}
        onDragEnd={handleDragEnd}
        style={{
          background: `${accent}12`,
          borderBottom: `1px solid ${accent}22`,
          cursor: 'grab',
        }}
      >
        {/* SVG grip handle */}
        <svg
          width="8" height="12" viewBox="0 0 8 12"
          fill="none"
          aria-label={`Drag "${dimension.label}" to a Compare column`}
          className="shrink-0 opacity-40 group-hover/card:opacity-70 transition-opacity"
          style={{ color: accent }}
        >
          {[0, 4].map((cx) =>
            [0, 4, 8].map((cy) => (
              <circle key={`${cx}-${cy}`} cx={cx + 1.5} cy={cy + 1.5} r="1.5" fill="currentColor" />
            ))
          )}
        </svg>

        <span
          className="text-sm shrink-0 font-mono leading-none"
          aria-hidden="true"
          style={{ color: accent }}
        >
          {ICON[dimension.icon] ?? '◇'}
        </span>

        <span
          className="text-xs font-semibold flex-1 truncate"
          style={{
            color: accent,
            letterSpacing: '0.05em',
            textTransform: 'uppercase',
            fontSize: '10px',
            fontFamily: 'JetBrains Mono, monospace',
          }}
        >
          {dimension.label}
        </span>

        <button
          onClick={handleCopy}
          disabled={readOnly}
          title="Copy dimension config to clipboard"
          className="opacity-0 group-hover/card:opacity-100 transition-opacity text-xs px-1 py-0.5 rounded disabled:opacity-30 disabled:cursor-not-allowed"
          style={{
            color: copied ? accent : 'var(--text-3)',
            background: copied ? `${accent}18` : 'transparent',
          }}
        >
          {copied ? '✓' : '⎘'}
        </button>
      </div>

      {/* Card body */}
      <div className="p-3.5">
        <p className="text-xs mb-3 leading-relaxed" style={{ color: 'var(--text-3)' }}>
          {dimension.description}
        </p>

        <fieldset
          disabled={readOnly}
          style={{ margin: 0, padding: 0, border: 0, opacity: readOnly ? 0.7 : 1 }}
        >
          {dimension.multi_select
            ? <MultiSelectCard {...props} />
            : <SingleSelectCard {...props} />
          }
        </fieldset>
      </div>
    </div>
  )
}
