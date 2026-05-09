import { useEffect, useState } from 'react'
import { Info, Zap } from 'lucide-react'
import type { ChatMessage, MessageBlock, StepTrace } from '../../api/types'
import { useUIStore } from '../../store/ui'
import { StepDetailPanel } from './StepDetailPanel'
import { QueryContextPanel } from './QueryContextPanel'
import { MarkdownText } from './MarkdownText'

interface Props {
  message: ChatMessage
}

const TOOL_COLOR = '#0099c0'

// ─── Block sub-components ────────────────────────────────────────────────────

function ThinkingBlock({
  block,
  streaming,
  postStreamMode,
}: {
  block: Extract<MessageBlock, { type: 'thinking' }>
  streaming?: boolean
  postStreamMode: 'collapse' | 'keep'
}) {
  const [open, setOpen] = useState(Boolean(streaming))
  const [touched, setTouched] = useState(false)

  useEffect(() => {
    if (streaming) {
      if (!touched) setOpen(true)
      return
    }
    if (!touched && postStreamMode === 'collapse') {
      setOpen(false)
    }
  }, [streaming, touched, postStreamMode])

  function toggleOpen() {
    setTouched(true)
    setOpen((v) => !v)
  }

  return (
    <div
      className="my-2 rounded-xl overflow-hidden"
      style={{ border: '1px solid var(--border)', background: 'var(--bg-base)' }}
    >
      <button
        onClick={toggleOpen}
        className="w-full flex items-center gap-2 px-3.5 py-2.5 text-left"
        style={{ color: 'var(--text-3)', background: 'transparent' }}
      >
        <span style={{ opacity: 0.6, fontSize: '12px' }}>{open ? '▾' : '▸'}</span>
        <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', letterSpacing: '0.05em', textTransform: 'uppercase' as const }}>
          Thinking
        </span>
        {!open && (
          <span style={{ color: 'var(--text-4)', fontSize: '12px' }}>
            ({Math.ceil(block.content.length / 4)} tokens est.)
          </span>
        )}
      </button>
      {open && (
        <div
          className="px-3.5 pb-3.5 whitespace-pre-wrap break-words leading-relaxed"
          style={{ color: 'var(--text-3)', fontSize: '13px', fontFamily: 'JetBrains Mono, monospace', borderTop: '1px solid var(--border)' }}
        >
          {block.content}
        </div>
      )}
    </div>
  )
}

function ToolUseBlock({ block }: { block: Extract<MessageBlock, { type: 'tool_use' }> }) {
  const [open, setOpen] = useState(false)
  return (
    <div
      className="my-2 rounded-xl overflow-hidden"
      style={{ border: `1px solid ${TOOL_COLOR}28`, background: `${TOOL_COLOR}06` }}
    >
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2.5 px-3.5 py-2.5 text-left"
        style={{ background: 'transparent' }}
      >
        <span style={{ color: TOOL_COLOR, opacity: 0.7, fontSize: '12px' }}>{open ? '▾' : '▸'}</span>
        <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', letterSpacing: '0.05em', textTransform: 'uppercase' as const, color: TOOL_COLOR, opacity: 0.9 }}>
          Tool
        </span>
        <span className="font-mono font-semibold" style={{ color: 'var(--text-1)', fontSize: '13px' }}>
          {block.name}
        </span>
        <span className="ml-auto font-mono" style={{ color: 'var(--text-4)', fontSize: '11px' }}>
          #{block.id.slice(-6)}
        </span>
      </button>
      {open && (
        <pre
          className="px-3.5 pb-3.5 overflow-x-auto leading-relaxed m-0 whitespace-pre"
          style={{ color: 'var(--text-2)', fontSize: '13px', borderTop: `1px solid ${TOOL_COLOR}28`, fontFamily: 'JetBrains Mono, monospace' }}
        >
          {JSON.stringify(block.input, null, 2)}
        </pre>
      )}
    </div>
  )
}

function ToolResultBlock({ block }: { block: Extract<MessageBlock, { type: 'tool_result' }> }) {
  const [open, setOpen] = useState(false)
  const hasError = !!block.error
  const accent = hasError ? '#ef4444' : '#059669'
  const preview = (block.output || '').slice(0, 140).replace(/\n/g, ' ')
  return (
    <div
      className="my-2 rounded-xl overflow-hidden"
      style={{ border: `1px solid ${accent}28`, background: `${accent}06` }}
    >
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2.5 px-3.5 py-2.5 text-left"
        style={{ background: 'transparent' }}
      >
        <span style={{ color: accent, opacity: 0.7, fontSize: '12px' }}>{open ? '▾' : '▸'}</span>
        <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', letterSpacing: '0.05em', textTransform: 'uppercase' as const, color: accent }}>
          {hasError ? 'Error' : 'Result'}
        </span>
        <span className="font-mono font-semibold" style={{ color: 'var(--text-1)', fontSize: '13px' }}>
          {block.name}
        </span>
        {!open && preview && (
          <span className="truncate flex-1 ml-1" style={{ color: 'var(--text-3)', fontSize: '12px', maxWidth: '240px' }}>
            {preview}
          </span>
        )}
        {block.duration_ms > 0 && (
          <span className="ml-auto font-mono shrink-0" style={{ color: 'var(--text-4)', fontSize: '11px' }}>
            {block.duration_ms < 1000
              ? `${Math.round(block.duration_ms)}ms`
              : `${(block.duration_ms / 1000).toFixed(1)}s`}
          </span>
        )}
      </button>
      {open && (
        <pre
          className="px-3.5 pb-3.5 overflow-x-auto leading-relaxed m-0 whitespace-pre-wrap break-all"
          style={{ color: hasError ? accent : 'var(--text-2)', fontSize: '13px', borderTop: `1px solid ${accent}28`, fontFamily: 'JetBrains Mono, monospace' }}
        >
          {hasError ? block.error : block.output}
        </pre>
      )}
    </div>
  )
}

function AssistantBlocks({
  blocks,
  streaming,
  thinkingPostStream,
}: {
  blocks: MessageBlock[]
  streaming?: boolean
  thinkingPostStream: 'collapse' | 'keep'
}) {
  return (
    <div>
      {blocks.map((block, i) => {
        if (block.type === 'thinking') {
          return (
            <ThinkingBlock
              key={i}
              block={block}
              streaming={streaming}
              postStreamMode={thinkingPostStream}
            />
          )
        }
        if (block.type === 'tool_use')    return <ToolUseBlock    key={i} block={block} />
        if (block.type === 'tool_result') return <ToolResultBlock key={i} block={block} />
        if (block.type === 'text') return (
          <MarkdownText key={i} content={block.content} />
        )
        return null
      })}
      {streaming && (
        <span
          className="inline-block w-2 h-4 ml-1 align-middle animate-pulse rounded-sm"
          style={{ background: 'var(--accent)', opacity: 0.8 }}
        />
      )}
    </div>
  )
}

// ─── Pre-step hints (shown above reply content when Zap icon is active) ───────

const MONO: React.CSSProperties = {
  fontFamily: 'JetBrains Mono, monospace',
  fontSize: '11px',
}

function shortHook(hook: string): string {
  if (hook === 'step_start' || hook === 'on_step_start') return 'step'
  if (hook === 'before_model' || hook === 'on_before_model' || hook === 'on_model_start') return 'model'
  return hook
}

function PreStepHints({ traces }: { traces: StepTrace[] }) {
  return (
    <div
      className="mb-2 pb-2"
      style={{ borderBottom: '1px solid var(--border)' }}
    >
      {traces.map((trace) => {
        const inp = trace.input
        // Only render steps that have something interesting
        const hasCounts = inp && (inp.message_count > 0 || inp.tool_names.length > 0)
        const hasTriggers = inp && inp.on_step_start_triggers.length > 0
        if (!hasCounts && !hasTriggers) return null
        return (
          <div key={trace.step} className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 mb-1 last:mb-0">
            {/* Step number */}
            <span style={{ ...MONO, color: 'var(--text-4)', fontWeight: 700 }}>
              step {trace.step}
            </span>
            {/* Counts from BeforeModelEvent */}
            {inp && inp.message_count > 0 && (
              <span style={{ ...MONO, color: 'var(--text-4)' }}>
                msgs:{inp.message_count}
              </span>
            )}
            {inp && inp.tool_names.length > 0 && (
              <span style={{ ...MONO, color: 'var(--text-4)' }}>
                tools:{inp.tool_names.length}
              </span>
            )}
            {/* on_step_start processor triggers (only those that fired) */}
            {inp && inp.on_step_start_triggers.map((t, i) => (
              <span
                key={i}
                className="rounded px-1.5"
                style={{
                  ...MONO,
                  color: '#ca8a04',
                  background: 'rgba(234,179,8,0.10)',
                  border: '1px solid rgba(234,179,8,0.20)',
                }}
              >
                ⚡ {shortHook(t.hook)} · {t.processor}{t.action ? ` · ${t.action}` : ''}
              </span>
            ))}
          </div>
        )
      })}
    </div>
  )
}

function QueryPostHints({
  triggers,
}: {
  triggers: { processor: string; hook: string; action: string; detail: Record<string, unknown> }[]
}) {
  if (!triggers.length) return null
  return (
    <div
      className="mt-2 pb-2"
      style={{ borderBottom: '1px solid var(--border)' }}
    >
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
        {triggers.map((t, i) => (
          <span
            key={i}
            className="rounded px-1.5"
            style={{
              ...MONO,
              color: '#ca8a04',
              background: 'rgba(234,179,8,0.10)',
              border: '1px solid rgba(234,179,8,0.20)',
            }}
          >
            ⚡ {shortHook(t.hook)} · {t.processor}{t.action ? ` · ${t.action}` : ''}
          </span>
        ))}
      </div>
    </div>
  )
}

// ─── ChatBubble ───────────────────────────────────────────────────────────────

export function ChatBubble({ message }: Props) {
  const isUser = message.role === 'user'
  const thinkingPostStream = useUIStore((s) => s.thinkingPostStream)
  const [detailOpen, setDetailOpen] = useState(false)
  const [inputPreOpen, setInputPreOpen] = useState(false)
  const [inputPostOpen, setInputPostOpen] = useState(false)
  const [replyPreOpen, setReplyPreOpen] = useState(false)

  // ── System divider ─────────────────────────────────────────────────────────
  if (message.role === 'system') {
    return (
      <div className="flex items-center gap-3 my-4 px-1">
        <div className="flex-1 h-px" style={{ background: 'var(--border)' }} />
        <span
          style={{
            fontSize: '11px',
            color: 'var(--text-4)',
            fontFamily: 'JetBrains Mono, monospace',
            letterSpacing: '0.04em',
            whiteSpace: 'nowrap',
          }}
        >
          {message.content}
        </span>
        <div className="flex-1 h-px" style={{ background: 'var(--border)' }} />
      </div>
    )
  }

  // ── User bubble ─────────────────────────────────────────────────────────────
  if (isUser) {
    const qctx = message.query_context
    const hasQueryCtx = !!qctx && (qctx.tool_names.length > 0 || !!qctx.system || qctx.on_task_start_triggers.length > 0)
    const showPreContext = hasQueryCtx
    const postQueryTriggers = qctx?.post_query_triggers ?? []
    const hasPostQueryDetail = postQueryTriggers.length > 0
    const inputTools = qctx?.tool_names.length ?? 0
    const inputTriggers = qctx?.on_task_start_triggers.length ?? 0
    const hasSystem = Boolean(qctx?.system)
    return (
      <div className="flex justify-end mb-5">
        <div className="max-w-[80%]">
          {/* Input summary (before bubble) */}
          {showPreContext && (
            <div
              className="mb-1.5 flex items-center justify-end gap-2"
              style={{ ...MONO, color: 'var(--text-4)' }}
            >
              <span>input-context</span>
              {inputTools > 0 && (
                <>
                  <span style={{ opacity: 0.4 }}>·</span>
                  <span>tools:{inputTools}</span>
                </>
              )}
              {inputTriggers > 0 && (
                <>
                  <span style={{ opacity: 0.4 }}>·</span>
                  <span>triggers:{inputTriggers}</span>
                </>
              )}
              {hasSystem && (
                <>
                  <span style={{ opacity: 0.4 }}>·</span>
                  <span>system</span>
                </>
              )}
              <span style={{ opacity: 0.4 }}>·</span>
              <button
                onClick={() => setInputPreOpen((v) => !v)}
                className="hover:underline"
                style={{ color: 'var(--text-4)', background: 'transparent', padding: 0, cursor: 'pointer' }}
              >
                {inputPreOpen ? 'hide detail' : 'detail'}
              </button>
            </div>
          )}

          {/* Query context panel (pre) */}
          {inputPreOpen && showPreContext && qctx && <QueryContextPanel ctx={qctx} />}

          {/* Bubble + Info icon row */}
          <div className="flex items-start gap-1.5">
            <div
              className="flex-1 px-4 py-3 rounded-2xl"
              style={{
                background: 'var(--bg-elevated)',
                border: '1px solid var(--border)',
                color: 'var(--text-1)',
                fontSize: '15px',
                lineHeight: '1.65',
                borderBottomRightRadius: '6px',
              }}
            >
              {message.attachments && message.attachments.length > 0 && (
                <div className="flex flex-wrap gap-2 mb-2">
                  {message.attachments.map((att, i) => (
                    <img
                      key={i}
                      src={att.url || `data:${att.media_type};base64,${att.data}`}
                      alt={att.name ?? 'image'}
                      className="rounded-xl object-contain"
                      style={{ maxWidth: 240, maxHeight: 200, border: '1px solid var(--border)' }}
                    />
                  ))}
                </div>
              )}
              {message.content && (
                <MarkdownText content={message.content} />
              )}
            </div>

            {/* Info icon — right of bubble, only when query-post trigger exists */}
            {hasPostQueryDetail && (
              <button
                onClick={() => setInputPostOpen((v) => !v)}
                title="Input context"
                className="flex-shrink-0 mt-2 rounded-md p-1 transition-colors"
                style={{
                  color: inputPostOpen ? 'var(--accent)' : 'var(--text-4)',
                  background: inputPostOpen ? 'var(--accent-muted, rgba(99,102,241,0.08))' : 'transparent',
                  cursor: 'pointer',
                }}
              >
                <Info size={13} strokeWidth={2} />
              </button>
            )}
          </div>

          {/* Input summary (after bubble) — only query post-processing detail */}
          {hasPostQueryDetail && (
            <div
              className="mt-1.5 flex items-center justify-end gap-2"
              style={{ ...MONO, color: 'var(--text-4)' }}
            >
              <span>query-post</span>
              {postQueryTriggers.length > 0 && (
                <>
                  <span style={{ opacity: 0.4 }}>·</span>
                  <span>triggers:{postQueryTriggers.length}</span>
                </>
              )}
              <span style={{ opacity: 0.4 }}>·</span>
              <button
                onClick={() => setInputPostOpen((v) => !v)}
                className="hover:underline"
                style={{ color: 'var(--text-4)', background: 'transparent', padding: 0, cursor: 'pointer' }}
              >
                {inputPostOpen ? 'hide detail' : 'detail'}
              </button>
            </div>
          )}

          {/* Query post panel */}
          {inputPostOpen && hasPostQueryDetail && (
            <QueryPostHints triggers={postQueryTriggers} />
          )}
        </div>
      </div>
    )
  }

  // ── Assistant message ───────────────────────────────────────────────────────
  const stepTraces = message.stepTraces
  const hasTraces  = !message.streaming && stepTraces && stepTraces.length > 0
  const totalIn    = hasTraces ? stepTraces.reduce((s, t) => s + t.input_tokens,  0) : 0
  const totalOut   = hasTraces ? stepTraces.reduce((s, t) => s + t.output_tokens, 0) : 0
  const lastModel  = hasTraces ? (stepTraces[stepTraces.length - 1].model ?? '') : ''

  // Show Zap icon when there's step context data (any step with input or triggers)
  const hasPreContext = hasTraces && stepTraces.some(
    (t) => t.input && (t.input.message_count > 0 || t.input.tool_names.length > 0 || t.input.on_step_start_triggers.length > 0)
  )
  const preTriggerCount = hasTraces
    ? stepTraces.reduce((acc, t) => acc + (t.input?.on_step_start_triggers.length ?? 0), 0)
    : 0

  return (
    <div className="flex gap-2 mb-7">
      {/* Left gutter: Zap icon toggle */}
      <div className="flex-shrink-0 w-5 flex flex-col items-center pt-0.5">
        {hasPreContext && (
          <button
            onClick={() => setReplyPreOpen((v) => !v)}
            title="Pre-step context"
            className="rounded-md p-0.5 transition-colors"
            style={{
              color: replyPreOpen ? '#ca8a04' : 'var(--text-4)',
              background: replyPreOpen ? 'rgba(234,179,8,0.10)' : 'transparent',
              cursor: 'pointer',
              opacity: replyPreOpen ? 1 : 0.45,
            }}
          >
            <Zap size={12} strokeWidth={2} />
          </button>
        )}
      </div>

      {/* Content column */}
      <div className="flex-1 min-w-0">
        {/* Pre-step summary hint (always shown when pre-context exists) */}
        {hasPreContext && (
          <div
            className="mb-1.5 flex items-center gap-2"
            style={{ ...MONO, color: 'var(--text-4)' }}
          >
            <span>pre-context</span>
            {preTriggerCount > 0 && (
              <>
                <span style={{ opacity: 0.4 }}>·</span>
                <span>triggers:{preTriggerCount}</span>
              </>
            )}
            <span style={{ opacity: 0.4 }}>·</span>
            <button
              onClick={() => setReplyPreOpen((v) => !v)}
              className="hover:underline"
              style={{ color: 'var(--text-4)', background: 'transparent', padding: 0, cursor: 'pointer' }}
            >
              {replyPreOpen ? 'hide detail' : 'detail'}
            </button>
          </div>
        )}

        {/* Pre-step hints — above content, only when inputOpen and post-stream */}
        {replyPreOpen && hasTraces && (
          <PreStepHints traces={stepTraces} />
        )}

        {/* Reply content */}
        <div style={{ fontSize: '15px', lineHeight: '1.75', color: 'var(--text-1)' }}>
          {message.blocks && message.blocks.length > 0 ? (
            <AssistantBlocks
              blocks={message.blocks}
              streaming={message.streaming}
              thinkingPostStream={thinkingPostStream}
            />
          ) : (
            <>
              <MarkdownText content={message.content} />
              {message.streaming && (
                <span
                  className="inline-block w-2 h-4 ml-1 align-middle animate-pulse rounded-sm"
                  style={{ background: 'var(--accent)', opacity: 0.8 }}
                />
              )}
            </>
          )}
        </div>

        {/* Footnote: model · tokens · detail toggle */}
        {hasTraces && (
          <div
            className="flex items-center gap-2 mt-1.5 select-none"
            style={{ ...MONO, color: 'var(--text-4)' }}
          >
            {lastModel && <span>{lastModel}</span>}
            {lastModel && <span style={{ opacity: 0.4 }}>·</span>}
            <span>↑{totalIn.toLocaleString()}</span>
            <span>↓{totalOut.toLocaleString()}</span>
            <span style={{ opacity: 0.4 }}>·</span>
            <button
              onClick={() => setDetailOpen((v) => !v)}
              className="hover:underline"
              style={{ color: 'var(--text-4)', background: 'transparent', padding: 0, cursor: 'pointer' }}
            >
              {detailOpen ? 'hide detail' : 'detail'}
            </button>
          </div>
        )}

        {/* Step detail panel (expandable) */}
        {detailOpen && stepTraces && stepTraces.length > 0 && (
          <StepDetailPanel traces={stepTraces} />
        )}
      </div>
    </div>
  )
}
