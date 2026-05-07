import { useEffect, useState, useCallback } from 'react'
import { Plus, RefreshCw, Square } from 'lucide-react'
import { api } from '@gw/api/client'
import { useSlotsStore } from '@lab/store/slots'
import { useT } from '@gw/i18n'
import type { GatewaySessionMeta, SessionDisplayMessage, ChatMessage, MessageBlock, Attachment, TaskBlock } from '@gw/api/types'
import { ChatPanel } from '@lab/components/chat/ChatPanel'
import { ChatInput } from '@lab/components/chat/ChatInput'

// ── Type adapter ──────────────────────────────────────────────────────────────

function parseMultimodalContent(content: unknown): { text: string; attachments: Attachment[] } {
  if (typeof content === 'string') return { text: content, attachments: [] }
  if (!Array.isArray(content)) return { text: String(content ?? ''), attachments: [] }
  const textParts: string[] = []
  const attachments: Attachment[] = []
  for (const block of content) {
    if (block?.type === 'text') textParts.push(block.text ?? '')
    else if (block?.type === 'image') {
      if (block.media_url) {
        // Server-referenced media file — use URL directly
        attachments.push({
          type: 'image',
          media_type: block.media_type ?? 'image/jpeg',
          data: '',
          url: block.media_url,
        })
      } else if (block.source?.data) {
        // Inline base64
        attachments.push({
          type: 'image',
          media_type: block.source.media_type ?? 'image/png',
          data: block.source.data,
        })
      }
    }
  }
  return { text: textParts.join('\n'), attachments }
}

function toLabMsg(msg: SessionDisplayMessage): ChatMessage {
  if (msg.role !== 'assistant' || !msg.tool_calls?.length) {
    const { text, attachments } = parseMultimodalContent(msg.content)
    return {
      role: msg.role,
      content: text,
      ...(attachments.length > 0 ? { attachments } : {}),
      stepTraces: msg.step_traces,
      query_context: msg.query_context,
    }
  }
  // Build structured blocks: tool_use + tool_result pairs, then final text
  const blocks: MessageBlock[] = []
  for (const tc of msg.tool_calls) {
    blocks.push({ type: 'tool_use', id: tc.id, name: tc.name, input: {} })
    if (tc.output !== undefined) {
      blocks.push({
        type: 'tool_result', id: tc.id, name: tc.name,
        output: tc.output, error: null, duration_ms: 0,
      })
    }
  }
  const { text: assistantText } = parseMultimodalContent(msg.content)
  if (assistantText) blocks.push({ type: 'text', content: assistantText })
  return {
    role: msg.role,
    content: assistantText,
    blocks,
    stepTraces: msg.step_traces,
    query_context: msg.query_context,
  }
}

// ── Session list item ─────────────────────────────────────────────────────────

interface SessionItemProps {
  session: GatewaySessionMeta
  active: boolean
  onClick: () => void
}

function SessionItem({ session, active, onClick }: SessionItemProps) {
  return (
    <button
      onClick={onClick}
      className="w-full text-left px-3 py-2.5 rounded-lg transition-colors"
      style={{
        background: active ? 'var(--accent-bg)' : 'transparent',
        border: `1px solid ${active ? 'var(--accent-ring)' : 'transparent'}`,
      }}
    >
      <div
        className="truncate font-medium"
        style={{ fontSize: 12, color: active ? 'var(--accent)' : 'var(--text-1)' }}
      >
        {session.first_query || session.session_id.split(':').pop()?.slice(0, 8) || session.session_id}
      </div>
      {session.updated_at && (
        <div style={{ fontSize: 10, color: 'var(--text-4)', marginTop: 2 }}>
          {new Date(session.updated_at).toLocaleDateString()}
          {' · '}{session.run_count} run{session.run_count !== 1 ? 's' : ''}
        </div>
      )}
    </button>
  )
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function sessionKey(agentId: string) {
  return `gw_console_session_id_${agentId}`
}

// ── Main ChatPage ─────────────────────────────────────────────────────────────

export function ChatPage() {
  const t = useT()
  const agentId = useSlotsStore((s) => s.agentId)

  const [sessionId, setSessionIdState] = useState<string>(() =>
    localStorage.getItem(sessionKey(agentId)) || ''
  )
  const [sessions, setSessions]           = useState<GatewaySessionMeta[]>([])
  const [sessionsLoading, setSessionsLoading] = useState(false)
  const [history, setHistory]             = useState<SessionDisplayMessage[]>([])
  const [pendingMsg, setPendingMsg]       = useState('')
  const [pendingAttachments, setPendingAttachments] = useState<Attachment[]>([])
  const [streamingText, setStreamingText] = useState('')
  const [isRunning, setIsRunning]         = useState(false)
  const [runId, setRunId]                 = useState<string | null>(null)

  useEffect(() => {
    const stored = localStorage.getItem(sessionKey(agentId)) || ''
    setSessionIdState(stored)
  }, [agentId])

  const persistSession = useCallback((sid: string) => {
    setSessionIdState(sid)
    localStorage.setItem(sessionKey(agentId), sid)
  }, [agentId])

  const loadSessions = useCallback(() => {
    setSessionsLoading(true)
    api.listGatewaySessions('web_ui')
      .then((s) => { setSessions(s); setSessionsLoading(false) })
      .catch(() => setSessionsLoading(false))
  }, [])

  useEffect(() => { loadSessions() }, [loadSessions])

  useEffect(() => {
    if (!sessionId) { setHistory([]); return }
    api.getSessionMessages(sessionId, agentId, 'web_ui', 'im-workspaces')
      .then((r) => setHistory(r.messages))
      .catch(() => setHistory([]))
  }, [sessionId, agentId])

  const newChat = () => {
    persistSession('')
    setHistory([])
    setStreamingText('')
    setPendingMsg('')
  }

  const selectSession = (sid: string) => {
    if (isRunning) return
    persistSession(sid)
    setStreamingText('')
    setPendingMsg('')
  }

  const cancelRun = () => {
    if (runId) api.cancelRun(runId).catch(console.error)
  }

  const send = useCallback(async (msg: string, attachments: Attachment[] = []) => {
    if ((!msg && attachments.length === 0) || isRunning) return
    setIsRunning(true)
    setPendingMsg(msg || '(image)')
    setPendingAttachments(attachments)
    setStreamingText('')

    let sid = sessionId

    try {
      const task: string | TaskBlock[] = attachments.length > 0
        ? [
            ...(msg ? [{ type: 'text' as const, text: msg }] : []),
            ...attachments.map((a) => ({
              type: 'image' as const,
              source: { type: 'base64' as const, media_type: a.media_type, data: a.data },
            })),
          ]
        : msg
      const res = await api.startConsoleRun(task, sid || undefined)
      if (!sid) {
        sid = res.session_id
        persistSession(sid)
      }
      setRunId(res.run_id)

      // eslint-disable-next-line prefer-const
      let stopStream: () => void = () => {}
      stopStream = api.streamRun(res.run_id, (event: any) => {
        if (event.type === 'warning') {
          setStreamingText((prev) => prev + `[${event.message}]\n\n`)
        } else if (event.type === 'token') {
          setStreamingText((prev) => prev + (event.content ?? ''))
        } else if (event.type === 'done' || event.type === 'error') {
          setIsRunning(false)
          setRunId(null)
          setPendingMsg('')
          setPendingAttachments([])
          setStreamingText('')
          stopStream()
          api.getSessionMessages(sid!, agentId, 'web_ui', 'im-workspaces')
            .then((r) => setHistory(r.messages))
            .catch(console.error)
          loadSessions()
        }
      })
    } catch (e) {
      setIsRunning(false)
      setPendingMsg('')
      setPendingAttachments([])
      console.error(e)
    }
  }, [isRunning, sessionId, agentId, persistSession, loadSessions])

  // Assemble display messages for ChatPanel
  const displayMessages: ChatMessage[] = [
    ...history.map(toLabMsg),
    ...(pendingMsg ? [{
      role: 'user' as const,
      content: pendingMsg,
      ...(pendingAttachments.length > 0 ? { attachments: pendingAttachments } : {}),
    }] : []),
    ...(isRunning || streamingText
      ? [{ role: 'assistant' as const, content: streamingText, streaming: isRunning }]
      : []),
  ]

  return (
    <div className="flex flex-1 min-h-0 overflow-hidden">
      {/* ── Left: session list ── */}
      <div
        className="flex flex-col shrink-0 overflow-hidden"
        style={{ width: 220, borderRight: '1px solid var(--border)', background: 'var(--bg-card)' }}
      >
        <div
          className="flex items-center justify-between px-3 py-2.5 shrink-0"
          style={{ borderBottom: '1px solid var(--border)' }}
        >
          <span className="font-semibold" style={{ fontSize: 12, color: 'var(--text-2)' }}>
            {t('gw.chat.sessions')}
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={loadSessions}
              className="p-1 rounded"
              style={{ color: 'var(--text-4)', background: 'transparent' }}
              onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--text-1)' }}
              onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-4)' }}
            >
              <RefreshCw size={11} className={sessionsLoading ? 'animate-spin' : ''} />
            </button>
            <button
              onClick={newChat}
              className="p-1 rounded"
              style={{ color: 'var(--text-4)', background: 'transparent' }}
              onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--accent)' }}
              onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-4)' }}
              title={t('gw.chat.new')}
            >
              <Plus size={13} />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-1.5 flex flex-col gap-0.5">
          {sessions.length === 0 && !sessionsLoading && (
            <p className="px-2 py-2" style={{ fontSize: 11, color: 'var(--text-4)' }}>
              {t('gw.sessions.empty')}
            </p>
          )}
          {sessions.map((s) => (
            <SessionItem
              key={s.session_id}
              session={s}
              active={s.session_id === sessionId}
              onClick={() => selectSession(s.session_id)}
            />
          ))}
        </div>
      </div>

      {/* ── Right: chat area ── */}
      <div className="flex flex-col flex-1 min-h-0 overflow-hidden">
        <div className="flex-1 min-h-0 flex flex-col max-w-3xl mx-auto w-full">
          <ChatPanel messages={displayMessages} autoScroll={true} />

          {/* Stop button — only visible while running */}
          {isRunning && (
            <div
              className="shrink-0 flex justify-center py-1.5"
              style={{ background: 'var(--bg-card)' }}
            >
              <button
                onClick={cancelRun}
                className="flex items-center gap-1.5 px-3 py-1 rounded-full"
                style={{
                  fontSize: 12, background: '#ff3b3015',
                  color: '#ff3b30', border: '1px solid #ff3b3030',
                }}
              >
                <Square size={10} />
                {t('gw.chat.stop')}
              </button>
            </div>
          )}

          <ChatInput
            onSend={(text, attachments) => send(text, attachments)}
            onNewChat={newChat}
            disabled={isRunning}
            placeholder={t('gw.chat.placeholder')}
          />
        </div>
      </div>
    </div>
  )
}
