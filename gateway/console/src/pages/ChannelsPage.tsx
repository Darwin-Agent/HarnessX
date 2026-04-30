import { useEffect, useState, useCallback } from 'react'
import { RefreshCw, Plus, Trash2, X } from 'lucide-react'
import { api } from '@gw/api/client'
import type { ChannelInfo, ChannelStatus, ChannelConfigResponse, ChannelTypeInfo, GatewaySessionMeta, SessionDisplayMessage, ChatMessage, MessageBlock } from '@gw/api/types'
import { useSlotsStore } from '@lab/store/slots'
import { ChannelCard } from '../components/channels/ChannelCard'
import { ChannelConfigSheet } from '../components/channels/ChannelConfigSheet'
import { ChatPanel } from '@lab/components/chat/ChatPanel'
import { useT } from '@gw/i18n'

type Tab = 'status' | 'config' | 'sessions'

function toMsg(msg: SessionDisplayMessage): ChatMessage {
  if (msg.role !== 'assistant' || !msg.tool_calls?.length) {
    return { role: msg.role, content: msg.content, stepTraces: msg.step_traces, query_context: msg.query_context }
  }
  const blocks: MessageBlock[] = []
  for (const tc of msg.tool_calls) {
    blocks.push({ type: 'tool_use', id: tc.id, name: tc.name, input: {} })
    if (tc.output !== undefined) {
      blocks.push({ type: 'tool_result', id: tc.id, name: tc.name, output: tc.output, error: null, duration_ms: 0 })
    }
  }
  if (msg.content) blocks.push({ type: 'text', content: msg.content })
  return { role: msg.role, content: msg.content, blocks, stepTraces: msg.step_traces, query_context: msg.query_context }
}

const STATE_COLORS: Record<string, string> = {
  online:     '#34c759',
  connecting: '#ff9f0a',
  offline:    '#8e8e93',
  error:      '#ff3b30',
}

export function ChannelsPage() {
  const t = useT()
  const agentId = useSlotsStore((s) => s.agentId)

  const [channels, setChannels]       = useState<ChannelInfo[]>([])
  const [loading, setLoading]         = useState(true)
  const [fetchError, setFetchError]   = useState<string | null>(null)

  const [selected, setSelected]       = useState<string | null>(null)
  const [tab, setTab]                 = useState<Tab>('status')

  const [status, setStatus]           = useState<ChannelStatus | null>(null)
  const [cfgData, setCfgData]         = useState<ChannelConfigResponse | null>(null)
  const [sessions, setSessions]       = useState<GatewaySessionMeta[]>([])
  const [pairingCode, setPairingCode] = useState<{ code: string; ttl_seconds: number } | null>(null)
  const [pairingError, setPairingError] = useState<string | null>(null)

  const [showAdd, setShowAdd] = useState(false)

  const loadChannels = useCallback(() => {
    setLoading(true)
    setFetchError(null)
    api.listChannels()
      .then((chs) => { setChannels(chs); setLoading(false) })
      .catch((e)  => { setFetchError(String(e)); setLoading(false) })
  }, [])

  useEffect(() => { loadChannels() }, [loadChannels])

  // Auto-refresh every 15s
  useEffect(() => {
    const id = window.setInterval(loadChannels, 15_000)
    return () => window.clearInterval(id)
  }, [loadChannels])

  const selectChannel = useCallback((name: string) => {
    setSelected(name)
    setTab('status')
    setStatus(null)
    setCfgData(null)
    setSessions([])
    setPairingCode(null)
    setPairingError(null)
    api.getChannelStatus(name).then(setStatus).catch(console.error)
  }, [])

  useEffect(() => {
    if (!selected) return
    if (tab === 'config' && !cfgData) {
      api.getChannelConfig(selected).then(setCfgData).catch(console.error)
    }
    if (tab === 'sessions' && sessions.length === 0) {
      api.listGatewaySessions(selected).then(setSessions).catch(console.error)
    }
  }, [selected, tab, cfgData, sessions.length])

  const handleDeleteChannel = (name: string) => {
    if (!window.confirm(t('gw.add.delete_confirm'))) return
    api.deleteChannel(name)
      .then(() => {
        if (selected === name) setSelected(null)
        loadChannels()
      })
      .catch(console.error)
  }

  const selectedInfo = channels.find((c) => c.name === selected)
  const stateColor = selectedInfo
    ? (STATE_COLORS[selectedInfo.connection_state] ?? '#8e8e93')
    : 'var(--text-4)'

  return (
    <div className="flex flex-1 min-h-0 overflow-hidden" style={{ background: 'var(--bg-base)' }}>

      {/* ── Left: channel list ── */}
      <div
        className="flex flex-col shrink-0 overflow-hidden"
        style={{ width: 260, borderRight: '1px solid var(--border)', background: 'var(--bg-card)' }}
      >
        <div
          className="flex items-center justify-between px-4 py-3 shrink-0"
          style={{ borderBottom: '1px solid var(--border)' }}
        >
          <span className="font-semibold" style={{ fontSize: 13, color: 'var(--text-1)' }}>
            {t('ch.title')}
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => setShowAdd(true)}
              className="p-1 rounded transition-all"
              style={{ color: 'var(--accent)', background: 'transparent' }}
              onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--accent-bg)' }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent' }}
              title="Add channel"
            >
              <Plus size={14} />
            </button>
            <button
              onClick={loadChannels}
              className="p-1 rounded transition-all"
              style={{ color: 'var(--text-4)', background: 'transparent' }}
              onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--text-1)'; e.currentTarget.style.background = 'var(--bg-elevated)' }}
              onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-4)'; e.currentTarget.style.background = 'transparent' }}
              title="Refresh"
            >
              <RefreshCw size={13} />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-2">
          {loading && (
            <p style={{ fontSize: 12, color: 'var(--text-4)', padding: '12px 8px' }}>{t('ch.loading')}</p>
          )}
          {!loading && fetchError && (
            <p style={{ fontSize: 12, color: '#ff3b30', padding: '8px' }}>{t('ch.error')}</p>
          )}
          {!loading && !fetchError && channels.length === 0 && (
            <div className="flex flex-col gap-3 p-4" style={{ textAlign: 'center' }}>
              <p style={{ fontSize: 12, color: 'var(--text-4)' }}>{t('ch.empty')}</p>
              <button
                onClick={() => setShowAdd(true)}
                className="px-3 py-1.5 rounded-lg font-medium mx-auto"
                style={{ fontSize: 12, background: 'var(--accent-bg)', color: 'var(--accent)', border: '1px solid var(--accent-ring)' }}
              >
                + {t('gw.add.title')}
              </button>
            </div>
          )}
          {channels.map((ch) => (
            <ChannelCard
              key={ch.name}
              channel={ch}
              selected={ch.name === selected}
              onClick={() => selectChannel(ch.name)}
            />
          ))}
        </div>
      </div>

      {/* ── Right: detail ── */}
      <div className="flex flex-col flex-1 min-w-0 overflow-hidden">
        {!selected ? (
          <div className="flex flex-1 items-center justify-center" style={{ color: 'var(--text-4)', fontSize: 14 }}>
            {!loading && channels.length > 0 ? 'Select a channel' : ''}
          </div>
        ) : (
          <>
            {/* Detail header */}
            <div
              className="flex items-center gap-3 px-6 py-4 shrink-0"
              style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-card)' }}
            >
              <div className="flex flex-col min-w-0">
                <span className="font-semibold" style={{ fontSize: 16, color: 'var(--text-1)' }}>
                  {selectedInfo?.display_name ?? selected}
                </span>
                <span style={{ fontSize: 11, color: 'var(--text-4)', fontFamily: 'monospace' }}>
                  {selected}
                </span>
              </div>
              <span
                className="ml-2 px-2 py-0.5 rounded"
                style={{
                  fontSize: 11, fontWeight: 700,
                  background: `${stateColor}18`,
                  color: stateColor,
                  border: `1px solid ${stateColor}33`,
                }}
              >
                {selectedInfo?.connection_state ?? '—'}
              </span>
              <div className="ml-auto flex items-center gap-2">
                <button
                  onClick={() => { setStatus(null); api.getChannelStatus(selected).then(setStatus).catch(console.error) }}
                  className="p-1 rounded transition-all"
                  style={{ color: 'var(--text-4)', background: 'transparent' }}
                  onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--text-1)'; e.currentTarget.style.background = 'var(--bg-elevated)' }}
                  onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-4)'; e.currentTarget.style.background = 'transparent' }}
                  title="Refresh status"
                >
                  <RefreshCw size={13} />
                </button>
                <button
                  onClick={() => handleDeleteChannel(selected)}
                  className="p-1 rounded transition-all"
                  style={{ color: 'var(--text-4)', background: 'transparent' }}
                  onMouseEnter={(e) => { e.currentTarget.style.color = '#ff3b30'; e.currentTarget.style.background = 'rgba(255,59,48,0.08)' }}
                  onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-4)'; e.currentTarget.style.background = 'transparent' }}
                  title="Remove channel"
                >
                  <Trash2 size={13} />
                </button>
              </div>
            </div>

            {/* Tabs */}
            <div
              className="flex items-center px-6 shrink-0"
              style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-card)' }}
            >
              {(['status', 'config', 'sessions'] as Tab[]).map((tb) => (
                <button
                  key={tb}
                  className="relative px-4 py-2.5 font-medium transition-colors"
                  style={{
                    fontSize: 13,
                    color: tab === tb ? 'var(--text-1)' : 'var(--text-3)',
                    background: 'transparent',
                    border: 'none',
                  }}
                  onClick={() => setTab(tb)}
                >
                  {t(`ch.tab.${tb}`)}
                  {tab === tb && (
                    <span
                      className="absolute bottom-0 left-3 right-3 h-0.5 rounded-t-full"
                      style={{ background: 'var(--accent)' }}
                    />
                  )}
                </button>
              ))}
            </div>

            {/* Tab content */}
            {tab !== 'sessions' ? (
              <div className="flex-1 overflow-y-auto p-6">
                {tab === 'status' && (
                  <StatusTab
                    status={status}
                    channelName={selected}
                    pairingCode={pairingCode}
                    pairingError={pairingError}
                    onGeneratePairing={() => {
                      setPairingError(null)
                      api.generatePairingCode(selected)
                        .then(setPairingCode)
                        .catch((e) => setPairingError(String(e)))
                    }}
                    onRestart={() => {
                      api.restartChannel(selected)
                        .then(() => {
                          setStatus(null)
                          setTimeout(() => api.getChannelStatus(selected).then(setStatus).catch(console.error), 2000)
                          loadChannels()
                        })
                        .catch(console.error)
                    }}
                    t={t}
                  />
                )}
                {tab === 'config' && (
                  cfgData
                    ? <ChannelConfigSheet cfg={cfgData} onSaved={() => setCfgData(null)} />
                    : <p style={{ fontSize: 13, color: 'var(--text-4)' }}>Loading…</p>
                )}
              </div>
            ) : (
              <div className="flex-1 min-h-0 overflow-hidden">
                <SessionsTab sessions={sessions} t={t} />
              </div>
            )}
          </>
        )}
      </div>

      {/* ── Add Channel dialog ── */}
      {showAdd && (
        <AddChannelDialog
          onClose={() => setShowAdd(false)}
          onSaved={() => { setShowAdd(false); loadChannels() }}
          t={t}
        />
      )}
    </div>
  )
}

// ── StatusTab ─────────────────────────────────────────────────────────────────

function StatusTab({ status, channelName, pairingCode, pairingError, onGeneratePairing, onRestart, t }: {
  status: ChannelStatus | null
  channelName: string
  pairingCode: { code: string; ttl_seconds: number } | null
  pairingError: string | null
  onGeneratePairing: () => void
  onRestart: () => void
  t: (k: string) => string
}) {
  const [restarting, setRestarting] = useState(false)
  const [restartMsg, setRestartMsg] = useState<string | null>(null)

  const handleRestart = () => {
    setRestarting(true)
    setRestartMsg(null)
    onRestart()
    setTimeout(() => { setRestarting(false); setRestartMsg('Restarting…') }, 300)
    setTimeout(() => setRestartMsg(null), 4000)
  }

  return (
    <div className="flex flex-col gap-6 max-w-lg">
      {!status && <p style={{ fontSize: 13, color: 'var(--text-4)' }}>Loading…</p>}

      {status && (
        <div
          className="grid overflow-hidden rounded-xl"
          style={{ border: '1px solid var(--border)', gridTemplateColumns: '1fr 1fr 1fr', gap: '1px', background: 'var(--border)' }}
        >
          {[
            { label: t('ch.status.state'),    value: status.connection_state },
            { label: t('ch.status.queue'),    value: String(status.queue_size) },
            { label: t('ch.status.sessions'), value: String(status.active_sessions) },
          ].map(({ label, value }) => (
            <div key={label} className="flex flex-col gap-1 p-4" style={{ background: 'var(--bg-card)' }}>
              <span style={{ fontSize: 11, color: 'var(--text-4)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                {label}
              </span>
              <span style={{ fontSize: 20, fontWeight: 700, color: 'var(--text-1)', fontFamily: 'monospace' }}>
                {value}
              </span>
            </div>
          ))}
        </div>
      )}

      {/* Restart */}
      <div
        className="flex items-center gap-3 p-4 rounded-xl"
        style={{ border: '1px solid var(--border)', background: 'var(--bg-card)' }}
      >
        <button
          onClick={handleRestart}
          disabled={restarting}
          className="px-3 py-1.5 rounded-lg font-medium"
          style={{
            fontSize: 13,
            background: 'var(--bg-elevated)',
            color: 'var(--text-2)',
            border: '1px solid var(--border)',
            opacity: restarting ? 0.5 : 1,
          }}
          onMouseEnter={(e) => { if (!restarting) { e.currentTarget.style.color = 'var(--accent)'; e.currentTarget.style.borderColor = 'var(--accent-ring)' } }}
          onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-2)'; e.currentTarget.style.borderColor = 'var(--border)' }}
        >
          {restarting ? '…' : 'Restart'}
        </button>
        {restartMsg && <span style={{ fontSize: 12, color: 'var(--text-4)' }}>{restartMsg}</span>}
        <span style={{ fontSize: 12, color: 'var(--text-4)' }}>
          Reconnects the bot. Use after saving config changes.
        </span>
      </div>

      <div
        className="flex flex-col gap-3 p-4 rounded-xl"
        style={{ border: '1px solid var(--border)', background: 'var(--bg-card)' }}
      >
        <span className="font-semibold" style={{ fontSize: 13, color: 'var(--text-1)' }}>
          {t('ch.pairing.title')}
        </span>
        <button
          onClick={onGeneratePairing}
          className="px-3 py-1.5 rounded-lg font-medium w-fit"
          style={{ fontSize: 13, background: 'var(--accent-bg)', color: 'var(--accent)', border: '1px solid var(--accent-ring)' }}
          onMouseEnter={(e) => { e.currentTarget.style.opacity = '0.8' }}
          onMouseLeave={(e) => { e.currentTarget.style.opacity = '1' }}
        >
          {t('ch.pairing.generate')}
        </button>
        {pairingError && <p style={{ fontSize: 12, color: '#ff3b30' }}>{pairingError}</p>}
        {pairingCode && (
          <div className="flex items-center gap-3">
            <span
              className="px-3 py-1.5 rounded-lg font-mono font-bold"
              style={{ fontSize: 18, background: 'var(--bg-elevated)', color: 'var(--accent)', border: '1px solid var(--accent-ring)', letterSpacing: '0.15em' }}
            >
              {pairingCode.code}
            </span>
            <span style={{ fontSize: 11, color: 'var(--text-4)' }}>{t('ch.pairing.ttl')}</span>
          </div>
        )}
      </div>
    </div>
  )
}

// ── SessionsTab ───────────────────────────────────────────────────────────────

function SessionsTab({ sessions, t }: {
  sessions: GatewaySessionMeta[]
  t:        (k: string) => string
}) {
  const [active, setActive]     = useState<GatewaySessionMeta | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [loading, setLoading]   = useState(false)

  const openSession = (s: GatewaySessionMeta) => {
    if (active?.session_id === s.session_id) return
    setActive(s)
    setMessages([])
    setLoading(true)
    api.getSessionMessages(s.session_id, s.agent_id, s.project, 'im-workspaces')
      .then((r) => { setMessages(r.messages.map(toMsg)); setLoading(false) })
      .catch(() => setLoading(false))
  }

  return (
    <div className="flex h-full">
      {/* Session list */}
      <div
        className="flex flex-col overflow-y-auto shrink-0 p-2 gap-0.5"
        style={{ width: 260, borderRight: '1px solid var(--border)', background: 'var(--bg-card)' }}
      >
        {sessions.length === 0 && (
          <p className="px-2 py-3" style={{ fontSize: 12, color: 'var(--text-4)' }}>{t('ch.sessions.empty')}</p>
        )}
        {sessions.map((s) => (
          <button
            key={s.session_id}
            onClick={() => openSession(s)}
            className="w-full text-left px-3 py-2.5 rounded-lg transition-colors"
            style={{
              background: active?.session_id === s.session_id ? 'var(--accent-bg)' : 'transparent',
              border: `1px solid ${active?.session_id === s.session_id ? 'var(--accent-ring)' : 'transparent'}`,
            }}
          >
            <div
              className="truncate font-medium"
              style={{ fontSize: 12, color: active?.session_id === s.session_id ? 'var(--accent)' : 'var(--text-1)' }}
            >
              {s.first_query || s.session_id.split(':').pop()?.slice(0, 12) || s.session_id}
            </div>
            <div style={{ fontSize: 10, color: 'var(--text-4)', marginTop: 2 }}>
              {s.updated_at ? new Date(s.updated_at).toLocaleDateString() : '—'}
              {' · '}{s.run_count} run{s.run_count !== 1 ? 's' : ''}
            </div>
          </button>
        ))}
      </div>

      {/* Message preview */}
      <div className="flex-1 min-w-0 min-h-0 flex flex-col">
        {!active && (
          <div className="flex-1 flex items-center justify-center" style={{ fontSize: 13, color: 'var(--text-4)' }}>
            {t('ch.sessions.select')}
          </div>
        )}
        {active && loading && (
          <div className="flex-1 flex items-center justify-center" style={{ fontSize: 13, color: 'var(--text-4)' }}>
            Loading…
          </div>
        )}
        {active && !loading && (
          <ChatPanel messages={messages} autoScroll={false} />
        )}
      </div>
    </div>
  )
}

// ── AddChannelDialog ──────────────────────────────────────────────────────────

function AddChannelDialog({ onClose, onSaved, t }: {
  onClose: () => void
  onSaved: () => void
  t: (k: string) => string
}) {
  const [types, setTypes]         = useState<ChannelTypeInfo[]>([])
  const [loadingTypes, setLoadingTypes] = useState(true)
  const [pickedType, setPickedType]     = useState<ChannelTypeInfo | null>(null)
  const [channelName, setChannelName]   = useState('')
  const [formValues, setFormValues]     = useState<Record<string, string>>({})
  const [saving, setSaving]   = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [saved, setSaved]     = useState(false)

  useEffect(() => {
    api.listChannelTypes()
      .then((ts) => { setTypes(ts); setLoadingTypes(false) })
      .catch(() => setLoadingTypes(false))
  }, [])

  const pickType = (type: ChannelTypeInfo) => {
    if (!type.available) return
    setPickedType(type)
    setChannelName(type.name)
    setFormValues({})
    setSaveError(null)
    setSaved(false)
  }

  const save = () => {
    if (!pickedType || !channelName.trim()) return
    setSaving(true)
    setSaveError(null)
    const config: Record<string, unknown> = { ...formValues, enabled: true }
    api.createChannel(channelName.trim(), config, pickedType.name)
      .then(() => { setSaved(true); setSaving(false); setTimeout(onSaved, 1200) })
      .catch((e) => { setSaveError(String(e)); setSaving(false) })
  }

  const props = (pickedType?.schema?.properties ?? {}) as Record<string, {
    type?: string; title?: string; format?: string; default?: unknown
  }>

  return (
    <div
      className="fixed inset-0 flex items-center justify-center z-50"
      style={{ background: 'rgba(0,0,0,0.45)' }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div
        className="flex flex-col rounded-2xl overflow-hidden"
        style={{
          width: 520, maxHeight: '80vh',
          background: 'var(--bg-card)',
          border: '1px solid var(--border)',
          boxShadow: '0 20px 60px rgba(0,0,0,0.35)',
        }}
      >
        <div
          className="flex items-center justify-between px-5 py-4 shrink-0"
          style={{ borderBottom: '1px solid var(--border)' }}
        >
          <span className="font-semibold" style={{ fontSize: 15, color: 'var(--text-1)' }}>
            {t('gw.add.title')}
          </span>
          <button
            onClick={onClose}
            className="p-1 rounded-lg"
            style={{ color: 'var(--text-4)', background: 'transparent' }}
            onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--text-1)'; e.currentTarget.style.background = 'var(--bg-elevated)' }}
            onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-4)'; e.currentTarget.style.background = 'transparent' }}
          >
            <X size={15} />
          </button>
        </div>

        <div className="overflow-y-auto flex-1 p-5 flex flex-col gap-4">
          {!pickedType ? (
            <>
              <p style={{ fontSize: 13, color: 'var(--text-3)' }}>{t('gw.add.pick_type')}</p>
              {loadingTypes && <p style={{ fontSize: 12, color: 'var(--text-4)' }}>Loading…</p>}
              <div className="grid gap-2" style={{ gridTemplateColumns: '1fr 1fr' }}>
                {types.map((tp) => (
                  <button
                    key={tp.name}
                    onClick={() => pickType(tp)}
                    className="flex items-center gap-2 px-4 py-3 rounded-xl text-left transition-all"
                    style={{
                      background: tp.available ? 'var(--bg-elevated)' : 'transparent',
                      border: `1px solid ${tp.available ? 'var(--border)' : 'var(--border-sub)'}`,
                      opacity: tp.available ? 1 : 0.5,
                      cursor: tp.available ? 'pointer' : 'not-allowed',
                    }}
                    onMouseEnter={(e) => { if (tp.available) e.currentTarget.style.borderColor = 'var(--accent-ring)' }}
                    onMouseLeave={(e) => { e.currentTarget.style.borderColor = tp.available ? 'var(--border)' : 'var(--border-sub)' }}
                  >
                    <div className="flex flex-col min-w-0">
                      <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-1)' }}>{tp.display_name}</span>
                      {!tp.available && (
                        <span style={{ fontSize: 11, color: 'var(--text-4)' }}>{t('gw.add.unavailable')}</span>
                      )}
                    </div>
                  </button>
                ))}
              </div>
            </>
          ) : (
            <>
              <button
                onClick={() => setPickedType(null)}
                style={{ fontSize: 12, color: 'var(--accent)', background: 'transparent', border: 'none', textAlign: 'left', cursor: 'pointer', padding: 0 }}
              >
                ← {t('gw.add.pick_type')}
              </button>

              <div className="flex flex-col gap-1">
                <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  {t('gw.add.channel_name')}
                </label>
                <input
                  type="text"
                  value={channelName}
                  onChange={(e) => setChannelName(e.target.value)}
                  placeholder={t('gw.add.name_hint')}
                  style={{ fontSize: 13, padding: '7px 10px', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-elevated)', color: 'var(--text-1)', outline: 'none' }}
                  onFocus={(e) => { e.currentTarget.style.borderColor = 'var(--accent-ring)' }}
                  onBlur={(e) => { e.currentTarget.style.borderColor = 'var(--border)' }}
                />
              </div>

              {Object.entries(props).map(([key, def]) => (
                <div key={key} className="flex flex-col gap-1">
                  <label style={{ fontSize: 12, fontWeight: 600, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    {def.title ?? key}
                  </label>
                  <input
                    type={def.format === 'password' ? 'password' : def.type === 'integer' ? 'number' : 'text'}
                    value={formValues[key] ?? ''}
                    placeholder={String(def.default ?? '')}
                    onChange={(e) => setFormValues((prev) => ({ ...prev, [key]: e.target.value }))}
                    style={{ fontSize: 13, padding: '7px 10px', borderRadius: 8, border: '1px solid var(--border)', background: 'var(--bg-elevated)', color: 'var(--text-1)', outline: 'none' }}
                    onFocus={(e) => { e.currentTarget.style.borderColor = 'var(--accent-ring)' }}
                    onBlur={(e) => { e.currentTarget.style.borderColor = 'var(--border)' }}
                  />
                </div>
              ))}

              <div className="flex items-center gap-3 pt-1">
                <button
                  onClick={save}
                  disabled={saving || !channelName.trim()}
                  className="px-4 py-1.5 rounded-lg font-medium"
                  style={{
                    fontSize: 13,
                    background: 'var(--accent-bg)', color: 'var(--accent)',
                    border: '1px solid var(--accent-ring)',
                    opacity: (saving || !channelName.trim()) ? 0.6 : 1,
                  }}
                >
                  {saving ? '…' : t('gw.add.save')}
                </button>
                {saved && <span style={{ fontSize: 12, color: '#34c759' }}>{t('gw.add.saved')}</span>}
                {saveError && <span style={{ fontSize: 12, color: '#ff3b30' }}>{saveError}</span>}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
