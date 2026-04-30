import { useEffect, useState, useRef } from 'react'
import { RefreshCw, Plus, Play, Pause, RotateCcw, Trash2, X, ChevronDown, ChevronUp } from 'lucide-react'
import { api } from '@gw/api/client'
import { useT } from '@gw/i18n'

// ── Types ────────────────────────────────────────────────────────────────────

type ScheduleMode = 'every' | 'cron'
type CronTarget = 'last' | 'channel'
type JobStatus = 'success' | 'error' | 'running' | 'skipped' | null

interface HeartbeatConfig {
  enabled: boolean
  every: string
  cron: string
  target: CronTarget
  channel: string
  chat_id: string
  session_id: string
  timezone: string
  active_hours?: { start: string; end: string }
}

interface HeartbeatState extends HeartbeatConfig {
  next_run_at?: string | null
  last_run_at?: string | null
  last_status?: JobStatus
  last_error?: string | null
}

interface CronJobSpec {
  id?: string
  name: string
  enabled: boolean
  every: string
  cron: string
  prompt: string
  target: CronTarget
  channel: string
  chat_id: string
  session_id: string
  timezone: string
  timeout: number
}

interface CronJob extends CronJobSpec {
  id: string
  state?: {
    next_run_at?: string | null
    last_run_at?: string | null
    last_status?: JobStatus
    last_error?: string | null
  }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

const STATUS_COLOR: Record<string, string> = {
  success: '#34c759',
  error:   '#ff3b30',
  running: '#ff9f0a',
  skipped: '#8e8e93',
}

function fmtDate(iso?: string | null): string {
  if (!iso) return '—'
  try { return new Date(iso).toLocaleString() } catch { return iso }
}

const FIELD_LABEL: React.CSSProperties = {
  display: 'block',
  fontSize: 11,
  fontWeight: 600,
  color: 'var(--text-3)',
  textTransform: 'uppercase',
  letterSpacing: '0.06em',
  marginBottom: 5,
}

function InputField({
  label, value, onChange, placeholder, type = 'text', disabled,
}: {
  label: string; value: string; onChange: (v: string) => void
  placeholder?: string; type?: string; disabled?: boolean
}) {
  return (
    <div>
      <label style={FIELD_LABEL}>{label}</label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        disabled={disabled}
        style={{
          width: '100%',
          fontSize: 13,
          padding: '6px 10px',
          borderRadius: 8,
          border: '1px solid var(--border)',
          background: disabled ? 'var(--bg-base)' : 'var(--bg-elevated)',
          color: 'var(--text-1)',
          outline: 'none',
          opacity: disabled ? 0.5 : 1,
          boxSizing: 'border-box',
        }}
        onFocus={(e) => { if (!disabled) e.currentTarget.style.borderColor = 'var(--accent-ring)' }}
        onBlur={(e) => { e.currentTarget.style.borderColor = 'var(--border)' }}
      />
    </div>
  )
}

function ToggleSwitch({
  checked, onChange, label,
}: {
  checked: boolean; onChange: (v: boolean) => void; label: string
}) {
  return (
    <label className="flex items-center gap-3 cursor-pointer" style={{ userSelect: 'none' }}>
      <div
        onClick={() => onChange(!checked)}
        style={{
          width: 36, height: 20, borderRadius: 10,
          background: checked ? 'var(--accent)' : 'var(--bg-elevated)',
          border: `1px solid ${checked ? 'var(--accent)' : 'var(--border)'}`,
          position: 'relative', cursor: 'pointer', flexShrink: 0,
          transition: 'background 0.15s, border-color 0.15s',
        }}
      >
        <div style={{
          position: 'absolute', top: 2, left: checked ? 17 : 2,
          width: 14, height: 14, borderRadius: '50%',
          background: checked ? '#fff' : 'var(--text-4)',
          transition: 'left 0.15s',
        }} />
      </div>
      <span style={{ fontSize: 13, color: 'var(--text-2)' }}>{label}</span>
    </label>
  )
}

function StatusBadge({ status }: { status?: JobStatus }) {
  if (!status) return <span style={{ fontSize: 11, color: 'var(--text-4)' }}>—</span>
  return (
    <span
      style={{
        fontSize: 10, fontWeight: 600, padding: '2px 7px', borderRadius: 6,
        background: `${STATUS_COLOR[status]}22`,
        color: STATUS_COLOR[status] ?? 'var(--text-3)',
        border: `1px solid ${STATUS_COLOR[status]}44`,
        textTransform: 'uppercase', letterSpacing: '0.04em',
      }}
    >
      {status}
    </span>
  )
}

// ── Schedule Mode Picker ─────────────────────────────────────────────────────

function SchedulePicker({
  mode, onModeChange, every, onEveryChange, cron, onCronChange, t,
}: {
  mode: ScheduleMode; onModeChange: (m: ScheduleMode) => void
  every: string; onEveryChange: (v: string) => void
  cron: string; onCronChange: (v: string) => void
  t: (k: string) => string
}) {
  return (
    <div className="space-y-3">
      <label style={FIELD_LABEL}>{t('gw.cron.form.schedule')}</label>
      <div
        className="flex gap-1 rounded-lg p-0.5"
        style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', width: 'fit-content' }}
      >
        {(['every', 'cron'] as ScheduleMode[]).map((m) => (
          <button
            key={m}
            onClick={() => onModeChange(m)}
            className="px-3 py-1 rounded-md transition-colors"
            style={{
              fontSize: 12,
              background: mode === m ? 'var(--bg-card)' : 'transparent',
              color: mode === m ? 'var(--accent)' : 'var(--text-3)',
              border: mode === m ? '1px solid var(--accent-ring)' : '1px solid transparent',
              fontFamily: 'JetBrains Mono, monospace',
            }}
          >
            {m}
          </button>
        ))}
      </div>
      {mode === 'every' ? (
        <input
          type="text"
          value={every}
          onChange={(e) => onEveryChange(e.target.value)}
          placeholder="30m / 1h / 2h30m"
          style={{
            fontSize: 13, padding: '6px 10px', borderRadius: 8,
            border: '1px solid var(--border)', background: 'var(--bg-elevated)',
            color: 'var(--text-1)', outline: 'none', width: 200, boxSizing: 'border-box',
          }}
          onFocus={(e) => { e.currentTarget.style.borderColor = 'var(--accent-ring)' }}
          onBlur={(e) => { e.currentTarget.style.borderColor = 'var(--border)' }}
        />
      ) : (
        <input
          type="text"
          value={cron}
          onChange={(e) => onCronChange(e.target.value)}
          placeholder="0 9 * * *"
          style={{
            fontSize: 13, padding: '6px 10px', borderRadius: 8,
            border: '1px solid var(--border)', background: 'var(--bg-elevated)',
            color: 'var(--text-1)', outline: 'none', width: 200, fontFamily: 'JetBrains Mono, monospace',
            boxSizing: 'border-box',
          }}
          onFocus={(e) => { e.currentTarget.style.borderColor = 'var(--accent-ring)' }}
          onBlur={(e) => { e.currentTarget.style.borderColor = 'var(--border)' }}
        />
      )}
    </div>
  )
}

// ── Heartbeat Section ─────────────────────────────────────────────────────────

function HeartbeatSection({ t }: { t: (k: string) => string }) {
  const DEFAULT_CFG: HeartbeatConfig = {
    enabled: false,
    every: '1h',
    cron: '',
    target: 'last',
    channel: '',
    chat_id: '',
    session_id: 'heartbeat',
    timezone: 'UTC',
  }

  const [cfg, setCfg] = useState<HeartbeatConfig>(DEFAULT_CFG)
  const [state, setState] = useState<HeartbeatState | null>(null)
  const [schedMode, setSchedMode] = useState<ScheduleMode>('every')
  const [showActive, setShowActive] = useState(false)
  const [activeStart, setActiveStart] = useState('09:00')
  const [activeEnd, setActiveEnd] = useState('22:00')
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [loading, setLoading] = useState(true)
  const [collapsed, setCollapsed] = useState(false)

  useEffect(() => {
    setLoading(true)
    Promise.all([
      api.getHeartbeatConfig(),
      api.getHeartbeatState().catch(() => null),
    ]).then(([rawCfg, rawState]) => {
      const loaded = rawCfg as unknown as HeartbeatConfig
      setCfg({ ...DEFAULT_CFG, ...loaded })
      if (loaded.cron) setSchedMode('cron')
      if (loaded.active_hours) {
        setShowActive(true)
        setActiveStart((loaded.active_hours as { start: string; end: string }).start ?? '09:00')
        setActiveEnd((loaded.active_hours as { start: string; end: string }).end ?? '22:00')
      }
      if (rawState) setState(rawState as unknown as HeartbeatState)
    }).finally(() => setLoading(false))
  }, [])

  function up<K extends keyof HeartbeatConfig>(key: K, val: HeartbeatConfig[K]) {
    setCfg((c) => ({ ...c, [key]: val }))
    setSaved(false)
  }

  const save = () => {
    setSaving(true)
    const payload: Record<string, unknown> = {
      ...cfg,
      cron: schedMode === 'cron' ? cfg.cron : '',
      every: schedMode === 'every' ? cfg.every : '',
    }
    if (showActive) payload.active_hours = { start: activeStart, end: activeEnd }
    else delete payload.active_hours

    api.updateHeartbeatConfig(payload)
      .then(() => { setSaved(true); setSaving(false); setTimeout(() => setSaved(false), 2000) })
      .catch(() => setSaving(false))
  }

  const card: React.CSSProperties = {
    borderRadius: 16, border: '1px solid var(--border)', background: 'var(--bg-card)', overflow: 'hidden',
  }

  return (
    <div style={card}>
      {/* Header */}
      <div
        className="flex items-center justify-between px-5 py-3.5 cursor-pointer"
        style={{ borderBottom: collapsed ? 'none' : '1px solid var(--border)', background: 'var(--bg-elevated)' }}
        onClick={() => setCollapsed((c) => !c)}
      >
        <div className="flex items-center gap-3">
          <span className="font-semibold" style={{ fontSize: 14, color: 'var(--text-1)' }}>
            {t('gw.cron.heartbeat.title')}
          </span>
          {state?.last_status && <StatusBadge status={state.last_status} />}
          {state?.next_run_at && (
            <span style={{ fontSize: 11, color: 'var(--text-4)' }}>
              {t('gw.cron.heartbeat.next_run')}: {fmtDate(state.next_run_at)}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {cfg.enabled && (
            <span
              style={{
                fontSize: 10, fontWeight: 600, padding: '2px 8px', borderRadius: 6,
                background: '#34c75922', color: '#34c759', border: '1px solid #34c75944',
                textTransform: 'uppercase', letterSpacing: '0.04em',
              }}
            >
              ON
            </span>
          )}
          {collapsed ? <ChevronDown size={15} style={{ color: 'var(--text-4)' }} /> : <ChevronUp size={15} style={{ color: 'var(--text-4)' }} />}
        </div>
      </div>

      {!collapsed && (
        <div className="p-5 space-y-5">
          {loading ? (
            <p style={{ fontSize: 13, color: 'var(--text-4)' }}>Loading…</p>
          ) : (
            <>
              <p style={{ fontSize: 12.5, color: 'var(--text-4)', lineHeight: 1.6 }}>
                {t('gw.cron.heartbeat.hint')}
              </p>

              {/* Enabled toggle */}
              <ToggleSwitch
                checked={cfg.enabled}
                onChange={(v) => up('enabled', v)}
                label={t('gw.cron.heartbeat.enabled')}
              />

              {/* Schedule */}
              <SchedulePicker
                mode={schedMode}
                onModeChange={setSchedMode}
                every={cfg.every}
                onEveryChange={(v) => up('every', v)}
                cron={cfg.cron}
                onCronChange={(v) => up('cron', v)}
                t={t}
              />

              <div className="grid gap-4" style={{ gridTemplateColumns: '1fr 1fr' }}>
                {/* Target */}
                <div>
                  <label style={FIELD_LABEL}>{t('gw.cron.heartbeat.target')}</label>
                  <div
                    className="flex gap-1 rounded-lg p-0.5"
                    style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', width: 'fit-content' }}
                  >
                    {(['last', 'channel'] as CronTarget[]).map((tgt) => (
                      <button
                        key={tgt}
                        onClick={() => up('target', tgt)}
                        className="px-3 py-1 rounded-md transition-colors"
                        style={{
                          fontSize: 12,
                          background: cfg.target === tgt ? 'var(--bg-card)' : 'transparent',
                          color: cfg.target === tgt ? 'var(--accent)' : 'var(--text-3)',
                          border: cfg.target === tgt ? '1px solid var(--accent-ring)' : '1px solid transparent',
                        }}
                      >
                        {t(`gw.cron.heartbeat.target.${tgt}`)}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Timezone */}
                <InputField
                  label={t('gw.cron.heartbeat.timezone')}
                  value={cfg.timezone}
                  onChange={(v) => up('timezone', v)}
                  placeholder="UTC / Asia/Shanghai"
                />
              </div>

              {/* Channel / chat_id — only for target=channel */}
              {cfg.target === 'channel' && (
                <div className="grid gap-4" style={{ gridTemplateColumns: '1fr 1fr' }}>
                  <InputField
                    label={t('gw.cron.heartbeat.channel')}
                    value={cfg.channel}
                    onChange={(v) => up('channel', v)}
                    placeholder="telegram"
                  />
                  <InputField
                    label={t('gw.cron.heartbeat.chat_id')}
                    value={cfg.chat_id}
                    onChange={(v) => up('chat_id', v)}
                    placeholder="123456789"
                  />
                </div>
              )}

              {/* Active hours toggle */}
              <div>
                <ToggleSwitch
                  checked={showActive}
                  onChange={setShowActive}
                  label={t('gw.cron.heartbeat.active_hours')}
                />
                {showActive && (
                  <div className="flex items-center gap-3 mt-3">
                    <input
                      type="time"
                      value={activeStart}
                      onChange={(e) => setActiveStart(e.target.value)}
                      style={{
                        fontSize: 13, padding: '5px 8px', borderRadius: 8,
                        border: '1px solid var(--border)', background: 'var(--bg-elevated)',
                        color: 'var(--text-1)', outline: 'none',
                      }}
                    />
                    <span style={{ fontSize: 12, color: 'var(--text-4)' }}>→</span>
                    <input
                      type="time"
                      value={activeEnd}
                      onChange={(e) => setActiveEnd(e.target.value)}
                      style={{
                        fontSize: 13, padding: '5px 8px', borderRadius: 8,
                        border: '1px solid var(--border)', background: 'var(--bg-elevated)',
                        color: 'var(--text-1)', outline: 'none',
                      }}
                    />
                  </div>
                )}
              </div>

              {/* Last run info */}
              {state && (
                <div
                  className="flex items-center gap-5 px-4 py-2.5 rounded-xl"
                  style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)' }}
                >
                  <div>
                    <span style={{ fontSize: 10, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      {t('gw.cron.heartbeat.last_run')}
                    </span>
                    <div style={{ fontSize: 12, color: 'var(--text-2)', fontFamily: 'JetBrains Mono, monospace' }}>
                      {fmtDate(state.last_run_at)}
                    </div>
                  </div>
                  {state.last_status && (
                    <div>
                      <span style={{ fontSize: 10, color: 'var(--text-4)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        {t('gw.cron.heartbeat.status')}
                      </span>
                      <div className="mt-0.5">
                        <StatusBadge status={state.last_status} />
                      </div>
                    </div>
                  )}
                  {state.last_error && (
                    <div className="flex-1 min-w-0">
                      <span style={{ fontSize: 10, color: '#ff3b30', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                        Error
                      </span>
                      <div className="truncate" style={{ fontSize: 11, color: '#ff3b30', fontFamily: 'JetBrains Mono, monospace' }}>
                        {state.last_error}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Actions */}
              <div className="flex items-center gap-2 pt-1">
                <button
                  onClick={save}
                  disabled={saving}
                  className="px-4 py-1.5 rounded-lg font-medium transition-colors"
                  style={{
                    fontSize: 13,
                    background: 'var(--accent)', color: '#fff',
                    opacity: saving ? 0.6 : 1,
                  }}
                >
                  {saving ? '…' : t('gw.cron.heartbeat.save')}
                </button>
                {saved && <span style={{ fontSize: 12, color: '#34c759' }}>{t('gw.cron.heartbeat.saved')}</span>}
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}

// ── Job Form Dialog ───────────────────────────────────────────────────────────

const EMPTY_JOB: Omit<CronJobSpec, 'id'> = {
  name: '',
  enabled: true,
  every: '1h',
  cron: '',
  prompt: '',
  target: 'channel',
  channel: '',
  chat_id: '',
  session_id: 'cron',
  timezone: 'UTC',
  timeout: 120,
}

function JobFormDialog({
  initial, onSave, onClose, t,
}: {
  initial?: CronJob | null
  onSave: (spec: CronJobSpec) => Promise<void>
  onClose: () => void
  t: (k: string) => string
}) {
  const [form, setForm] = useState<Omit<CronJobSpec, 'id'>>(
    initial ? { ...EMPTY_JOB, ...initial } : { ...EMPTY_JOB }
  )
  const [schedMode, setSchedMode] = useState<ScheduleMode>(initial?.cron ? 'cron' : 'every')
  const [saving, setSaving] = useState(false)

  function up<K extends keyof typeof form>(key: K, val: (typeof form)[K]) {
    setForm((f) => ({ ...f, [key]: val }))
  }

  const submit = async () => {
    if (!form.name.trim() || !form.prompt.trim()) return
    setSaving(true)
    const spec: CronJobSpec = {
      ...form,
      cron: schedMode === 'cron' ? form.cron : '',
      every: schedMode === 'every' ? form.every : '',
      ...(initial?.id ? { id: initial.id } : {}),
    }
    try {
      await onSave(spec)
      onClose()
    } finally {
      setSaving(false)
    }
  }

  return (
    <div
      className="fixed inset-0 flex items-center justify-center"
      style={{ background: 'rgba(0,0,0,0.5)', zIndex: 1000 }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose() }}
    >
      <div
        className="flex flex-col"
        style={{
          width: 520, maxHeight: '85vh', borderRadius: 16,
          background: 'var(--bg-card)', border: '1px solid var(--border)',
          boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
        }}
      >
        {/* Dialog header */}
        <div
          className="flex items-center justify-between px-5 py-4 shrink-0"
          style={{ borderBottom: '1px solid var(--border)' }}
        >
          <span className="font-semibold" style={{ fontSize: 15, color: 'var(--text-1)' }}>
            {initial ? t('gw.cron.form.title_edit') : t('gw.cron.form.title_new')}
          </span>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg"
            style={{ color: 'var(--text-4)' }}
            onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--text-1)'; e.currentTarget.style.background = 'var(--bg-elevated)' }}
            onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-4)'; e.currentTarget.style.background = 'transparent' }}
          >
            <X size={15} />
          </button>
        </div>

        {/* Dialog body */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4">
          <div className="grid gap-4" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <InputField
              label={t('gw.cron.form.name')}
              value={form.name}
              onChange={(v) => up('name', v)}
              placeholder="Daily report"
            />
            <div style={{ paddingTop: 16 }}>
              <ToggleSwitch
                checked={form.enabled}
                onChange={(v) => up('enabled', v)}
                label={t('gw.cron.heartbeat.enabled')}
              />
            </div>
          </div>

          <SchedulePicker
            mode={schedMode}
            onModeChange={setSchedMode}
            every={form.every}
            onEveryChange={(v) => up('every', v)}
            cron={form.cron}
            onCronChange={(v) => up('cron', v)}
            t={t}
          />

          {/* Prompt */}
          <div>
            <label style={FIELD_LABEL}>{t('gw.cron.form.prompt')}</label>
            <textarea
              value={form.prompt}
              onChange={(e) => up('prompt', e.target.value)}
              placeholder="What should the agent do?"
              rows={3}
              style={{
                width: '100%', fontSize: 13, padding: '7px 10px', borderRadius: 8,
                border: '1px solid var(--border)', background: 'var(--bg-elevated)',
                color: 'var(--text-1)', outline: 'none', resize: 'vertical', lineHeight: 1.5,
                boxSizing: 'border-box',
              }}
              onFocus={(e) => { e.currentTarget.style.borderColor = 'var(--accent-ring)' }}
              onBlur={(e) => { e.currentTarget.style.borderColor = 'var(--border)' }}
            />
          </div>

          {/* Target */}
          <div>
            <label style={FIELD_LABEL}>{t('gw.cron.form.target')}</label>
            <div
              className="flex gap-1 rounded-lg p-0.5"
              style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', width: 'fit-content' }}
            >
              {(['last', 'channel'] as CronTarget[]).map((tgt) => (
                <button
                  key={tgt}
                  onClick={() => up('target', tgt)}
                  className="px-3 py-1 rounded-md transition-colors"
                  style={{
                    fontSize: 12,
                    background: form.target === tgt ? 'var(--bg-card)' : 'transparent',
                    color: form.target === tgt ? 'var(--accent)' : 'var(--text-3)',
                    border: form.target === tgt ? '1px solid var(--accent-ring)' : '1px solid transparent',
                  }}
                >
                  {t(`gw.cron.heartbeat.target.${tgt}`)}
                </button>
              ))}
            </div>
          </div>

          {form.target === 'channel' && (
            <div className="grid gap-4" style={{ gridTemplateColumns: '1fr 1fr' }}>
              <InputField
                label={t('gw.cron.form.channel')}
                value={form.channel}
                onChange={(v) => up('channel', v)}
                placeholder="telegram"
              />
              <InputField
                label={t('gw.cron.form.chat_id')}
                value={form.chat_id}
                onChange={(v) => up('chat_id', v)}
                placeholder="123456789"
              />
            </div>
          )}

          <div className="grid gap-4" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <InputField
              label={t('gw.cron.form.timezone')}
              value={form.timezone}
              onChange={(v) => up('timezone', v)}
              placeholder="UTC"
            />
            <InputField
              label={t('gw.cron.form.timeout')}
              value={String(form.timeout)}
              onChange={(v) => up('timeout', parseInt(v) || 120)}
              placeholder="120"
            />
          </div>
        </div>

        {/* Dialog footer */}
        <div
          className="flex items-center justify-end gap-2 px-5 py-4 shrink-0"
          style={{ borderTop: '1px solid var(--border)' }}
        >
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded-lg"
            style={{
              fontSize: 13, color: 'var(--text-2)',
              border: '1px solid var(--border)', background: 'var(--bg-elevated)',
            }}
          >
            {t('gw.cron.form.cancel')}
          </button>
          <button
            onClick={submit}
            disabled={saving || !form.name.trim() || !form.prompt.trim()}
            className="px-4 py-1.5 rounded-lg font-medium"
            style={{
              fontSize: 13,
              background: 'var(--accent)', color: '#fff',
              opacity: (saving || !form.name.trim() || !form.prompt.trim()) ? 0.5 : 1,
            }}
          >
            {saving ? '…' : t('gw.cron.form.save')}
          </button>
        </div>
      </div>
    </div>
  )
}

// ── Cron Jobs Section ─────────────────────────────────────────────────────────

function CronJobsSection({ t }: { t: (k: string) => string }) {
  const [jobs, setJobs] = useState<CronJob[]>([])
  const [loading, setLoading] = useState(true)
  const [showForm, setShowForm] = useState(false)
  const [editJob, setEditJob] = useState<CronJob | null>(null)
  const [actionMsg, setActionMsg] = useState<string | null>(null)
  const msgTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  const flash = (msg: string) => {
    setActionMsg(msg)
    if (msgTimer.current) clearTimeout(msgTimer.current)
    msgTimer.current = setTimeout(() => setActionMsg(null), 2500)
  }

  const load = () => {
    setLoading(true)
    api.listCronJobs()
      .then((list) => setJobs(list as unknown as CronJob[]))
      .catch(console.error)
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  const handleSave = async (spec: CronJobSpec) => {
    if ((spec as CronJob).id) {
      await api.updateCronJob((spec as CronJob).id!, spec as unknown as Record<string, unknown>)
      flash('Updated')
    } else {
      await api.createCronJob(spec as unknown as Record<string, unknown>)
      flash('Created')
    }
    load()
  }

  const handleDelete = async (id: string) => {
    if (!confirm(t('gw.cron.jobs.delete_confirm'))) return
    await api.deleteCronJob(id)
    flash('Deleted')
    load()
  }

  const handleRunNow = async (id: string) => {
    await api.runCronJobNow(id)
    flash('Started')
    setTimeout(load, 1200)
  }

  const handlePauseResume = async (job: CronJob) => {
    if (job.enabled) {
      await api.pauseCronJob(job.id)
    } else {
      await api.resumeCronJob(job.id)
    }
    load()
  }

  const scheduleLabel = (job: CronJob) => {
    if (job.cron) return <span style={{ fontFamily: 'JetBrains Mono, monospace' }}>{job.cron}</span>
    if (job.every) return job.every
    return '—'
  }

  return (
    <div
      style={{
        borderRadius: 16, border: '1px solid var(--border)',
        background: 'var(--bg-card)', overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-5 py-3.5"
        style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-elevated)' }}
      >
        <span className="font-semibold" style={{ fontSize: 14, color: 'var(--text-1)' }}>
          {t('gw.cron.jobs.title')}
        </span>
        <div className="flex items-center gap-2">
          {actionMsg && (
            <span style={{ fontSize: 12, color: '#34c759' }}>{actionMsg}</span>
          )}
          <button
            onClick={load}
            className="p-1.5 rounded-lg"
            style={{ color: 'var(--text-4)', background: 'transparent' }}
            title="Refresh"
            onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--text-1)'; e.currentTarget.style.background = 'var(--bg-card)' }}
            onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-4)'; e.currentTarget.style.background = 'transparent' }}
          >
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
          </button>
          <button
            onClick={() => { setEditJob(null); setShowForm(true) }}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg font-medium"
            style={{
              fontSize: 12,
              background: 'var(--accent-bg)', color: 'var(--accent)',
              border: '1px solid var(--accent-ring)',
            }}
          >
            <Plus size={12} />
            {t('gw.cron.jobs.new')}
          </button>
        </div>
      </div>

      {/* Table / empty state */}
      {!loading && jobs.length === 0 ? (
        <div className="px-5 py-10 text-center">
          <p style={{ fontSize: 13, color: 'var(--text-4)' }}>{t('gw.cron.jobs.empty')}</p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-base)' }}>
                {[
                  t('gw.cron.jobs.name'),
                  t('gw.cron.jobs.schedule'),
                  t('gw.cron.jobs.channel'),
                  t('gw.cron.jobs.last_run'),
                  t('gw.cron.jobs.status'),
                  t('gw.cron.jobs.actions'),
                ].map((h) => (
                  <th
                    key={h}
                    style={{
                      textAlign: 'left', padding: '8px 16px',
                      fontSize: 11, fontWeight: 600, color: 'var(--text-4)',
                      textTransform: 'uppercase', letterSpacing: '0.05em',
                    }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => (
                <tr
                  key={job.id}
                  style={{ borderBottom: '1px solid var(--border)' }}
                  onMouseEnter={(e) => { (e.currentTarget as HTMLTableRowElement).style.background = 'var(--bg-elevated)' }}
                  onMouseLeave={(e) => { (e.currentTarget as HTMLTableRowElement).style.background = 'transparent' }}
                >
                  <td style={{ padding: '10px 16px' }}>
                    <div className="flex items-center gap-2">
                      {!job.enabled && (
                        <span
                          style={{
                            fontSize: 9, fontWeight: 700, padding: '1px 5px', borderRadius: 4,
                            background: '#8e8e9322', color: '#8e8e93', border: '1px solid #8e8e9344',
                            textTransform: 'uppercase',
                          }}
                        >
                          paused
                        </span>
                      )}
                      <span style={{ fontSize: 13, color: 'var(--text-1)', fontWeight: 500 }}>{job.name}</span>
                    </div>
                    <span className="font-mono" style={{ fontSize: 10, color: 'var(--text-4)' }}>{job.id}</span>
                  </td>
                  <td style={{ padding: '10px 16px', fontSize: 12, color: 'var(--text-2)' }}>
                    {scheduleLabel(job)}
                  </td>
                  <td style={{ padding: '10px 16px', fontSize: 12, color: 'var(--text-3)' }}>
                    {job.channel || <span style={{ color: 'var(--text-4)' }}>—</span>}
                  </td>
                  <td style={{ padding: '10px 16px', fontSize: 11, color: 'var(--text-4)', fontFamily: 'JetBrains Mono, monospace' }}>
                    {fmtDate(job.state?.last_run_at)}
                  </td>
                  <td style={{ padding: '10px 16px' }}>
                    <StatusBadge status={job.state?.last_status} />
                  </td>
                  <td style={{ padding: '10px 16px' }}>
                    <div className="flex items-center gap-1">
                      <ActionBtn
                        icon={<Play size={11} />}
                        title={t('gw.cron.jobs.run_now')}
                        onClick={() => handleRunNow(job.id)}
                        color="var(--accent)"
                      />
                      <ActionBtn
                        icon={job.enabled ? <Pause size={11} /> : <RotateCcw size={11} />}
                        title={job.enabled ? t('gw.cron.jobs.pause') : t('gw.cron.jobs.resume')}
                        onClick={() => handlePauseResume(job)}
                      />
                      <ActionBtn
                        icon={<span style={{ fontSize: 11 }}>✎</span>}
                        title="Edit"
                        onClick={() => { setEditJob(job); setShowForm(true) }}
                      />
                      <ActionBtn
                        icon={<Trash2 size={11} />}
                        title={t('gw.cron.jobs.delete')}
                        onClick={() => handleDelete(job.id)}
                        color="#ff3b30"
                      />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {showForm && (
        <JobFormDialog
          initial={editJob}
          onSave={handleSave}
          onClose={() => { setShowForm(false); setEditJob(null) }}
          t={t}
        />
      )}
    </div>
  )
}

function ActionBtn({
  icon, title, onClick, color,
}: {
  icon: React.ReactNode; title: string; onClick: () => void; color?: string
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className="flex items-center justify-center rounded-lg"
      style={{
        width: 26, height: 26,
        color: color ?? 'var(--text-3)',
        background: 'transparent',
        border: '1px solid transparent',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = 'var(--bg-elevated)'
        e.currentTarget.style.color = color ?? 'var(--text-1)'
        e.currentTarget.style.borderColor = 'var(--border)'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = 'transparent'
        e.currentTarget.style.color = color ?? 'var(--text-3)'
        e.currentTarget.style.borderColor = 'transparent'
      }}
    >
      {icon}
    </button>
  )
}

// ── CronPage ─────────────────────────────────────────────────────────────────

export function CronPage() {
  const t = useT()

  return (
    <div className="flex-1 overflow-y-auto p-6">
      <div className="max-w-3xl space-y-6">
        <h2 className="font-semibold" style={{ fontSize: 16, color: 'var(--text-1)', marginBottom: 2 }}>
          {t('gw.cron.title')}
        </h2>

        <HeartbeatSection t={t} />
        <CronJobsSection t={t} />
      </div>
    </div>
  )
}
