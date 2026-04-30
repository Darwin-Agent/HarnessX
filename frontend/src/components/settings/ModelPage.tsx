import { useRef, useState, useEffect } from 'react'
import {
  Plus, X, ChevronDown, ChevronRight,
  Download, Upload, Eye, EyeOff, Save, GripVertical,
} from 'lucide-react'
import { useSlotsStore, SLOT_HINTS } from '../../store/slots'
import { useT } from '../../i18n'
import { modelConfigToYaml, yamlToModelConfig } from '../../lib/yaml'
import type { ModelDef, ModelCapability, ModelSlot, VendorInfo } from '../../api/types'
import { DocLink } from '../docs/DocLink'
import { api } from '../../api/client'

// ── Capability metadata ───────────────────────────────────────────────────────

interface CapDef { key: ModelCapability; label: string; color: string; bg: string }

const CAPABILITY_DEFS: CapDef[] = [
  { key: 'text',      label: 'Text',      color: '#64748b', bg: 'rgba(100,116,139,0.10)' },
  { key: 'code',      label: 'Code',      color: '#3b82f6', bg: 'rgba(59,130,246,0.10)'  },
  { key: 'omni',      label: 'Omni',      color: '#8b5cf6', bg: 'rgba(139,92,246,0.12)'  },
  { key: 'vl',        label: 'Vision',    color: '#6366f1', bg: 'rgba(99,102,241,0.10)'  },
  { key: 'tts',       label: 'TTS',       color: '#14b8a6', bg: 'rgba(20,184,166,0.10)'  },
  { key: 'asr',       label: 'ASR',       color: '#06b6d4', bg: 'rgba(6,182,212,0.10)'   },
  { key: 'embedding', label: 'Embed',     color: '#f97316', bg: 'rgba(249,115,22,0.10)'  },
  { key: 'image_gen', label: 'ImgGen',    color: '#ec4899', bg: 'rgba(236,72,153,0.10)'  },
  { key: 'video_gen', label: 'VideoGen',  color: '#ef4444', bg: 'rgba(239,68,68,0.10)'   },
]

function capDef(k: ModelCapability): CapDef {
  return CAPABILITY_DEFS.find((d) => d.key === k) ?? { key: k, label: k, color: 'var(--text-3)', bg: 'var(--bg-elevated)' }
}

function headersToText(headers?: Record<string, string>): string {
  if (!headers || Object.keys(headers).length === 0) return ''
  return Object.entries(headers).map(([k, v]) => `${k}: ${v}`).join('\n')
}

function textToHeaders(text: string): Record<string, string> {
  const raw = text.trim()
  if (!raw) return {}
  try {
    const parsed = JSON.parse(raw)
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      return Object.fromEntries(
        Object.entries(parsed).map(([k, v]) => [String(k).trim(), String(v)]).filter(([k]) => k)
      )
    }
  } catch {
    // fallback to line-based parsing
  }
  const out: Record<string, string> = {}
  for (const line of raw.split(/\r?\n/)) {
    const s = line.trim()
    if (!s) continue
    const i = s.indexOf(':')
    if (i <= 0) continue
    const k = s.slice(0, i).trim()
    const v = s.slice(i + 1).trim()
    if (k) out[k] = v
  }
  return out
}

function backendNameForVendor(vendor: string): string {
  if (vendor === 'anthropic') return 'AnthropicProvider'
  if (vendor === 'openai') return 'OpenAIProvider'
  return 'LiteLLMProvider'
}

// ── Small capability chip ────────────────────────────────────────────────────

function CapChip({ cap, onRemove, onClick, active }: {
  cap: ModelCapability
  onRemove?: () => void
  onClick?: () => void
  active?: boolean
}) {
  const d = capDef(cap)
  return (
    <span
      className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md font-mono font-medium select-none"
      style={{
        fontSize: '10.5px',
        color:      active === false ? 'var(--text-4)' : d.color,
        background: active === false ? 'transparent' : d.bg,
        border:    `1px solid ${active === false ? 'var(--border)' : d.color}22`,
        cursor: onClick ? 'pointer' : 'default',
        opacity: active === false ? 0.55 : 1,
        transition: 'all 0.12s',
      }}
      onClick={onClick}
    >
      {d.label}
      {onRemove && (
        <button
          onClick={(e) => { e.stopPropagation(); onRemove() }}
          style={{ lineHeight: 1, marginLeft: '2px', color: d.color, opacity: 0.7 }}
        >×</button>
      )}
    </span>
  )
}

// ── Capability multi-selector ─────────────────────────────────────────────────

function CapSelector({ selected, onChange }: {
  selected: ModelCapability[]
  onChange: (caps: ModelCapability[]) => void
}) {
  function toggle(k: ModelCapability) {
    onChange(
      selected.includes(k) ? selected.filter((c) => c !== k) : [...selected, k]
    )
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {CAPABILITY_DEFS.map((d) => (
        <CapChip
          key={d.key}
          cap={d.key}
          active={selected.includes(d.key)}
          onClick={() => toggle(d.key)}
        />
      ))}
    </div>
  )
}

// ── Vendor badge ──────────────────────────────────────────────────────────────

const VENDOR_COLORS: Record<string, string> = {
  anthropic: '#c96442',
  openai:    '#10a37f',
  gemini:    '#4285f4',
  deepseek:  '#5b6cf6',
  custom:    '#8b5cf6',
}

function VendorBadge({ vendor }: { vendor: string }) {
  const color = VENDOR_COLORS[vendor] ?? '#8b5cf6'
  return (
    <span
      className="inline-block px-2 py-0.5 rounded-md font-mono font-medium"
      style={{ fontSize: '11px', background: `${color}18`, color, border: `1px solid ${color}30` }}
    >
      {vendor}
    </span>
  )
}

// ── ModelDef card ─────────────────────────────────────────────────────────────

interface ModelDefCardProps {
  model:    ModelDef
  vendors:  VendorInfo[]
  onUpdate: (updates: Partial<ModelDef>) => void
  onRemove: () => void
}

function ModelDefCard({ model, vendors, onUpdate, onRemove }: ModelDefCardProps) {
  const t = useT()
  const [open, setOpen]       = useState(false)
  const [showKey, setShowKey] = useState(false)
  const [headersText, setHeadersText] = useState(headersToText(model.extra_headers))

  const currentVendor = vendors.find((v) => v.id === model.vendor)
  const isAnthropic = model.vendor === 'anthropic'
  const backendName = backendNameForVendor(model.vendor)

  useEffect(() => {
    setHeadersText(headersToText(model.extra_headers))
  }, [model.extra_headers])

  const fieldLabel: React.CSSProperties = {
    fontSize: '11px', color: 'var(--text-3)',
    fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.06em',
    textTransform: 'uppercase', display: 'block', marginBottom: '6px',
  }

  function handleVendorChange(id: string) {
    if (id === 'anthropic') {
      onUpdate({
        vendor: id,
        reasoning_effort: undefined,
        reasoning_summary: undefined,
      })
      return
    }
    onUpdate({ vendor: id })
  }

  return (
    <div
      className="rounded-2xl overflow-hidden transition-shadow"
      style={{
        border: '1px solid var(--border)',
        background: 'var(--bg-card)',
        boxShadow: open ? 'var(--shadow-card-hover)' : 'var(--shadow-card)',
      }}
    >
      {/* ── Header row ── */}
      <div
        className="flex items-center gap-3 px-4 py-3 cursor-pointer select-none"
        onClick={() => setOpen((o) => !o)}
        style={{ borderBottom: open ? '1px solid var(--border)' : 'none' }}
      >
        <VendorBadge vendor={model.vendor} />
        <span
          className="inline-block px-2 py-0.5 rounded-md font-mono"
          style={{ fontSize: '10px', color: 'var(--text-3)', border: '1px solid var(--border)', background: 'var(--bg-elevated)' }}
          title={t('model.backend_impl')}
        >
          {backendName}
        </span>

        <span className="font-semibold truncate" style={{ fontSize: '14px', color: 'var(--text-1)', flex: '1 1 0' }}>
          {model.display_name}
        </span>

        <span className="font-mono truncate" style={{ fontSize: '12px', color: 'var(--text-3)', flex: '1 1 0' }}>
          {model.model_id || <span style={{ fontStyle: 'italic' }}>no model</span>}
        </span>

        {/* Capability chips (collapsed view) */}
        <div className="flex flex-wrap gap-1 shrink-0">
          {model.capabilities.slice(0, 4).map((c) => <CapChip key={c} cap={c} />)}
          {model.capabilities.length > 4 && (
            <span style={{ fontSize: '11px', color: 'var(--text-4)' }}>+{model.capabilities.length - 4}</span>
          )}
        </div>

        <button
          onClick={(e) => { e.stopPropagation(); onRemove() }}
          className="p-1 rounded-lg transition-colors shrink-0"
          style={{ color: 'var(--text-4)' }}
          onMouseEnter={(e) => (e.currentTarget.style.color = '#ef4444')}
          onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-4)')}
          title={t('model.remove')}
        ><X size={13} /></button>

        <span style={{ color: 'var(--text-4)', flexShrink: 0 }}>
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </span>
      </div>

      {/* ── Expanded body ── */}
      {open && (
        <div className="px-4 py-4" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '20px' }}>

          {/* Col 1: identity */}
          <div className="space-y-4">
            <div>
              <label style={fieldLabel}>{t('model.display_name')}</label>
              <input
                type="text"
                value={model.display_name}
                onChange={(e) => onUpdate({ display_name: e.target.value })}
                className="w-full rounded-xl px-3 py-2"
                style={{ fontSize: '13px' }}
              />
            </div>

            <div>
              <label style={fieldLabel}>{t('model.vendor')}</label>
              <div className="flex flex-wrap gap-1.5">
                {vendors.map((v) => (
                  <button
                    key={v.id}
                    onClick={() => handleVendorChange(v.id)}
                    className="px-2.5 py-1 rounded-lg font-mono transition-all text-xs"
                    style={{
                      ...(model.vendor === v.id
                        ? { background: 'var(--accent-bg)', color: 'var(--accent)', border: '1px solid var(--accent-ring)' }
                        : { background: 'var(--bg-elevated)', color: 'var(--text-2)', border: '1px solid var(--border)' }),
                    }}
                  >{v.id}</button>
                ))}
              </div>
            </div>

            <div>
              <label style={fieldLabel}>{t('model.model_id')}</label>
              <input
                type="text"
                value={model.model_id}
                onChange={(e) => onUpdate({ model_id: e.target.value })}
                placeholder="e.g. claude-sonnet-4-6"
                className="w-full rounded-xl px-3 py-2"
                style={{ fontSize: '13px' }}
              />
            </div>
          </div>

          {/* Col 2: credentials */}
          <div className="space-y-4">
            <div>
              <label style={fieldLabel}>{t('vendor.api_key')}</label>
              <div className="flex gap-2">
                <input
                  type={showKey ? 'text' : 'password'}
                  value={model.api_key}
                  onChange={(e) => onUpdate({ api_key: e.target.value })}
                  placeholder={currentVendor?.env_key ?? 'API_KEY'}
                  className="flex-1 rounded-xl px-3 py-2" style={{ fontSize: '13px' }}
                />
                <button
                  onClick={() => setShowKey((o) => !o)}
                  className="px-2 rounded-xl transition-colors shrink-0"
                  style={{ color: showKey ? 'var(--accent)' : 'var(--text-3)', border: '1px solid var(--border)' }}
                  title={showKey ? 'Hide' : 'Show'}
                >
                  {showKey ? <EyeOff size={13} /> : <Eye size={13} />}
                </button>
              </div>
              <p style={{ fontSize: '11px', color: 'var(--text-4)', marginTop: '4px' }}>{t('vendor.api_key_hint')}</p>
            </div>

            <div>
              <label style={fieldLabel}>{t('vendor.base_url')}</label>
              <input
                type="text"
                value={model.base_url}
                onChange={(e) => onUpdate({ base_url: e.target.value })}
                placeholder={currentVendor?.default_base_url ?? 'Vendor default'}
                className="w-full rounded-xl px-3 py-2" style={{ fontSize: '13px' }}
              />
              <p style={{ fontSize: '11px', color: 'var(--text-4)', marginTop: '4px' }}>{t('vendor.base_url_hint')}</p>
            </div>

            <div>
              <label style={fieldLabel}>{t('model.extra_headers')}</label>
              <textarea
                value={headersText}
                onChange={(e) => setHeadersText(e.target.value)}
                onBlur={() => onUpdate({ extra_headers: textToHeaders(headersText) })}
                placeholder="X-Org: demo\nX-Trace: 1"
                rows={3}
                className="w-full rounded-xl px-3 py-2 resize-y font-mono"
                style={{ fontSize: '12px' }}
              />
              <p style={{ fontSize: '11px', color: 'var(--text-4)', marginTop: '4px' }}>{t('model.extra_headers_hint')}</p>
            </div>
          </div>

          {/* Col 3: capabilities */}
          <div>
            <label style={fieldLabel}>{t('model.capabilities')}</label>
            <CapSelector
              selected={model.capabilities}
              onChange={(caps) => onUpdate({ capabilities: caps })}
            />
            {model.capabilities.length === 0 && (
              <p style={{ fontSize: '12px', color: 'var(--text-4)', marginTop: '6px' }}>
                Click chips to select
              </p>
            )}

            <div className="mt-4 pt-3" style={{ borderTop: '1px dashed var(--border)' }}>
              <label style={fieldLabel}>{t('model.reasoning')}</label>

              {isAnthropic ? (
                <div className="space-y-2.5">
                  <label className="flex items-center gap-2" style={{ fontSize: '12px', color: 'var(--text-2)' }}>
                    <input
                      type="checkbox"
                      checked={!!model.extended_thinking}
                      onChange={(e) => onUpdate({ extended_thinking: e.target.checked })}
                    />
                    {t('model.extended_thinking')}
                  </label>

                  <div>
                    <div style={{ fontSize: '11px', color: 'var(--text-3)', marginBottom: '4px' }}>
                      {t('model.thinking_budget_tokens')}
                    </div>
                    <input
                      type="number"
                      min={1024}
                      max={200000}
                      step={256}
                      disabled={!model.extended_thinking}
                      value={model.thinking_budget_tokens ?? 10000}
                      onChange={(e) => onUpdate({
                        thinking_budget_tokens: Number.isFinite(Number(e.target.value))
                          ? Number(e.target.value)
                          : 10000,
                      })}
                      className="w-full rounded-xl px-3 py-2 disabled:opacity-50"
                      style={{ fontSize: '13px' }}
                    />
                  </div>
                </div>
              ) : (
                <div>
                  <div style={{ fontSize: '11px', color: 'var(--text-3)', marginBottom: '4px' }}>
                    {t('model.reasoning_effort')}
                  </div>
                  <select
                    value={model.reasoning_effort ?? ''}
                    onChange={(e) => onUpdate({
                      reasoning_effort: e.target.value
                        ? (e.target.value as 'low' | 'medium' | 'high')
                        : undefined,
                    })}
                    className="w-full rounded-xl px-3 py-2"
                    style={{ fontSize: '13px' }}
                  >
                    <option value="">{t('model.reasoning_effort.auto')}</option>
                    <option value="low">{t('model.reasoning_effort.low')}</option>
                    <option value="medium">{t('model.reasoning_effort.medium')}</option>
                    <option value="high">{t('model.reasoning_effort.high')}</option>
                  </select>

                  <label className="mt-2.5 flex items-center gap-2" style={{ fontSize: '12px', color: 'var(--text-2)' }}>
                    <input
                      type="checkbox"
                      checked={!!model.reasoning_summary}
                      onChange={(e) => onUpdate({ reasoning_summary: e.target.checked || undefined })}
                    />
                    {t('model.reasoning_summary')}
                  </label>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// ── Add model inline form ────────────────────────────────────────────────────

interface AddModelFormProps {
  vendors:  VendorInfo[]
  onAdd:    (m: Omit<ModelDef, 'id'>) => void
  onCancel: () => void
}

function AddModelForm({ vendors, onAdd, onCancel }: AddModelFormProps) {
  const t = useT()
  const [vendor, setVendor]         = useState(vendors[0]?.id ?? 'anthropic')
  const [modelId, setModelId]       = useState('')
  const [displayName, setDisplayName] = useState('')
  const [apiKey, setApiKey]         = useState('')
  const [baseUrl, setBaseUrl]       = useState('')
  const [extraHeadersText, setExtraHeadersText] = useState('')
  const [caps, setCaps]             = useState<ModelCapability[]>(['text'])
  const [extendedThinking, setExtendedThinking] = useState(false)
  const [thinkingBudgetTokens, setThinkingBudgetTokens] = useState(10000)
  const [reasoningEffort, setReasoningEffort] = useState<'' | 'low' | 'medium' | 'high'>('')
  const [reasoningSummary, setReasoningSummary] = useState(false)

  const currentVendor = vendors.find((v) => v.id === vendor)

  function handleSubmit() {
    if (!modelId.trim()) return
    onAdd({
      display_name: displayName.trim() || modelId.trim(),
      vendor,
      model_id: modelId.trim(),
      api_key: apiKey,
      base_url: baseUrl,
      extra_headers: textToHeaders(extraHeadersText),
      capabilities: caps,
      extended_thinking: vendor === 'anthropic' ? extendedThinking : undefined,
      thinking_budget_tokens: vendor === 'anthropic' ? thinkingBudgetTokens : undefined,
      reasoning_effort: vendor !== 'anthropic' && reasoningEffort ? reasoningEffort : undefined,
      reasoning_summary: vendor !== 'anthropic' ? reasoningSummary : undefined,
    })
  }

  const fieldLabel: React.CSSProperties = {
    fontSize: '11px', color: 'var(--text-3)',
    fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.06em',
    textTransform: 'uppercase', display: 'block', marginBottom: '6px',
  }

  return (
    <div
      className="rounded-2xl p-5"
      style={{ border: '1px solid var(--accent-ring)', background: 'var(--accent-bg)' }}
    >
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '20px' }}>

        {/* Col 1 */}
        <div className="space-y-4">
          <div>
            <label style={fieldLabel}>{t('model.vendor')}</label>
            <div className="flex flex-wrap gap-1.5">
              {vendors.map((v) => (
                <button key={v.id} onClick={() => setVendor(v.id)}
                  className="px-2.5 py-1 rounded-lg font-mono text-xs transition-all"
                  style={{
                    ...(vendor === v.id
                      ? { background: 'var(--accent-bg)', color: 'var(--accent)', border: '1px solid var(--accent-ring)' }
                      : { background: 'var(--bg-card)', color: 'var(--text-2)', border: '1px solid var(--border)' }),
                  }}
                >{v.id}</button>
              ))}
            </div>
          </div>

          <div>
            <label style={fieldLabel}>{t('model.model_id')}</label>
            <input autoFocus type="text" value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              placeholder="e.g. claude-sonnet-4-6"
              className="w-full rounded-xl px-3 py-2" style={{ fontSize: '13px' }}
            />
          </div>

          <div>
            <label style={fieldLabel}>{t('model.display_name')}</label>
            <input type="text" value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder={modelId || 'e.g. My Sonnet'}
              className="w-full rounded-xl px-3 py-2" style={{ fontSize: '13px' }}
            />
          </div>
        </div>

        {/* Col 2 */}
        <div className="space-y-4">
          <div>
            <label style={fieldLabel}>{t('vendor.api_key')}</label>
            <input type="password" value={apiKey} onChange={(e) => setApiKey(e.target.value)}
              placeholder={currentVendor?.env_key ?? 'API_KEY'}
              className="w-full rounded-xl px-3 py-2" style={{ fontSize: '13px' }}
            />
            <p style={{ fontSize: '11px', color: 'var(--text-4)', marginTop: '4px' }}>{t('vendor.api_key_hint')}</p>
          </div>
          <div>
            <label style={fieldLabel}>{t('vendor.base_url')}</label>
            <input type="text" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)}
              placeholder={currentVendor?.default_base_url ?? 'Vendor default'}
              className="w-full rounded-xl px-3 py-2" style={{ fontSize: '13px' }}
            />
          </div>
          <div>
            <label style={fieldLabel}>{t('model.extra_headers')}</label>
            <textarea
              value={extraHeadersText}
              onChange={(e) => setExtraHeadersText(e.target.value)}
              rows={3}
              placeholder="X-Org: demo"
              className="w-full rounded-xl px-3 py-2 resize-y font-mono"
              style={{ fontSize: '12px' }}
            />
          </div>
        </div>

        {/* Col 3 */}
        <div>
          <label style={fieldLabel}>{t('model.capabilities')}</label>
          <CapSelector selected={caps} onChange={setCaps} />

          <div className="mt-4 pt-3" style={{ borderTop: '1px dashed var(--border)' }}>
            <label style={fieldLabel}>{t('model.reasoning')}</label>
            {vendor === 'anthropic' ? (
              <div className="space-y-2.5">
                <label className="flex items-center gap-2" style={{ fontSize: '12px', color: 'var(--text-2)' }}>
                  <input
                    type="checkbox"
                    checked={extendedThinking}
                    onChange={(e) => setExtendedThinking(e.target.checked)}
                  />
                  {t('model.extended_thinking')}
                </label>
                <div>
                  <div style={{ fontSize: '11px', color: 'var(--text-3)', marginBottom: '4px' }}>
                    {t('model.thinking_budget_tokens')}
                  </div>
                  <input
                    type="number"
                    min={1024}
                    max={200000}
                    step={256}
                    disabled={!extendedThinking}
                    value={thinkingBudgetTokens}
                    onChange={(e) => setThinkingBudgetTokens(Number(e.target.value) || 10000)}
                    className="w-full rounded-xl px-3 py-2 disabled:opacity-50"
                    style={{ fontSize: '13px' }}
                  />
                </div>
              </div>
            ) : (
              <div>
                <div style={{ fontSize: '11px', color: 'var(--text-3)', marginBottom: '4px' }}>
                  {t('model.reasoning_effort')}
                </div>
                <select
                  value={reasoningEffort}
                  onChange={(e) => setReasoningEffort(e.target.value as '' | 'low' | 'medium' | 'high')}
                  className="w-full rounded-xl px-3 py-2"
                  style={{ fontSize: '13px' }}
                >
                  <option value="">{t('model.reasoning_effort.auto')}</option>
                  <option value="low">{t('model.reasoning_effort.low')}</option>
                  <option value="medium">{t('model.reasoning_effort.medium')}</option>
                  <option value="high">{t('model.reasoning_effort.high')}</option>
                </select>

                <label className="mt-2.5 flex items-center gap-2" style={{ fontSize: '12px', color: 'var(--text-2)' }}>
                  <input
                    type="checkbox"
                    checked={reasoningSummary}
                    onChange={(e) => setReasoningSummary(e.target.checked)}
                  />
                  {t('model.reasoning_summary')}
                </label>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="flex justify-end gap-2 mt-5">
        <button onClick={onCancel}
          className="px-4 py-2 rounded-xl text-sm"
          style={{ color: 'var(--text-3)', border: '1px solid var(--border)' }}
        >{t('fs.cancel')}</button>
        <button onClick={handleSubmit}
          disabled={!modelId.trim()}
          className="px-4 py-2 rounded-xl text-sm font-semibold disabled:opacity-40"
          style={{ background: 'var(--accent)', color: '#fff' }}
        >{t('model.add')}</button>
      </div>
    </div>
  )
}

// ── Provider-group card (nested models) ─────────────────────────────────────

interface ProviderGroupCardProps {
  vendor: string
  models: ModelDef[]
  vendors: VendorInfo[]
  onUpdateModel: (id: string, updates: Partial<ModelDef>) => void
  onRemoveModel: (id: string) => void
  onQuickAddModel: (m: Omit<ModelDef, 'id'>) => void
}

function ProviderGroupCard({
  vendor,
  models,
  vendors,
  onUpdateModel,
  onRemoveModel,
  onQuickAddModel,
}: ProviderGroupCardProps) {
  const t = useT()
  const [open, setOpen] = useState(true)
  const [quickAddOpen, setQuickAddOpen] = useState(false)
  const [quickModelId, setQuickModelId] = useState('')
  const [quickDisplayName, setQuickDisplayName] = useState('')

  const modelCount = models.length
  const template = models.find((m) => !!m.api_key?.trim() || !!m.base_url?.trim()) ?? models[0]

  function handleQuickAdd() {
    const modelId = quickModelId.trim()
    if (!modelId) return

    onQuickAddModel({
      display_name: quickDisplayName.trim() || modelId,
      vendor,
      model_id: modelId,
      api_key: template?.api_key ?? '',
      base_url: template?.base_url ?? '',
      extra_headers: template?.extra_headers ?? {},
      capabilities: template?.capabilities?.length ? template.capabilities : ['text'],
      extended_thinking: vendor === 'anthropic' ? (template?.extended_thinking ?? false) : undefined,
      thinking_budget_tokens: vendor === 'anthropic' ? (template?.thinking_budget_tokens ?? 10000) : undefined,
      reasoning_effort: vendor !== 'anthropic' ? template?.reasoning_effort : undefined,
      reasoning_summary: vendor !== 'anthropic' ? template?.reasoning_summary : undefined,
    })

    setQuickModelId('')
    setQuickDisplayName('')
    setQuickAddOpen(false)
  }

  return (
    <div
      className="rounded-2xl overflow-hidden"
      style={{ border: '1px solid var(--border)', background: 'var(--bg-card)' }}
    >
      <div
        className="flex items-center gap-3 px-4 py-3 cursor-pointer select-none"
        style={{ borderBottom: open ? '1px solid var(--border)' : 'none' }}
        onClick={() => setOpen((v) => !v)}
      >
        <VendorBadge vendor={vendor} />
        <span className="font-semibold" style={{ fontSize: '14px', color: 'var(--text-1)' }}>
          {vendor}
        </span>
        <span
          className="px-2 py-0.5 rounded-md font-mono"
          style={{ fontSize: '11px', color: 'var(--text-3)', background: 'var(--bg-elevated)', border: '1px solid var(--border)' }}
        >
          {modelCount} {t('model.provider_models')}
        </span>
        <div className="flex-1" />
        <button
          onClick={(e) => {
            e.stopPropagation()
            setQuickAddOpen((v) => !v)
            setOpen(true)
          }}
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs transition-all"
          style={{ color: 'var(--text-2)', border: '1px dashed var(--border)', background: 'var(--bg-elevated)' }}
        >
          <Plus size={12} />
          {t('model.provider_add')}
        </button>
        <span style={{ color: 'var(--text-4)', flexShrink: 0 }}>
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        </span>
      </div>

      {open && (
        <div className="p-3 space-y-3">
          {quickAddOpen && (
            <div
              className="rounded-xl p-3"
              style={{ border: '1px solid var(--accent-ring)', background: 'var(--accent-bg)' }}
            >
              <div className="grid gap-2" style={{ gridTemplateColumns: '1fr 1fr auto auto' }}>
                <input
                  type="text"
                  value={quickDisplayName}
                  onChange={(e) => setQuickDisplayName(e.target.value)}
                  placeholder={t('model.display_name')}
                  className="rounded-xl px-3 py-2"
                  style={{ fontSize: '13px' }}
                />
                <input
                  autoFocus
                  type="text"
                  value={quickModelId}
                  onChange={(e) => setQuickModelId(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleQuickAdd()
                    if (e.key === 'Escape') setQuickAddOpen(false)
                  }}
                  placeholder={t('model.model_id')}
                  className="rounded-xl px-3 py-2 font-mono"
                  style={{ fontSize: '13px' }}
                />
                <button
                  onClick={handleQuickAdd}
                  disabled={!quickModelId.trim()}
                  className="px-3 py-2 rounded-xl text-xs font-semibold disabled:opacity-40"
                  style={{ background: 'var(--accent)', color: '#fff' }}
                >
                  {t('model.add')}
                </button>
                <button
                  onClick={() => setQuickAddOpen(false)}
                  className="px-3 py-2 rounded-xl text-xs"
                  style={{ color: 'var(--text-3)', border: '1px solid var(--border)' }}
                >
                  {t('fs.cancel')}
                </button>
              </div>
              <p style={{ fontSize: '11px', color: 'var(--text-3)', marginTop: '6px' }}>
                {t('model.provider_add_hint')}
              </p>
            </div>
          )}

          <div className="space-y-2">
            {models.map((m) => (
              <ModelDefCard
                key={m.id}
                model={m}
                vendors={vendors}
                onUpdate={(updates) => onUpdateModel(m.id, updates)}
                onRemove={() => onRemoveModel(m.id)}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Slot card ────────────────────────────────────────────────────────────────

interface SlotCardProps {
  slot:     ModelSlot
  registry: ModelDef[]
  isFixed:  boolean     // main slot cannot be deleted
  onUpdate: (s: ModelSlot) => void
  onRemove: () => void
}

function SlotCard({ slot, registry, isFixed, onUpdate, onRemove }: SlotCardProps) {
  const t = useT()
  const [nameEditing, setNameEditing] = useState(false)
  const [nameVal, setNameVal]         = useState(slot.slot_name)
  const [pickerOpen, setPickerOpen]   = useState(false)
  const [dragModelId, setDragModelId] = useState<string | null>(null)
  const [dragOverModelId, setDragOverModelId] = useState<string | null>(null)

  const hint = SLOT_HINTS[slot.slot_name]
  const assignedModels = slot.model_ids
    .map((id) => registry.find((m) => m.id === id))
    .filter(Boolean) as ModelDef[]

  const availableToAdd = registry.filter((m) => !slot.model_ids.includes(m.id))

  function addModel(id: string) {
    onUpdate({ ...slot, model_ids: [...slot.model_ids, id] })
    setPickerOpen(false)
  }

  function removeModel(id: string) {
    onUpdate({ ...slot, model_ids: slot.model_ids.filter((mid) => mid !== id) })
  }

  function setPrimaryModel(id: string) {
    if (slot.model_ids[0] === id) return
    onUpdate({
      ...slot,
      model_ids: [id, ...slot.model_ids.filter((mid) => mid !== id)],
    })
  }

  function moveModelBefore(sourceId: string, targetId: string) {
    if (!sourceId || !targetId || sourceId === targetId) return
    const ids = [...slot.model_ids]
    const from = ids.indexOf(sourceId)
    const to = ids.indexOf(targetId)
    if (from < 0 || to < 0) return
    ids.splice(from, 1)
    ids.splice(to, 0, sourceId)
    onUpdate({ ...slot, model_ids: ids })
  }

  function commitName() {
    if (nameVal.trim() && nameVal !== slot.slot_name) {
      onUpdate({ ...slot, slot_name: nameVal.trim() })
    }
    setNameEditing(false)
  }

  const STRATEGIES: ModelSlot['strategy'][] = ['primary', 'fallback', 'round_robin']

  return (
    <div
      className="rounded-2xl p-4"
      style={{ border: `1px solid ${isFixed ? 'var(--accent-ring)' : 'var(--border)'}`, background: 'var(--bg-card)' }}
    >
      <div className="flex items-start gap-3">
        {/* Slot name */}
        <div className="shrink-0">
          {!nameEditing || isFixed ? (
            <div className="flex items-center gap-2">
              <span
                className="font-mono font-bold px-2.5 py-1 rounded-xl"
                style={{
                  fontSize: '13px',
                  ...(isFixed
                    ? { background: 'var(--accent-bg)', color: 'var(--accent)', border: '1px solid var(--accent-ring)' }
                    : { background: 'var(--bg-elevated)', color: 'var(--text-2)', border: '1px solid var(--border)' }),
                }}
                onDoubleClick={() => { if (!isFixed) { setNameVal(slot.slot_name); setNameEditing(true) } }}
              >
                {slot.slot_name}
              </span>
              {isFixed && (
                <span className="px-1.5 py-0.5 rounded-md font-mono"
                  style={{ fontSize: '10px', background: 'rgba(16,185,129,0.08)', color: '#10b981', border: '1px solid rgba(16,185,129,0.2)' }}
                >{t('slot.required')}</span>
              )}
            </div>
          ) : (
            <input autoFocus value={nameVal}
              onChange={(e) => setNameVal(e.target.value)}
              onBlur={commitName}
              onKeyDown={(e) => { if (e.key === 'Enter') commitName(); if (e.key === 'Escape') setNameEditing(false) }}
              className="rounded-xl px-2.5 py-1 font-mono font-bold"
              style={{ fontSize: '13px', width: '120px', border: '1px solid var(--accent-ring)' }}
            />
          )}
        </div>

        <div className="flex-1 min-w-0 space-y-3">
          {hint && (
            <p style={{ fontSize: '12px', color: 'var(--text-3)' }}>{hint}</p>
          )}

          {/* Assigned models row */}
          <div className="flex flex-wrap items-center gap-2">
            {assignedModels.length === 0 && (
              slot.slot_name !== 'main' ? (
                <span
                  className="flex items-center gap-1.5 px-2.5 py-1 rounded-xl"
                  style={{ fontSize: '12px', color: 'var(--text-3)', fontStyle: 'italic', border: '1px dashed var(--border)', background: 'var(--bg-elevated)' }}
                >
                  <span style={{ opacity: 0.6 }}>↩</span> uses main
                </span>
              ) : (
                <span style={{ fontSize: '12px', color: 'var(--text-4)', fontStyle: 'italic' }}>
                  {t('slot.no_models')}
                </span>
              )
            )}
            {assignedModels.map((m, i) => (
              <span
                key={m.id}
                draggable={assignedModels.length > 1}
                onDragStart={(e) => {
                  if (assignedModels.length <= 1) return
                  setDragModelId(m.id)
                  e.dataTransfer.effectAllowed = 'move'
                  e.dataTransfer.setData('text/plain', m.id)
                }}
                onDragOver={(e) => {
                  if (assignedModels.length <= 1) return
                  e.preventDefault()
                  e.dataTransfer.dropEffect = 'move'
                  if (dragOverModelId !== m.id) setDragOverModelId(m.id)
                }}
                onDragLeave={() => {
                  if (dragOverModelId === m.id) setDragOverModelId(null)
                }}
                onDrop={(e) => {
                  if (assignedModels.length <= 1) return
                  e.preventDefault()
                  const sourceId = e.dataTransfer.getData('text/plain') || dragModelId || ''
                  moveModelBefore(sourceId, m.id)
                  setDragModelId(null)
                  setDragOverModelId(null)
                }}
                onDragEnd={() => {
                  setDragModelId(null)
                  setDragOverModelId(null)
                }}
                className="flex items-center gap-1.5 px-2.5 py-1 rounded-xl"
                style={{
                  fontSize: '12px',
                  border: `1px solid ${dragOverModelId === m.id ? 'var(--accent-ring)' : 'var(--border)'}`,
                  background: i === 0 ? 'var(--accent-bg)' : 'var(--bg-elevated)',
                  color: i === 0 ? 'var(--accent)' : 'var(--text-2)',
                  cursor: assignedModels.length > 1 ? 'grab' : 'default',
                }}
              >
                {assignedModels.length > 1 && (
                  <span style={{ color: 'var(--text-4)', display: 'inline-flex' }} title={t('slot.drag_reorder')}>
                    <GripVertical size={11} />
                  </span>
                )}
                <VendorBadge vendor={m.vendor} />
                <span className="font-medium">{m.display_name}</span>
                {i === 0 ? (
                  <span
                    className="px-1.5 py-0.5 rounded-md font-mono"
                    style={{
                      fontSize: '10px',
                      background: 'rgba(59,130,246,0.10)',
                      color: '#3b82f6',
                      border: '1px solid rgba(59,130,246,0.25)',
                    }}
                  >
                    {t('slot.default')}
                  </span>
                ) : (
                  <button
                    onClick={() => setPrimaryModel(m.id)}
                    className="px-1.5 py-0.5 rounded-md font-mono transition-all"
                    style={{
                      fontSize: '10px',
                      color: 'var(--text-3)',
                      border: '1px solid var(--border)',
                      background: 'var(--bg-card)',
                    }}
                    onMouseEnter={(e) => {
                      e.currentTarget.style.color = 'var(--accent)'
                      e.currentTarget.style.borderColor = 'var(--accent-ring)'
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.color = 'var(--text-3)'
                      e.currentTarget.style.borderColor = 'var(--border)'
                    }}
                  >
                    {t('slot.set_default')}
                  </button>
                )}
                {m.capabilities.slice(0, 2).map((c) => <CapChip key={c} cap={c} />)}
                {i > 0 && (
                  <button onClick={() => removeModel(m.id)}
                    className="ml-0.5 transition-colors"
                    style={{ color: 'var(--text-4)' }}
                    onMouseEnter={(e) => (e.currentTarget.style.color = '#ef4444')}
                    onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-4)')}
                  ><X size={11} /></button>
                )}
                {/* Allow removing even first model if not the only slot */}
                {i === 0 && !isFixed && (
                  <button onClick={() => removeModel(m.id)}
                    className="ml-0.5 transition-colors"
                    style={{ color: 'var(--text-4)' }}
                    onMouseEnter={(e) => (e.currentTarget.style.color = '#ef4444')}
                    onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-4)')}
                  ><X size={11} /></button>
                )}
              </span>
            ))}

            {/* Add model picker */}
            {availableToAdd.length > 0 && (
              <div className="relative">
                <button
                  onClick={() => setPickerOpen((o) => !o)}
                  className="flex items-center gap-1 px-2 py-1 rounded-xl transition-all text-xs"
                  style={{ border: '1px dashed var(--border)', color: 'var(--text-3)' }}
                  onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'var(--accent-ring)'; e.currentTarget.style.color = 'var(--accent)' }}
                  onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text-3)' }}
                >
                  <Plus size={11} />{t('slot.add_model')}
                </button>
                {pickerOpen && (
                  <div
                    className="absolute left-0 top-full mt-1 rounded-xl shadow-lg z-20 py-1"
                    style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', minWidth: '200px' }}
                  >
                    {availableToAdd.map((m) => (
                      <button key={m.id} onClick={() => addModel(m.id)}
                        className="w-full flex items-center gap-2 px-3 py-2 text-left transition-colors"
                        style={{ fontSize: '13px', color: 'var(--text-1)' }}
                        onMouseEnter={(e) => (e.currentTarget.style.background = 'var(--bg-elevated)')}
                        onMouseLeave={(e) => (e.currentTarget.style.background = 'transparent')}
                      >
                        <VendorBadge vendor={m.vendor} />
                        <span>{m.display_name}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Strategy selector (only shown when >1 model) */}
          {slot.model_ids.length > 1 && (
            <div className="flex items-center gap-2">
              <span style={{ fontSize: '11px', color: 'var(--text-3)', fontFamily: 'JetBrains Mono, monospace' }}>
                {t('slot.strategy')}:
              </span>
              <div className="flex gap-1">
                {STRATEGIES.map((s) => (
                  <button key={s} onClick={() => onUpdate({ ...slot, strategy: s })}
                    className="px-2.5 py-1 rounded-lg text-xs font-medium transition-all"
                    style={{
                      ...(slot.strategy === s
                        ? { background: 'var(--accent-bg)', color: 'var(--accent)', border: '1px solid var(--accent-ring)' }
                        : { background: 'transparent', color: 'var(--text-3)', border: '1px solid var(--border)' }),
                    }}
                  >{t(`slot.strategy.${s}`)}</button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Delete slot */}
        {!isFixed && (
          <button onClick={onRemove}
            className="p-1.5 rounded-lg transition-colors shrink-0"
            style={{ color: 'var(--text-4)', border: '1px solid var(--border)' }}
            onMouseEnter={(e) => { e.currentTarget.style.color = '#ef4444'; e.currentTarget.style.borderColor = '#ef444440' }}
            onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-4)'; e.currentTarget.style.borderColor = 'var(--border)' }}
            title={t('slot.remove')}
          ><X size={13} /></button>
        )}
      </div>
    </div>
  )
}

// ── YAML Import modal ──────────────────────────────────────────────────────────

function YamlImportModal({ onImport, onClose }: {
  onImport: (text: string) => void
  onClose: () => void
}) {
  const t = useT()
  const [text, setText] = useState('')

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: 'rgba(0,0,0,0.4)', backdropFilter: 'blur(4px)' }}>
      <div className="rounded-2xl shadow-2xl w-full max-w-2xl mx-4" style={{ background: 'var(--bg-card)', border: '1px solid var(--border)' }}>
        <div className="flex items-center gap-3 px-5 py-3.5" style={{ borderBottom: '1px solid var(--border)' }}>
          <Upload size={15} style={{ color: 'var(--accent)' }} />
          <span className="font-semibold" style={{ fontSize: '15px', color: 'var(--text-1)' }}>{t('model.import_yaml')}</span>
          <div className="flex-1" />
          <button onClick={onClose} style={{ color: 'var(--text-3)' }}><X size={16} /></button>
        </div>
        <div className="p-5 space-y-4">
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={14}
            className="w-full rounded-xl p-3 font-mono resize-none"
            style={{ fontSize: '12px', lineHeight: 1.6, background: 'var(--bg-elevated)', border: '1px solid var(--border)', color: 'var(--text-1)' }}
            placeholder="# Paste HarnessX Model Config YAML here…"
          />
          <div className="flex justify-end gap-2">
            <button onClick={onClose}
              className="px-4 py-2 rounded-xl text-sm"
              style={{ color: 'var(--text-3)', border: '1px solid var(--border)' }}
            >{t('fs.cancel')}</button>
            <button onClick={() => onImport(text)} disabled={!text.trim()}
              className="px-4 py-2 rounded-xl text-sm font-semibold disabled:opacity-40"
              style={{ background: 'var(--accent)', color: '#fff' }}
            >{t('model.import_yaml')}</button>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Main ModelPage ────────────────────────────────────────────────────────────

export function ModelPage() {
  const t = useT()
  const {
    modelRegistry, addModelDef, removeModelDef, updateModelDef, importModelConfig,
    modelSlots, upsertModelSlot, removeModelSlot,
    vendors,
  } = useSlotsStore()

  const [showAddForm,    setShowAddForm]    = useState(false)
  const [showImport,     setShowImport]     = useState(false)
  const [actionMsg,      setActionMsg]      = useState<{ ok: boolean; text: string } | null>(null)
  const [addSlotName,    setAddSlotName]    = useState('')
  const [showAddSlot,    setShowAddSlot]    = useState(false)
  const [isSaving,       setIsSaving]       = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  function flash(ok: boolean, text: string) {
    setActionMsg({ ok, text })
    window.setTimeout(() => setActionMsg(null), 4000)
  }

  function addModelWithVendorInheritance(m: Omit<ModelDef, 'id'>) {
    const lastSameVendor = [...modelRegistry]
      .reverse()
      .find((x) => x.vendor === m.vendor && (!!x.api_key?.trim() || !!x.base_url?.trim()))
    addModelDef({
      ...m,
      api_key:  m.api_key?.trim() || lastSameVendor?.api_key || '',
      base_url: m.base_url?.trim() || lastSameVendor?.base_url || '',
    })
  }

  // ── Export ──────────────────────────────────────────────────────────────────
  function handleExport() {
    const yaml = modelConfigToYaml({ model_registry: modelRegistry, model_slots: modelSlots })
    const blob  = new Blob([yaml], { type: 'text/yaml' })
    const url   = URL.createObjectURL(blob)
    const a     = document.createElement('a')
    a.href = url; a.download = 'harnessx-models.yaml'; a.click()
    URL.revokeObjectURL(url)
  }

  // ── Import ──────────────────────────────────────────────────────────────────
  function handleImport(text: string) {
    try {
      const parsed = yamlToModelConfig(text)
      if (!parsed.model_registry?.length) throw new Error('No model_registry entries found')
      importModelConfig(
        parsed.model_registry,
        parsed.model_slots ?? modelSlots,
      )
      setShowImport(false)
      flash(true, t('model.import_success'))
    } catch (e) {
      flash(false, t('model.import_error') + ' — ' + String(e))
    }
  }

  async function handleSave() {
    setIsSaving(true)
    try {
      const saved = await api.saveModelConfig({
        registry: modelRegistry.map((m) => ({
          id: m.id,
          display_name: m.display_name,
          vendor: m.vendor,
          model_id: m.model_id,
          api_key: m.api_key,
          base_url: m.base_url,
          extra_headers: m.extra_headers,
          capabilities: m.capabilities,
          extended_thinking: m.extended_thinking,
          thinking_budget_tokens: m.thinking_budget_tokens,
          reasoning_effort: m.reasoning_effort,
          reasoning_summary: m.reasoning_summary,
        })),
        slots: modelSlots.map((s) => ({
          slot_name: s.slot_name,
          model_ids: s.model_ids,
          strategy: s.strategy,
        })),
      })
      importModelConfig(
        saved.registry.map((m) => ({
          id: m.id,
          display_name: m.display_name,
          vendor: m.vendor,
          model_id: m.model_id,
          api_key: m.api_key,
          base_url: m.base_url,
          extra_headers: m.extra_headers,
          capabilities: m.capabilities as ModelCapability[],
          extended_thinking: m.extended_thinking,
          thinking_budget_tokens: m.thinking_budget_tokens,
          reasoning_effort: m.reasoning_effort,
          reasoning_summary: m.reasoning_summary,
        })),
        saved.slots.map((s) => ({
          slot_name: s.slot_name,
          model_ids: s.model_ids,
          strategy: s.strategy as 'primary' | 'fallback' | 'round_robin',
        })),
      )
      flash(true, t('model.save_success'))
    } catch (e) {
      flash(false, t('model.save_error') + ' — ' + String(e))
    } finally {
      setIsSaving(false)
    }
  }

  // ── Add custom slot ─────────────────────────────────────────────────────────
  function handleAddSlot() {
    const name = addSlotName.trim()
    if (!name) return
    upsertModelSlot({ slot_name: name, model_ids: [], strategy: 'primary' })
    setAddSlotName('')
    setShowAddSlot(false)
  }

  const FIXED_SLOTS = new Set(['main'])

  const sectionHeader: React.CSSProperties = {
    fontSize: '17px', fontWeight: 700, color: 'var(--text-1)', letterSpacing: '-0.02em',
  }
  const sectionDesc: React.CSSProperties = {
    fontSize: '13px', color: 'var(--text-3)', marginTop: '4px', lineHeight: 1.6,
  }

  const groupedByVendor = new Map<string, ModelDef[]>()
  for (const m of modelRegistry) {
    if (!groupedByVendor.has(m.vendor)) groupedByVendor.set(m.vendor, [])
    groupedByVendor.get(m.vendor)!.push(m)
  }
  const vendorOrder = [
    ...vendors.map((v) => v.id),
    ...Array.from(groupedByVendor.keys()).filter((v) => !vendors.some((x) => x.id === v)),
  ]
  const providerGroups = vendorOrder
    .filter((vendor) => (groupedByVendor.get(vendor)?.length ?? 0) > 0)
    .map((vendor) => ({ vendor, models: groupedByVendor.get(vendor)! }))

  return (
    <div className="space-y-10">

      {/* Import feedback toast */}
      {actionMsg && (
        <div
          className="flex items-center gap-2 px-4 py-3 rounded-xl"
          style={{
            fontSize: '13px',
            background: actionMsg.ok ? 'rgba(16,185,129,0.08)' : 'rgba(239,68,68,0.08)',
            border: `1px solid ${actionMsg.ok ? 'rgba(16,185,129,0.2)' : 'rgba(239,68,68,0.2)'}`,
            color: actionMsg.ok ? '#10b981' : '#ef4444',
          }}
        >
          {actionMsg.text}
        </div>
      )}

      {/* ════════════════════════════════════════════════════════════════
          Section 1: Model Registry
          ════════════════════════════════════════════════════════════════ */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 style={sectionHeader}>{t('model.registry')}</h2>
            <p style={sectionDesc}>{t('model.registry_desc')}</p>
          </div>
          <div className="flex items-center gap-2">
            <DocLink path="feats/models" label="Docs" />
            <button onClick={() => setShowImport(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-sm font-medium transition-all"
              style={{ color: 'var(--text-2)', border: '1px solid var(--border)', background: 'var(--bg-elevated)' }}
              onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--accent)'; e.currentTarget.style.borderColor = 'var(--accent-ring)' }}
              onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-2)'; e.currentTarget.style.borderColor = 'var(--border)' }}
            ><Upload size={13} /> {t('model.import_yaml')}</button>
            <button
              onClick={handleSave}
              disabled={isSaving}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-sm font-medium transition-all disabled:opacity-50"
              style={{ color: 'var(--text-2)', border: '1px solid var(--border)', background: 'var(--bg-elevated)' }}
              onMouseEnter={(e) => { if (!isSaving) { e.currentTarget.style.color = 'var(--accent)'; e.currentTarget.style.borderColor = 'var(--accent-ring)' } }}
              onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-2)'; e.currentTarget.style.borderColor = 'var(--border)' }}
            >
              <Save size={13} />
              {isSaving ? t('model.saving') : t('model.save')}
            </button>
            <button onClick={handleExport}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-sm font-medium transition-all"
              style={{ color: 'var(--text-2)', border: '1px solid var(--border)', background: 'var(--bg-elevated)' }}
              onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--accent)'; e.currentTarget.style.borderColor = 'var(--accent-ring)' }}
              onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-2)'; e.currentTarget.style.borderColor = 'var(--border)' }}
            ><Download size={13} /> {t('model.export_yaml')}</button>
          </div>
        </div>

        {modelRegistry.length === 0 && !showAddForm && (
          <div className="text-center py-10 rounded-2xl"
            style={{ border: '1px dashed var(--border)', color: 'var(--text-4)', fontSize: '14px' }}
          >{t('model.no_models')}</div>
        )}

        <div className="space-y-3">
          {providerGroups.map((g) => (
            <ProviderGroupCard
              key={g.vendor}
              vendor={g.vendor}
              models={g.models}
              vendors={vendors}
              onUpdateModel={updateModelDef}
              onRemoveModel={removeModelDef}
              onQuickAddModel={addModelWithVendorInheritance}
            />
          ))}
        </div>

        {showAddForm ? (
          <div className="mt-3">
            <AddModelForm
              vendors={vendors}
              onAdd={(m) => {
                addModelWithVendorInheritance(m)
                setShowAddForm(false)
              }}
              onCancel={() => setShowAddForm(false)}
            />
          </div>
        ) : (
          <button
            onClick={() => setShowAddForm(true)}
            className="flex items-center justify-center gap-2 w-full mt-3 py-3 rounded-2xl transition-all"
            style={{ border: '1px dashed var(--border)', color: 'var(--text-3)', fontSize: '14px', background: 'transparent' }}
            onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'var(--accent-ring)'; e.currentTarget.style.color = 'var(--accent)'; e.currentTarget.style.background = 'var(--accent-bg)' }}
            onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text-3)'; e.currentTarget.style.background = 'transparent' }}
          >
            <Plus size={15} />{t('model.add')}
          </button>
        )}
      </section>

      {/* ════════════════════════════════════════════════════════════════
          Section 2: Slot Configuration
          ════════════════════════════════════════════════════════════════ */}
      <section>
        <div className="mb-4">
          <h2 style={sectionHeader}>{t('slot.section')}</h2>
          <p style={sectionDesc}>{t('slot.section_desc')}</p>
        </div>

        <div className="space-y-3">
          {modelSlots.map((slot) => (
            <SlotCard
              key={slot.slot_name}
              slot={slot}
              registry={modelRegistry}
              isFixed={FIXED_SLOTS.has(slot.slot_name)}
              onUpdate={upsertModelSlot}
              onRemove={() => removeModelSlot(slot.slot_name)}
            />
          ))}
        </div>

        {/* Add custom slot */}
        <div className="mt-3">
          {showAddSlot ? (
            <div className="flex items-center gap-2 px-4 py-3 rounded-2xl"
              style={{ border: '1px solid var(--accent-ring)', background: 'var(--accent-bg)' }}
            >
              <input autoFocus type="text" value={addSlotName}
                onChange={(e) => setAddSlotName(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') handleAddSlot(); if (e.key === 'Escape') setShowAddSlot(false) }}
                placeholder="e.g. judge, vision, fast"
                className="flex-1 rounded-xl px-3 py-2 font-mono"
                style={{ fontSize: '13px' }}
              />
              <button onClick={handleAddSlot} disabled={!addSlotName.trim()}
                className="px-4 py-2 rounded-xl text-sm font-semibold disabled:opacity-40"
                style={{ background: 'var(--accent)', color: '#fff' }}
              >{t('slot.add')}</button>
              <button onClick={() => setShowAddSlot(false)}
                className="px-3 py-2 rounded-xl text-sm"
                style={{ color: 'var(--text-3)', border: '1px solid var(--border)' }}
              >{t('fs.cancel')}</button>
            </div>
          ) : (
            <button onClick={() => setShowAddSlot(true)}
              className="flex items-center justify-center gap-2 w-full py-3 rounded-2xl transition-all"
              style={{ border: '1px dashed var(--border)', color: 'var(--text-3)', fontSize: '14px', background: 'transparent' }}
              onMouseEnter={(e) => { e.currentTarget.style.borderColor = 'var(--accent-ring)'; e.currentTarget.style.color = 'var(--accent)'; e.currentTarget.style.background = 'var(--accent-bg)' }}
              onMouseLeave={(e) => { e.currentTarget.style.borderColor = 'var(--border)'; e.currentTarget.style.color = 'var(--text-3)'; e.currentTarget.style.background = 'transparent' }}
            ><Plus size={15} />{t('slot.add')}</button>
          )}
        </div>
      </section>

      {/* Import modal */}
      {showImport && <YamlImportModal onImport={handleImport} onClose={() => setShowImport(false)} />}

      {/* Hidden file input for future file-based import */}
      <input ref={fileRef} type="file" accept=".yaml,.yml" className="hidden" />
    </div>
  )
}
