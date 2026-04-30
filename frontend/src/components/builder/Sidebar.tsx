import { useState, useRef } from 'react'
import { X, Plus, ChevronDown, Upload, Copy, Terminal, AlertTriangle } from 'lucide-react'
import { useLabStore } from '../../store/lab'
import { api } from '../../api/client'
import { yamlToHarnessConfig } from '../../lib/yaml'
import { DEFAULT_HARNESS_CONFIG } from '../../api/types'
import { defaultCustomWorkspace } from '../../lib/labWorkspace'
import { useT } from '../../i18n'

interface CollapsibleProps {
  title: string
  defaultOpen?: boolean
  children: React.ReactNode
}

function Collapsible({ title, defaultOpen = false, children }: CollapsibleProps) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div style={{ borderBottom: '1px solid var(--border)' }} className="last:border-0">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between px-4 py-3 transition-colors"
        style={{ color: 'var(--text-3)' }}
        onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--text-2)')}
        onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-3)')}
      >
        <span className="label-mono">{title}</span>
        <ChevronDown
          size={13}
          style={{
            transform: open ? 'rotate(180deg)' : 'none',
            transition: 'transform 0.15s',
            color: 'var(--text-4)',
          }}
        />
      </button>
      {open && <div className="px-3 pb-3">{children}</div>}
    </div>
  )
}

export function Sidebar() {
  const t = useT()
  const {
    examples,
    selectedCustomId,
    selectedExampleKey,
    customHarnesses,
    harnessConfig,
    isDirty,
    saveCurrentHarness,
    discardCurrentEdits,
    selectCustom, selectExample, createAndSelectCustom, switchToDefault,
    addCustomHarness, removeCustomHarness, duplicateCustomHarness,
    brokenIds,
  } = useLabStore()

  const [newName, setNewName] = useState('')
  const [addingMode, setAddingMode] = useState<'none' | 'blank' | 'save'>('none')
  const fileInputRef = useRef<HTMLInputElement>(null)

  function confirmLeaveIfDirty(): boolean {
    if (!isDirty) return true
    const saveFirst = window.confirm(
      'Current harness has unsaved changes. Click OK to save before switching.',
    )
    if (saveFirst) {
      saveCurrentHarness()
      const s = useLabStore.getState()
      void api.saveAgentHarnessConfig(
        s.savedHarnessConfig,
        s.savedWorkspaceConfig.agent_id,
        s.savedWorkspaceConfig.project,
      ).catch((err) => {
        console.error('Failed to persist agent-shared harness_config.yaml', err)
      })
      return true
    }
    const discard = window.confirm(
      'Discard unsaved changes and continue switching?',
    )
    if (!discard) return false
    discardCurrentEdits()
    return true
  }

  function handleAddCustom() {
    const trimmed = newName.trim()
    if (!trimmed) return
    if (addingMode === 'blank') {
      if (!confirmLeaveIfDirty()) return
      createAndSelectCustom(trimmed, { ...DEFAULT_HARNESS_CONFIG }, defaultCustomWorkspace())
    } else {
      addCustomHarness(trimmed, harnessConfig, defaultCustomWorkspace())
    }
    setNewName('')
    setAddingMode('none')
  }

  function cancelAdding() {
    setAddingMode('none')
    setNewName('')
  }

  function handleImportFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    file.text().then((text) => {
      try {
        const partial = yamlToHarnessConfig(text)
        const cfg = { ...DEFAULT_HARNESS_CONFIG, ...partial }
        const name = file.name.replace(/\.ya?ml$/i, '')
        addCustomHarness(name || 'Imported', cfg, defaultCustomWorkspace())
      } catch {
        alert(t('sidebar.import_yaml_parse_error'))
      }
      if (fileInputRef.current) fileInputRef.current.value = ''
    })
  }

  const isDefault = selectedCustomId === null && selectedExampleKey === null

  return (
    <aside
      className="w-64 shrink-0 flex flex-col overflow-hidden"
      style={{ borderRight: '1px solid var(--border)', background: 'var(--bg-card)' }}
    >
      <div className="flex-1 overflow-y-auto">

        {/* ── CLI Agent (default) ── */}
        <div style={{ borderBottom: '1px solid var(--border)' }}>
          <div className="px-3 py-2">
            <button
              onClick={() => {
                if (!confirmLeaveIfDirty()) return
                switchToDefault()
              }}
              className="w-full text-left rounded-xl px-3 py-2.5 transition-all duration-100 flex items-center gap-2.5"
              style={{
                background: isDefault ? 'var(--accent-bg)' : 'transparent',
                borderLeft: `3px solid ${isDefault ? 'var(--accent)' : 'transparent'}`,
              }}
              onMouseEnter={(e) => {
                if (!isDefault) {
                  e.currentTarget.style.background = 'var(--bg-elevated)'
                  e.currentTarget.style.borderLeftColor = 'var(--accent-ring)'
                }
              }}
              onMouseLeave={(e) => {
                if (!isDefault) {
                  e.currentTarget.style.background = 'transparent'
                  e.currentTarget.style.borderLeftColor = 'transparent'
                }
              }}
            >
              <Terminal
                size={14}
                style={{ color: isDefault ? 'var(--accent)' : 'var(--text-3)', flexShrink: 0 }}
              />
              <div className="min-w-0">
                <div
                  className="font-semibold leading-tight"
                  style={{ fontSize: '14px', color: isDefault ? 'var(--accent)' : 'var(--text-1)' }}
                >
                  {t('sidebar.default_agent')}
                </div>
                <div className="leading-snug mt-0.5" style={{ fontSize: '11px', color: 'var(--text-4)' }}>
                  {t('sidebar.default_agent_desc')}
                </div>
              </div>
            </button>
          </div>
        </div>

        {/* ── Examples (collapsed by default) ── */}
        <Collapsible title={t('sidebar.examples')} defaultOpen={false}>
          <div className="space-y-0.5">
            {examples.map((ex) => (
              <div
                key={ex.key}
                className="rounded-xl px-3 py-2.5 transition-all duration-100"
                style={{
                  background: selectedExampleKey === ex.key ? 'var(--accent-bg)' : 'transparent',
                  borderLeft: `3px solid ${selectedExampleKey === ex.key ? 'var(--accent)' : 'transparent'}`,
                }}
              >
                <button
                  onClick={() => {
                    if (!confirmLeaveIfDirty()) return
                    selectExample(ex.key, { ...ex.harness_config }, ex.workspace ?? undefined)
                  }}
                  className="w-full text-left"
                >
                  <div
                    className="font-medium leading-tight"
                    style={{ fontSize: '14px', color: selectedExampleKey === ex.key ? 'var(--accent)' : 'var(--text-1)' }}
                  >
                    {ex.label}
                  </div>
                  <div className="leading-snug mt-0.5 line-clamp-2" style={{ fontSize: '12px', color: 'var(--text-3)' }}>
                    {ex.description}
                  </div>
                </button>
              </div>
            ))}

            {selectedExampleKey && (
              <div className="mt-2 px-1">
                <button
                  onClick={() => {
                    const ex = examples.find((e) => e.key === selectedExampleKey)
                    if (!ex) return
                    if (!confirmLeaveIfDirty()) return
                    createAndSelectCustom(
                      ex.label,
                      { ...ex.harness_config },
                      defaultCustomWorkspace(),
                    )
                  }}
                  className="w-full px-2.5 py-1.5 rounded-lg transition-colors text-left"
                  style={{
                    fontSize: '12px',
                    color: 'var(--text-3)',
                    border: '1px dashed var(--border)',
                    background: 'transparent',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.color = 'var(--accent)'
                    e.currentTarget.style.borderColor = 'var(--accent-ring)'
                    e.currentTarget.style.background = 'var(--accent-bg)'
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.color = 'var(--text-3)'
                    e.currentTarget.style.borderColor = 'var(--border)'
                    e.currentTarget.style.background = 'transparent'
                  }}
                >
                  {t('builder.copy_example_to_custom')}
                </button>
              </div>
            )}
          </div>
        </Collapsible>

        {/* ── Custom harnesses ── */}
        <Collapsible title={t('sidebar.custom')} defaultOpen={true}>
          <div className="space-y-0.5">
            {customHarnesses.map((ch) => {
              const active = selectedCustomId === ch.id
              const broken = brokenIds.includes(ch.id)
              return (
                <div
                  key={ch.id}
                  className="flex items-center gap-1 group/item rounded-xl px-3 py-2.5 cursor-pointer transition-all duration-100"
                  style={{
                    background: active ? 'var(--accent-bg)' : 'transparent',
                    borderLeft: `3px solid ${broken ? '#f59e0b' : active ? 'var(--accent)' : 'transparent'}`,
                  }}
                  onClick={() => {
                    if (!confirmLeaveIfDirty()) return
                    selectCustom(ch.id, ch.harness_config, ch.workspace)
                  }}
                >
                  {/* Broken processor warning icon */}
                  {broken && (
                    <span
                      className="shrink-0"
                      title={t('sidebar.broken_processor_hint')}
                      style={{ lineHeight: 0 }}
                    >
                      <AlertTriangle size={12} style={{ color: '#f59e0b' }} />
                    </span>
                  )}
                  <span
                    className="flex-1 font-medium truncate"
                    style={{ fontSize: '14px', color: broken ? '#f59e0b' : active ? 'var(--accent)' : 'var(--text-1)' }}
                  >
                    {ch.name}
                  </span>
                  {/* Duplicate button */}
                  <button
                    onClick={(e) => { e.stopPropagation(); duplicateCustomHarness(ch.id) }}
                    className="opacity-0 group-hover/item:opacity-100 transition-opacity shrink-0"
                    style={{ color: 'var(--text-3)' }}
                    title={t('sidebar.duplicate')}
                    onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--text-1)')}
                    onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-3)')}
                  >
                    <Copy size={12} />
                  </button>
                  {/* Delete button */}
                  <button
                    onClick={(e) => { e.stopPropagation(); removeCustomHarness(ch.id) }}
                    className="opacity-0 group-hover/item:opacity-100 transition-opacity shrink-0"
                    style={{ color: 'var(--text-3)' }}
                    title={t('sidebar.delete')}
                    onMouseEnter={(e) => (e.currentTarget.style.color = '#ef4444')}
                    onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-3)')}
                  >
                    <X size={12} />
                  </button>
                </div>
              )
            })}

            {addingMode !== 'none' ? (
              <div className="mt-1.5 space-y-1.5">
                <p style={{ fontSize: '12px', color: 'var(--text-4)' }}>
                  {addingMode === 'blank' ? t('sidebar.new_blank_hint') : t('sidebar.save_current_hint')}
                </p>
                <div className="flex items-center gap-1.5">
                  <input
                    autoFocus
                    type="text"
                    value={newName}
                    onChange={(e) => setNewName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') handleAddCustom()
                      if (e.key === 'Escape') cancelAdding()
                    }}
                    placeholder={t('sidebar.harness_name_placeholder')}
                    className="flex-1 rounded-lg px-2.5 py-1.5 min-w-0"
                    style={{ fontSize: '13px' }}
                  />
                  <button
                    onClick={handleAddCustom}
                    disabled={!newName.trim()}
                    className="disabled:opacity-40 px-1.5 shrink-0 transition-colors"
                    style={{ fontSize: '13px', color: 'var(--accent)' }}
                  >
                    {addingMode === 'blank' ? t('sidebar.create') : t('sidebar.save')}
                  </button>
                  <button
                    onClick={cancelAdding}
                    className="px-1.5 shrink-0 transition-colors"
                    style={{ fontSize: '13px', color: 'var(--text-3)' }}
                    onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--text-2)')}
                    onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-3)')}
                  >
                    {t('fs.cancel')}
                  </button>
                </div>
              </div>
            ) : (
              <div className="flex flex-col gap-1.5 mt-2">
                <button
                  onClick={() => setAddingMode('blank')}
                  className="flex items-center gap-1.5 transition-colors"
                  style={{ fontSize: '13px', color: 'var(--accent)' }}
                  onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--text-1)')}
                  onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--accent)')}
                >
                  <Plus size={12} />
                  {t('sidebar.new_blank')}
                </button>

                <button
                  onClick={() => setAddingMode('save')}
                  className="flex items-center gap-1.5 transition-colors"
                  style={{ fontSize: '13px', color: 'var(--text-3)' }}
                  onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--text-2)')}
                  onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-3)')}
                >
                  <Plus size={12} />
                  {t('sidebar.save_current')}
                </button>

                <label
                  className="flex items-center gap-1.5 transition-colors cursor-pointer"
                  style={{ fontSize: '13px', color: 'var(--text-3)' }}
                  onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--text-2)')}
                  onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-3)')}
                >
                  <Upload size={12} />
                  {t('sidebar.import_yaml')}
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".yaml,.yml"
                    className="hidden"
                    onChange={handleImportFile}
                  />
                </label>
              </div>
            )}
          </div>
        </Collapsible>
      </div>
    </aside>
  )
}
