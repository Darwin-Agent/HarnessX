import { useCallback, useEffect, useState } from 'react'
import { Download, FlaskConical, RotateCcw, Save } from 'lucide-react'
import { useLabStore } from '../store/lab'
import { useRunsStore } from '../store/runs'
import { useSlotsStore } from '../store/slots'
import { useUIStore } from '../store/ui'
import { api } from '../api/client'
import { Sidebar } from '../components/builder/Sidebar'
import { DescriptorPanel } from '../components/builder'
import { CustomProcessorImportModal } from '../components/builder/CustomProcessorImportModal'
import { ChatPanel } from '../components/chat/ChatPanel'
import { ChatInput } from '../components/chat/ChatInput'
import { Badge } from '../components/shared/Badge'
import { CostChip } from '../components/shared/CostChip'
import type { Attachment, TaskBlock } from '../api/types'
import { harnessConfigToYaml } from '../lib/yaml'
import { resolveBuilderHarnessName } from '../lib/labWorkspace'
import { useT } from '../i18n'

const STATUS_BADGE_KEYS = {
  idle:    { labelKey: 'status.idle',    variant: 'gray'  } as const,
  running: { labelKey: 'status.running', variant: 'blue'  } as const,
  done:    { labelKey: 'status.done',    variant: 'green' } as const,
  error:   { labelKey: 'status.error',   variant: 'red'   } as const,
}

export function BuilderPage() {
  const t = useT()
  const {
    setDimensions,
    successCriteria, harnessConfig, workspaceConfig,
    savedHarnessConfig, savedWorkspaceConfig,
    isDirty, updateProcessors, setWorkspaceConfig,
    saveCurrentHarness, discardCurrentEdits,
    customHarnesses, examples, builderView,
    selectedCustomId, selectedExampleKey,
    startChat, backToConfig,
    validateAgainstInstalled,
    repairFileTargets,
  } = useLabStore()
  const { columns, addColumn, updateColumn, startRun, stopRun, resetSession, resumeHistorySession } = useRunsStore()
  const { toSlotConfig, toModelConfigPayload, validateLaunch, mcpServers } = useSlotsStore()
  const setModelOpen = useUIStore((s) => s.setModelOpen)
  const [saveConfigNotice, setSaveConfigNotice] = useState<{ type: 'success' | 'error'; message: string; path?: string } | null>(null)
  const [customProcModalOpen, setCustomProcModalOpen] = useState(false)
  const [configEditing, setConfigEditing] = useState(false)

  const modelError = validateLaunch()

  useEffect(() => {
    if (columns.length === 0) {
      const s = useLabStore.getState()
      const name = resolveBuilderHarnessName({
        selectedCustomId: s.selectedCustomId,
        selectedExampleKey: s.selectedExampleKey,
        customHarnesses: s.customHarnesses,
        examples: s.examples,
      })
      addColumn(s.savedHarnessConfig, name, s.savedWorkspaceConfig)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Validate custom harnesses against installed processors on mount.
  // If any referenced file:// processor no longer exists on the backend,
  // the harness is marked broken so the Sidebar can show a warning.
  useEffect(() => {
    api.customProcessors()
      .then((list) => {
        validateAgainstInstalled(list.map((p) => p.target))
        repairFileTargets(list)
      })
      .catch(() => {})
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const col = columns[0] ?? null
  const isRunning = col?.status === 'running'
  const totalCost = col?.steps.reduce((s, r) => s + r.cost_usd, 0) ?? 0
  const { labelKey, variant } = STATUS_BADGE_KEYS[col?.status ?? 'idle']

  const harnessLabel = selectedCustomId
    ? (customHarnesses.find((c) => c.id === selectedCustomId)?.name ?? 'Custom')
    : selectedExampleKey
      ? (examples.find((e) => e.key === selectedExampleKey)?.label ?? t('sidebar.examples'))
      : t('sidebar.default_agent')
  const isDefaultSelected = selectedCustomId === null && selectedExampleKey === null
  const isExampleSelected = selectedExampleKey !== null
  const canEditConfig = selectedCustomId !== null && configEditing

  // Clear notices when switching harness selection.
  useEffect(() => {
    setSaveConfigNotice(null)
    setConfigEditing(false)
  }, [selectedCustomId, selectedExampleKey])

  // Auto-hide top notice after 3s.
  useEffect(() => {
    if (!saveConfigNotice) return
    const timer = window.setTimeout(() => setSaveConfigNotice(null), 3000)
    return () => window.clearTimeout(timer)
  }, [saveConfigNotice])

  const handleSend = useCallback((text: string, attachments: Attachment[]) => {
    if (!col) return
    const selectedName = resolveBuilderHarnessName({
      selectedCustomId,
      selectedExampleKey,
      customHarnesses,
      examples,
    })
    const ws = {
      agentId: savedWorkspaceConfig.agent_id,
      project: savedWorkspaceConfig.project,
    }
    updateColumn(col.id, {
      harnessConfig: savedHarnessConfig,
      harnessName: selectedName,
      workspaceAgentId: ws.agentId,
      workspaceProject: ws.project,
    })
    const task: string | TaskBlock[] = attachments.length > 0
      ? [
          ...(text ? [{ type: 'text' as const, text }] : []),
          ...attachments.map((a) => ({
            type:   'image' as const,
            source: { type: 'base64' as const, media_type: a.media_type, data: a.data },
          })),
        ]
      : text
    startRun(
      col.id,
      task,
      successCriteria,
      toModelConfigPayload(),
      toSlotConfig(),
      {
        harnessConfig: savedHarnessConfig,
        harnessName: selectedName,
        agentId: ws.agentId,
        project: ws.project,
      },
    )
  }, [col, savedHarnessConfig, savedWorkspaceConfig, successCriteria, updateColumn, startRun, toSlotConfig, toModelConfigPayload, selectedCustomId, selectedExampleKey, customHarnesses, examples])

  const handleNewChat = useCallback(() => {
    if (col) resetSession(col.id)
  }, [col, resetSession])

  const handleExportYaml = useCallback(() => {
    const includeMcpSidecar = window.confirm(
      'Also export mcp_servers.json sidecar for reproducible MCP setup?',
    )
    const exportConfig = includeMcpSidecar
      ? {
          ...harnessConfig,
          mcp_config: {
            source: 'file' as const,
            path: './mcp_servers.json',
          },
        }
      : harnessConfig
    const yamlStr = harnessConfigToYaml(exportConfig)
    const blob = new Blob([yamlStr], { type: 'text/yaml' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${harnessLabel.toLowerCase().replace(/\s+/g, '-')}.yaml`
    a.click()
    URL.revokeObjectURL(url)

    if (includeMcpSidecar) {
      const sidecar = new Blob([JSON.stringify(mcpServers, null, 2)], { type: 'application/json' })
      const sidecarUrl = URL.createObjectURL(sidecar)
      const sidecarA = document.createElement('a')
      sidecarA.href = sidecarUrl
      sidecarA.download = 'mcp_servers.json'
      sidecarA.click()
      URL.revokeObjectURL(sidecarUrl)
    }
  }, [harnessConfig, harnessLabel, mcpServers])

  const handleCustomProcessorsImported = useCallback(async () => {
    const [schemaRes, processorList] = await Promise.all([api.schema(), api.customProcessors()])
    setDimensions(schemaRes.dimensions)
    validateAgainstInstalled(processorList.map((p) => p.target))
    repairFileTargets(processorList)
  }, [setDimensions, validateAgainstInstalled, repairFileTargets])

  const handleSaveConfig = useCallback(async () => {
    setSaveConfigNotice(null)
    saveCurrentHarness()
    const s = useLabStore.getState()
    const selectedName = resolveBuilderHarnessName({
      selectedCustomId: s.selectedCustomId,
      selectedExampleKey: s.selectedExampleKey,
      customHarnesses: s.customHarnesses,
      examples: s.examples,
    })
    if (col) {
      updateColumn(col.id, {
        harnessConfig: s.savedHarnessConfig,
        harnessName: selectedName,
        workspaceAgentId: s.savedWorkspaceConfig.agent_id,
        workspaceProject: s.savedWorkspaceConfig.project,
      })
    }
    try {
      const saved = await api.saveAgentHarnessConfig(
        s.savedHarnessConfig,
        s.savedWorkspaceConfig.agent_id,
        s.savedWorkspaceConfig.project,
      )
      setSaveConfigNotice({
        type: 'success',
        message: '配置已保存到',
        path: saved.path,
      })
      setConfigEditing(false)
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      setSaveConfigNotice({
        type: 'error',
        message: `配置保存失败：${msg}`,
      })
      console.error('Failed to persist agent-shared harness_config.yaml', err)
    }
  }, [col, updateColumn, saveCurrentHarness])

  const handleDiscardConfig = useCallback(() => {
    discardCurrentEdits()
    setConfigEditing(false)
  }, [discardCurrentEdits])

  const handleChatTab = useCallback(async () => {
    if (!isDirty) {
      startChat()
      return
    }
    const shouldSave = window.confirm(
      'Configuration has unsaved changes. Click OK to save and switch to Chat.',
    )
    if (!shouldSave) return
    await handleSaveConfig()
    startChat()
  }, [isDirty, handleSaveConfig, startChat])

  const ghostBtn = {
    base:  { color: 'var(--text-2)', background: 'transparent', cursor: 'pointer' } as React.CSSProperties,
    hover: { color: 'var(--text-1)', background: 'var(--bg-elevated)' } as React.CSSProperties,
  }

  return (
    <div className="flex flex-1 min-h-0 overflow-hidden">
      <Sidebar />

      <main className="flex-1 min-w-0 flex flex-col min-h-0">
        {/* ── Shared header with tab switcher ── */}
        <div
          className="flex items-center gap-3 px-5 py-2.5 shrink-0"
          style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-card)' }}
        >
          <span
            className="font-semibold text-sm"
            style={{ color: 'var(--text-1)', letterSpacing: '-0.02em', marginRight: '4px' }}
          >
            {harnessLabel}
          </span>

          {/* Tab switcher */}
          <div
            className="flex items-center rounded-lg p-0.5"
            style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)' }}
          >
            {(['config', 'chat'] as const).map((view) => (
              <button
                key={view}
                onClick={() => {
                  if (view === 'chat') {
                    void handleChatTab()
                  } else {
                    backToConfig()
                  }
                }}
                className="px-3 py-1 rounded-md text-xs font-medium transition-all duration-150"
                style={
                  builderView === view
                    ? { background: 'var(--bg-card)', color: 'var(--text-1)', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }
                    : { background: 'transparent', color: 'var(--text-3)' }
                }
              >
                {view === 'config' ? 'Config' : 'Chat'}
              </button>
            ))}
          </div>

          <div className="flex-1" />

          {/* Status / cost — shown in header for both views */}
          {builderView === 'chat' && (
            <>
              <Badge variant={variant} size="xs">{t(labelKey)}</Badge>
              {totalCost > 0 && <CostChip usd={totalCost} />}
              {!!col && (
                <button
                  onClick={handleNewChat}
                  disabled={isRunning || (col.messages?.length ?? 0) === 0}
                  className="flex items-center gap-1.5 text-xs px-2 py-1 rounded transition-colors disabled:opacity-35 disabled:cursor-not-allowed"
                  style={{ color: 'var(--text-3)', border: '1px solid var(--border)', background: 'transparent' }}
                  title={t('status.new_chat')}
                >
                  <RotateCcw size={12} />
                  {t('status.new_chat')}
                </button>
              )}
              {isRunning && (
                <button
                  onClick={() => col && stopRun(col.id)}
                  className="text-xs px-2 py-0.5 rounded"
                  style={{ color: '#ef4444', background: 'rgba(239,68,68,0.08)' }}
                >
                  {t('status.stop')}
                </button>
              )}
            </>
          )}

          {/* Export YAML — shown in config view */}
          {builderView === 'config' && (
            <div className="flex items-center gap-1.5">
              <button
                onClick={() => setCustomProcModalOpen(true)}
                className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded transition-colors"
                style={ghostBtn.base}
                title="Import custom processor"
                onMouseEnter={(e) => Object.assign(e.currentTarget.style, ghostBtn.hover)}
                onMouseLeave={(e) => Object.assign(e.currentTarget.style, ghostBtn.base)}
              >
                <FlaskConical size={12} />
                Custom Processor
              </button>
              <button
                onClick={handleExportYaml}
                className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded transition-colors"
                style={ghostBtn.base}
                title={t('builder.export_yaml')}
                onMouseEnter={(e) => Object.assign(e.currentTarget.style, ghostBtn.hover)}
                onMouseLeave={(e) => Object.assign(e.currentTarget.style, ghostBtn.base)}
              >
                <Download size={12} />
                {t('builder.export_yaml')}
              </button>
              <button
                onClick={() => {
                  if (selectedCustomId === null) return
                  setConfigEditing((v) => !v)
                }}
                disabled={selectedCustomId === null}
                className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                style={ghostBtn.base}
                title={selectedCustomId === null ? 'Only custom harnesses are editable. Copy to Custom to edit.' : 'Edit config'}
                onMouseEnter={(e) => Object.assign(e.currentTarget.style, ghostBtn.hover)}
                onMouseLeave={(e) => Object.assign(e.currentTarget.style, ghostBtn.base)}
              >
                {canEditConfig ? 'Editing' : 'Edit'}
              </button>
              <button
                onClick={() => { void handleSaveConfig() }}
                disabled={!canEditConfig || !isDirty}
                className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                style={ghostBtn.base}
                title="Save config"
                onMouseEnter={(e) => Object.assign(e.currentTarget.style, ghostBtn.hover)}
                onMouseLeave={(e) => Object.assign(e.currentTarget.style, ghostBtn.base)}
              >
                <Save size={12} />
                Save Config
              </button>
              <button
                onClick={handleDiscardConfig}
                disabled={!canEditConfig || !isDirty}
                className="flex items-center gap-1.5 text-xs px-2.5 py-1.5 rounded transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                style={ghostBtn.base}
                title="Discard changes"
                onMouseEnter={(e) => Object.assign(e.currentTarget.style, ghostBtn.hover)}
                onMouseLeave={(e) => Object.assign(e.currentTarget.style, ghostBtn.base)}
              >
                Discard
              </button>
            </div>
          )}
        </div>

        {saveConfigNotice && (
          <div
            className="px-5 py-3 shrink-0"
            style={{
              background: saveConfigNotice.type === 'success' ? 'rgba(16,185,129,0.08)' : 'rgba(239,68,68,0.08)',
              borderBottom: `1px solid ${saveConfigNotice.type === 'success' ? 'rgba(16,185,129,0.22)' : 'rgba(239,68,68,0.25)'}`,
            }}
          >
            <div
              className="text-xs flex items-center gap-2"
              style={{ color: saveConfigNotice.type === 'success' ? '#10b981' : '#ef4444' }}
            >
              <span>{saveConfigNotice.message}</span>
              {saveConfigNotice.path && (
                <span className="font-mono" style={{ color: 'var(--text-2)' }}>
                  {saveConfigNotice.path}
                </span>
              )}
            </div>
          </div>
        )}

        {/* ── Config view ── */}
        {builderView === 'config' && (
          <>
            <div
              className="px-5 py-3 shrink-0 flex items-center gap-3"
              style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-card)' }}
            >
              <div className="text-xs" style={{ color: 'var(--text-3)', minWidth: 82 }}>agent_id</div>
              <input
                value={workspaceConfig.agent_id}
                disabled={!canEditConfig}
                onChange={(e) => setWorkspaceConfig({ agent_id: e.target.value })}
                className="px-2 py-1 rounded text-xs font-mono"
                style={{
                  background: 'var(--bg-base)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-1)',
                  width: '220px',
                  opacity: canEditConfig ? 1 : 0.65,
                }}
              />
              <div className="text-xs" style={{ color: 'var(--text-3)', minWidth: 62 }}>project</div>
              <input
                value={workspaceConfig.project}
                disabled={!canEditConfig}
                onChange={(e) => setWorkspaceConfig({ project: e.target.value })}
                className="px-2 py-1 rounded text-xs font-mono"
                style={{
                  background: 'var(--bg-base)',
                  border: '1px solid var(--border)',
                  color: 'var(--text-1)',
                  width: '280px',
                  opacity: canEditConfig ? 1 : 0.65,
                }}
              />
              {isExampleSelected && (
                <span className="text-xs" style={{ color: 'var(--text-4)' }}>
                  Example is read-only. Save as Custom to edit.
                </span>
              )}
              {isDefaultSelected && (
                <span className="text-xs" style={{ color: 'var(--text-4)' }}>
                  CLI Agent is read-only. Save current config as Custom to edit.
                </span>
              )}
              {selectedCustomId !== null && !configEditing && (
                <span className="text-xs" style={{ color: 'var(--text-4)' }}>
                  Click Edit to modify config.
                </span>
              )}
              <div className="flex-1" />
              <span
                className="text-xs font-mono"
                style={{ color: isDirty ? '#f59e0b' : 'var(--text-4)' }}
              >
                {isDirty ? 'unsaved changes' : 'saved'}
              </span>
            </div>
            <div className="flex-1 overflow-y-auto" style={{ background: 'var(--bg-base)' }}>
              <DescriptorPanel
                processors={harnessConfig.processors}
                onProcessorsChange={canEditConfig ? updateProcessors : () => {}}
                readOnly={!canEditConfig}
              />
            </div>
          </>
        )}

        {/* ── Chat view ── */}
        {builderView === 'chat' && (
          <div className="flex flex-col flex-1 min-h-0">
            {col?.result && (
              <div
                className="text-xs px-4 py-1 shrink-0 font-mono"
                style={{
                  background: col.result.passed === true
                    ? 'rgba(16,185,129,0.08)'
                    : col.result.passed === false
                    ? 'rgba(239,68,68,0.08)'
                    : 'var(--bg-elevated)',
                  borderBottom: '1px solid var(--border)',
                  color: col.result.passed === true ? '#10b981'
                    : col.result.passed === false ? '#ef4444'
                    : 'var(--text-3)',
                }}
              >
                {col.result.passed === true ? t('status.passed')
                  : col.result.passed === false ? t('status.failed')
                  : '—'}
                {' · '}{col.result.exit_reason}{' · '}{col.result.steps} steps
              </div>
            )}

            {col?.error && (
              <div
                className="text-xs px-4 py-1 shrink-0"
                style={{
                  background: 'rgba(239,68,68,0.08)',
                  color: '#ef4444',
                  borderBottom: '1px solid rgba(239,68,68,0.2)',
                }}
              >
                {col.error}
              </div>
            )}

            {modelError && (
              <div
                className="text-xs px-4 py-1.5 shrink-0 flex items-center gap-1.5"
                style={{
                  background: 'rgba(251,191,36,0.06)',
                  borderBottom: '1px solid rgba(251,191,36,0.15)',
                  color: '#d97706',
                }}
              >
                <span>{t('chat.no_model')}</span>
                <button
                  onClick={() => setModelOpen(true)}
                  style={{ textDecoration: 'underline', cursor: 'pointer', background: 'none', border: 'none', color: 'inherit', fontSize: 'inherit', padding: 0 }}
                >
                  {t('chat.no_model_link')}
                </button>
              </div>
            )}

            {/* Read-only history banner */}
            {col?.isReadOnly && (
              <div
                className="text-xs px-4 py-2 shrink-0 flex items-center gap-3"
                style={{
                  background: 'rgba(139,92,246,0.07)',
                  borderBottom: '1px solid rgba(139,92,246,0.2)',
                  color: 'var(--text-2)',
                }}
              >
                <span style={{ color: '#8b5cf6', flexShrink: 0 }}>📖</span>
                <span className="flex-1" style={{ fontFamily: 'JetBrains Mono, monospace' }}>
                  历史对话 · {col.historySessionId?.slice(0, 12)}… · 只读
                </span>
                <button
                  onClick={() => col && resumeHistorySession(col.id)}
                  className="px-3 py-1 rounded-lg font-semibold"
                  style={{
                    fontSize: 11,
                    background: 'rgba(139,92,246,0.12)',
                    border:     '1px solid rgba(139,92,246,0.3)',
                    color:      '#8b5cf6',
                    cursor:     'pointer',
                  }}
                >
                  续接此对话 →
                </button>
              </div>
            )}

            {/* Centered conversation container */}
            <div className="flex-1 min-h-0 flex flex-col max-w-3xl mx-auto w-full">
              <ChatPanel
                messages={col?.messages ?? []}
                autoScroll={isRunning}
              />
              <ChatInput
                onSend={handleSend}
                onNewChat={handleNewChat}
                disabled={isRunning || !!modelError || !!col?.isReadOnly}
              />
            </div>
          </div>
        )}
      </main>

      <CustomProcessorImportModal
        open={customProcModalOpen}
        onClose={() => setCustomProcModalOpen(false)}
        onImported={handleCustomProcessorsImported}
      />
    </div>
  )
}
