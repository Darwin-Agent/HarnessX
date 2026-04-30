import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { DEFAULT_HARNESS_CONFIG } from '../api/types'
import type {
  HarnessConfig,
  HarnessWorkspaceConfig,
  Dimension,
  ExampleDescriptor,
  CustomHarness,
  DimensionDragPayload,
} from '../api/types'
import { defaultCustomWorkspace, workspaceFromHarnessName } from '../lib/labWorkspace'

export type BuilderView = 'config' | 'chat'

interface LabState {
  // Schema loaded from /api/schema
  dimensions:       Dimension[]
  setDimensions:    (d: Dimension[]) => void

  // Example harnesses loaded from /api/examples
  examples:    ExampleDescriptor[]
  setExamples: (e: ExampleDescriptor[]) => void

  // Builder page: current harness config being edited
  harnessConfig:      HarnessConfig
  workspaceConfig:    HarnessWorkspaceConfig
  savedHarnessConfig: HarnessConfig
  savedWorkspaceConfig: HarnessWorkspaceConfig
  isDirty:            boolean
  setHarnessConfig:   (c: HarnessConfig) => void
  updateProcessors:   (processors: Record<string, unknown>[]) => void
  setWorkspaceConfig: (ws: Partial<HarnessWorkspaceConfig>) => void
  saveCurrentHarness: () => void
  discardCurrentEdits: () => void
  replaceCurrentConfig: (
    harnessConfig: HarnessConfig,
    workspace?: HarnessWorkspaceConfig,
  ) => void

  // Global drag state — set when any DimensionCard is being dragged
  draggingDimension:    DimensionDragPayload | null
  setDraggingDimension: (p: DimensionDragPayload | null) => void

  // Dimension clipboard — persists across route changes (Builder → Compare)
  dimensionClipboard:    DimensionDragPayload | null
  copyDimension:         (p: DimensionDragPayload) => void
  clearDimensionClipboard: () => void

  // Builder view state machine
  builderView:      BuilderView
  selectedCustomId: string | null
  selectedExampleKey: string | null

  selectCustom:  (id: string, harnessConfig: HarnessConfig, workspace?: HarnessWorkspaceConfig) => void
  selectExample: (key: string, harnessConfig: HarnessConfig, workspace?: HarnessWorkspaceConfig) => void
  startChat:     () => void
  backToConfig:  () => void

  // Custom harnesses (user-defined, persisted to localStorage)
  customHarnesses:          CustomHarness[]
  addCustomHarness:         (name: string, harnessConfig: HarnessConfig, workspace?: HarnessWorkspaceConfig) => void
  createAndSelectCustom:    (name: string, harnessConfig: HarnessConfig, workspace?: HarnessWorkspaceConfig) => void
  removeCustomHarness:      (id: string) => void
  renameCustomHarness:      (id: string, name: string) => void
  duplicateCustomHarness:   (id: string) => void
  switchToDefault:          () => void  // select the built-in CLI agent

  // Validation: custom harnesses whose processor files no longer exist on the backend.
  // Not persisted — recomputed on BuilderPage mount via GET /processors/custom.
  brokenIds:                  string[]
  validateAgainstInstalled:   (validTargets: string[]) => void

  // Repair: after re-importing a custom processor, replace stale file:// targets
  // in harnessConfig / savedHarnessConfig / customHarnesses with fresh paths.
  repairFileTargets: (installedProcessors: { target: string }[]) => void

  // Task input (shared across builder + compare)
  successCriteria: string
  setSuccessCriteria: (s: string) => void
}

let _nextCustomId = 1
function newId() { return `custom-${Date.now()}-${_nextCustomId++}` }

function cloneHarnessConfig(cfg: HarnessConfig): HarnessConfig {
  if (typeof structuredClone === 'function') return structuredClone(cfg)
  return JSON.parse(JSON.stringify(cfg)) as HarnessConfig
}

function cloneWorkspaceConfig(ws: HarnessWorkspaceConfig): HarnessWorkspaceConfig {
  return { agent_id: ws.agent_id, project: ws.project }
}

function normalizeWorkspaceConfig(
  ws: HarnessWorkspaceConfig | undefined | null,
  fallbackName: string,
): HarnessWorkspaceConfig {
  const fallback = workspaceFromHarnessName(fallbackName)
  const agent_id = (ws?.agent_id || '').trim() || fallback.agentId
  const project = (ws?.project || '').trim() || fallback.project
  return { agent_id, project }
}

function sameHarnessConfig(a: HarnessConfig, b: HarnessConfig): boolean {
  return JSON.stringify(a) === JSON.stringify(b)
}

function sameWorkspaceConfig(a: HarnessWorkspaceConfig, b: HarnessWorkspaceConfig): boolean {
  return a.agent_id === b.agent_id && a.project === b.project
}

/**
 * Remove duplicate processors by class name, keeping the last occurrence.
 * Handles file:// targets (file:///path.py::ClassName) and module-path targets
 * (module.path.ClassName). Eliminates stale file:// paths and module-path entries
 * that accumulated from multiple re-imports of the same processor class.
 */
function deduplicateProcessors(processors: Record<string, unknown>[]): Record<string, unknown>[] {
  const lastIndex = new Map<string, number>()
  processors.forEach((p, i) => {
    const t = p._target_
    if (typeof t !== 'string') return
    const cls = t.startsWith('file://')
      ? (t.split('::').pop()?.trim() ?? '')
      : (t.split('.').pop()?.trim() ?? '')
    if (cls) lastIndex.set(cls, i)
  })
  return processors.filter((p, i) => {
    const t = p._target_
    if (typeof t !== 'string') return true
    const cls = t.startsWith('file://')
      ? (t.split('::').pop()?.trim() ?? '')
      : (t.split('.').pop()?.trim() ?? '')
    return !cls || lastIndex.get(cls) === i
  })
}

/**
 * Replace stale file:// targets in a processor list with fresh installed ones
 * that share the same class name.  Module-path targets are left untouched.
 */
function repairProcessors(
  processors: Record<string, unknown>[],
  classToTarget: Map<string, string>,
): { repaired: Record<string, unknown>[]; changed: boolean } {
  let changed = false
  const repaired = processors.map((p) => {
    const t = p._target_
    if (typeof t !== 'string' || !t.startsWith('file://')) return p
    const cls = t.split('::').pop()?.trim() ?? ''
    if (!cls) return p
    const fresh = classToTarget.get(cls)
    if (!fresh || fresh === t) return p
    changed = true
    return { ...p, _target_: fresh }
  })
  return { repaired, changed }
}

/** Collect all file:// _target_ strings from a harness config's processor list. */
function getCustomTargets(harnessConfig: HarnessConfig): string[] {
  return harnessConfig.processors
    .map((p) => p._target_ as string | undefined)
    .filter((t): t is string => typeof t === 'string' && t.startsWith('file://'))
}

export const useLabStore = create<LabState>()(
  persist(
    (set) => ({
      dimensions:       [],
      setDimensions:    (dimensions) => set({ dimensions }),

      examples:    [],
      setExamples: (examples) => set({ examples }),

      harnessConfig:      { ...DEFAULT_HARNESS_CONFIG },
      workspaceConfig:    normalizeWorkspaceConfig(null, 'cli-agent'),
      savedHarnessConfig: { ...DEFAULT_HARNESS_CONFIG },
      savedWorkspaceConfig: normalizeWorkspaceConfig(null, 'cli-agent'),
      isDirty:            false,
      setHarnessConfig: (harnessConfig) =>
        set((s) => {
          const next = cloneHarnessConfig(harnessConfig)
          return {
            harnessConfig: next,
            isDirty: !sameHarnessConfig(next, s.savedHarnessConfig)
              || !sameWorkspaceConfig(s.workspaceConfig, s.savedWorkspaceConfig),
          }
        }),
      updateProcessors: (processors) =>
        set((s) => {
          const nextHarnessConfig = { ...s.harnessConfig, processors } as HarnessConfig
          return {
            harnessConfig: nextHarnessConfig,
            isDirty: !sameHarnessConfig(nextHarnessConfig, s.savedHarnessConfig)
              || !sameWorkspaceConfig(s.workspaceConfig, s.savedWorkspaceConfig),
          }
        }),
      setWorkspaceConfig: (ws) =>
        set((s) => {
          const selectedCustomName = s.selectedCustomId
            ? (s.customHarnesses.find((c) => c.id === s.selectedCustomId)?.name ?? s.selectedCustomId)
            : null
          const selectedExampleName = s.selectedExampleKey
            ? (s.examples.find((e) => e.key === s.selectedExampleKey)?.label ?? s.selectedExampleKey)
            : null
          const fallbackName = selectedCustomName ?? selectedExampleName ?? 'cli-agent'
          const next = normalizeWorkspaceConfig(
            { ...s.workspaceConfig, ...ws },
            fallbackName,
          )
          return {
            workspaceConfig: next,
            isDirty: !sameWorkspaceConfig(next, s.savedWorkspaceConfig)
              || !sameHarnessConfig(s.harnessConfig, s.savedHarnessConfig),
          }
        }),
      saveCurrentHarness: () =>
        set((s) => {
          const cleanHarness = {
            ...s.harnessConfig,
            processors: deduplicateProcessors(s.harnessConfig.processors),
          }
          const nextSavedHarness = cloneHarnessConfig(cleanHarness)
          const nextSavedWorkspace = cloneWorkspaceConfig(s.workspaceConfig)
          const statePatch: Partial<LabState> = {
            harnessConfig: cleanHarness,
            savedHarnessConfig: nextSavedHarness,
            savedWorkspaceConfig: nextSavedWorkspace,
            isDirty: false,
          }
          if (s.selectedCustomId !== null) {
            statePatch.customHarnesses = s.customHarnesses.map((c) =>
              c.id === s.selectedCustomId
                ? {
                    ...c,
                    harness_config: cloneHarnessConfig(nextSavedHarness),
                    workspace: cloneWorkspaceConfig(nextSavedWorkspace),
                  }
                : c
            )
          }
          return statePatch
        }),
      discardCurrentEdits: () =>
        set((s) => ({
          harnessConfig: cloneHarnessConfig(s.savedHarnessConfig),
          workspaceConfig: cloneWorkspaceConfig(s.savedWorkspaceConfig),
          isDirty: false,
        })),
      replaceCurrentConfig: (harnessConfig, workspace) =>
        set((s) => {
          const fallbackName = s.selectedCustomId ?? s.selectedExampleKey ?? 'cli-agent'
          const nextHarness = cloneHarnessConfig(harnessConfig)
          const nextWorkspace = normalizeWorkspaceConfig(workspace, fallbackName)
          return {
            harnessConfig: nextHarness,
            savedHarnessConfig: cloneHarnessConfig(nextHarness),
            workspaceConfig: nextWorkspace,
            savedWorkspaceConfig: cloneWorkspaceConfig(nextWorkspace),
            isDirty: false,
          }
        }),

      draggingDimension:    null,
      setDraggingDimension: (draggingDimension) => set({ draggingDimension }),

      dimensionClipboard:      null,
      copyDimension:           (dimensionClipboard) => set({ dimensionClipboard }),
      clearDimensionClipboard: () => set({ dimensionClipboard: null }),

      // Builder view state — default to 'chat' so Lab opens ready to converse
      builderView:      'chat',
      selectedCustomId: null,
      selectedExampleKey: null,

      selectCustom: (id, harnessConfig, workspace) =>
        set((s) => {
          const selected = s.customHarnesses.find((c) => c.id === id)
          const nextHarness = cloneHarnessConfig(selected?.harness_config ?? harnessConfig)
          const nextWorkspace = normalizeWorkspaceConfig(
            selected?.workspace ?? workspace,
            selected?.name ?? id,
          )
          return {
            selectedCustomId: id,
            selectedExampleKey: null,
            harnessConfig: nextHarness,
            savedHarnessConfig: cloneHarnessConfig(nextHarness),
            workspaceConfig: nextWorkspace,
            savedWorkspaceConfig: cloneWorkspaceConfig(nextWorkspace),
            isDirty: false,
            builderView: 'config',
          }
        }),

      selectExample: (key, harnessConfig, workspace) =>
        set((s) => {
          const ex = s.examples.find((e) => e.key === key)
          const nextHarness = cloneHarnessConfig(ex?.harness_config ?? harnessConfig)
          const fallbackName = ex?.label || ex?.key || key
          const nextWorkspace = normalizeWorkspaceConfig(ex?.workspace ?? workspace, fallbackName)
          return {
            selectedCustomId: null,
            selectedExampleKey: key,
            harnessConfig: nextHarness,
            savedHarnessConfig: cloneHarnessConfig(nextHarness),
            workspaceConfig: nextWorkspace,
            savedWorkspaceConfig: cloneWorkspaceConfig(nextWorkspace),
            isDirty: false,
            builderView: 'config',
          }
        }),

      startChat:    () => set({ builderView: 'chat' }),
      backToConfig: () => set({ builderView: 'config' }),

      // Custom harnesses
      customHarnesses: [],

      addCustomHarness: (name, harnessConfig, workspace) =>
        set((s) => {
          // New custom harnesses default to lab_agent/default unless explicitly overridden.
          // This avoids inheriting CLI workspace (e.g. hxagent/...) unintentionally.
          const customWs = normalizeWorkspaceConfig(
            workspace ?? defaultCustomWorkspace(),
            name,
          )
          return {
            customHarnesses: [
              ...s.customHarnesses,
              {
                id: newId(),
                name,
                harness_config: cloneHarnessConfig(harnessConfig),
                workspace: customWs,
              },
            ],
          }
        }),

      createAndSelectCustom: (name, harnessConfig, workspace) => {
        const id = newId()
        const next = cloneHarnessConfig(harnessConfig)
        const nextWorkspace = normalizeWorkspaceConfig(
          workspace ?? defaultCustomWorkspace(),
          name,
        )
        set((s) => ({
          customHarnesses:  [
            ...s.customHarnesses,
            { id, name, harness_config: cloneHarnessConfig(next), workspace: cloneWorkspaceConfig(nextWorkspace) },
          ],
          selectedCustomId: id,
          selectedExampleKey: null,
          harnessConfig: next,
          savedHarnessConfig: cloneHarnessConfig(next),
          workspaceConfig: nextWorkspace,
          savedWorkspaceConfig: cloneWorkspaceConfig(nextWorkspace),
          isDirty: false,
          builderView: 'config',
        }))
      },

      removeCustomHarness: (id) =>
        set((s) => ({
          customHarnesses: s.customHarnesses.filter((c) => c.id !== id),
          brokenIds:       s.brokenIds.filter((bid) => bid !== id),
          // If removed item was selected, go back to chat
          selectedCustomId:  s.selectedCustomId === id ? null : s.selectedCustomId,
          builderView:       s.selectedCustomId === id ? 'chat' : s.builderView,
        })),

      renameCustomHarness: (id, name) =>
        set((s) => ({
          customHarnesses: s.customHarnesses.map((c) => c.id === id ? { ...c, name } : c),
        })),

      duplicateCustomHarness: (id) =>
        set((s) => {
          const src = s.customHarnesses.find((c) => c.id === id)
          if (!src) return s
          const copy: CustomHarness = {
            id: newId(),
            name: `${src.name} (copy)`,
            harness_config: cloneHarnessConfig(src.harness_config),
            workspace: normalizeWorkspaceConfig(src.workspace, `${src.name} (copy)`),
          }
          return { customHarnesses: [...s.customHarnesses, copy] }
        }),

      switchToDefault: () =>
        set({ selectedCustomId: null, selectedExampleKey: null, builderView: 'config' }),

      brokenIds: [],
      validateAgainstInstalled: (validTargets) =>
        set((s) => {
          const validSet = new Set(validTargets)
          const brokenIds = s.customHarnesses
            .filter((ch) => getCustomTargets(ch.harness_config).some((t) => !validSet.has(t)))
            .map((ch) => ch.id)
          return { brokenIds }
        }),

      repairFileTargets: (installedProcessors) =>
        set((s) => {
          // Build class-name → freshTarget map from currently installed processors
          const classToTarget = new Map<string, string>()
          for (const { target } of installedProcessors) {
            if (typeof target !== 'string' || !target.startsWith('file://')) continue
            const cls = target.split('::').pop()?.trim() ?? ''
            if (cls) classToTarget.set(cls, target)
          }
          if (classToTarget.size === 0) return s

          const patch: Partial<typeof s> = {}

          // Repair live harnessConfig
          const { repaired: rh, changed: ch } = repairProcessors(s.harnessConfig.processors, classToTarget)
          if (ch) patch.harnessConfig = { ...s.harnessConfig, processors: rh }

          // Repair saved harnessConfig
          const { repaired: rs, changed: cs } = repairProcessors(s.savedHarnessConfig.processors, classToTarget)
          if (cs) patch.savedHarnessConfig = { ...s.savedHarnessConfig, processors: rs }

          // Repair each stored custom harness
          let anyCustomChanged = false
          const customHarnesses = s.customHarnesses.map((c) => {
            const { repaired: rp, changed: cp } = repairProcessors(c.harness_config.processors, classToTarget)
            if (!cp) return c
            anyCustomChanged = true
            return { ...c, harness_config: { ...c.harness_config, processors: rp } }
          })
          if (anyCustomChanged) patch.customHarnesses = customHarnesses

          // Re-run broken-ID validation after repair
          if (ch || cs || anyCustomChanged) {
            const validSet = new Set(installedProcessors.map((p) => p.target))
            const effectiveCustom = (patch.customHarnesses ?? s.customHarnesses)
            patch.brokenIds = effectiveCustom
              .filter((c) => getCustomTargets(c.harness_config).some((t) => !validSet.has(t)))
              .map((c) => c.id)
          }

          return patch
        }),

      successCriteria:    '',
      setSuccessCriteria: (successCriteria) => set({ successCriteria }),
    }),
    {
      name: 'harness-lab-custom',
      // Only persist custom harnesses; all other state is ephemeral
      partialize: (s) => ({ customHarnesses: s.customHarnesses }),
    },
  ),
)
