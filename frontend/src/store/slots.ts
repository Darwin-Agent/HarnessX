import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type {
  VendorInfo, ToolInfo, SkillInfo, SlotConfig,
  McpServerConfig, ModelDef, ModelSlot,
} from '../api/types'

// ── Defaults ──────────────────────────────────────────────────────────────────

export const DEFAULT_MODEL_ID = 'default-main'

const DEFAULT_MODEL_DEF: ModelDef = {
  id:           DEFAULT_MODEL_ID,
  display_name: 'Claude Sonnet 4.6',
  vendor:       'anthropic',
  model_id:     'claude-sonnet-4-6',
  api_key:      '',
  base_url:     '',
  extra_headers: {},
  capabilities: ['text', 'code', 'vl'],
  extended_thinking: false,
  thinking_budget_tokens: 10000,
}

const DEFAULT_MODEL_SLOTS: ModelSlot[] = [
  { slot_name: 'main', model_ids: [DEFAULT_MODEL_ID], strategy: 'primary' },
]

function normalizeModelDef(model: ModelDef): ModelDef {
  if (model.vendor === 'anthropic') {
    return {
      ...model,
      reasoning_effort: undefined,
      reasoning_summary: undefined,
    }
  }
  return model
}

/** Generate a short unique model def ID. */
function genModelId(): string {
  return `m${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`
}

// ── Well-known slot names that carry usage hints ───────────────────────────────
export const SLOT_HINTS: Record<string, string> = {
  main:    'Primary model for all harness steps.',
  compact: 'Lighter model for compaction & summarisation. Defaults to main if not set.',
  judge:   'Used by evaluation, self-verify, and scoring processors.',
}

// ── Well-known strategy labels ────────────────────────────────────────────────
export type SlotStrategy   = 'primary' | 'fallback' | 'round_robin'

// ─────────────────────────────────────────────────────────────────────────────

interface SlotsState {
  // ── Loaded metadata ─────────────────────────────────────────────────────────
  vendors:    VendorInfo[]
  toolInfos:  ToolInfo[]
  skillInfos: SkillInfo[]
  setVendors:    (v: VendorInfo[]) => void
  setToolInfos:  (t: ToolInfo[]) => void
  setSkillInfos: (s: SkillInfo[]) => void

  // ── Model Registry ───────────────────────────────────────────────────────────
  modelRegistry: ModelDef[]
  addModelDef:    (m: Omit<ModelDef, 'id'>) => ModelDef
  removeModelDef: (id: string) => void
  updateModelDef: (id: string, updates: Partial<ModelDef>) => void
  /** Replace the full registry + slots (e.g., after YAML import). */
  importModelConfig: (registry: ModelDef[], slots: ModelSlot[]) => void

  // ── Model Slots ──────────────────────────────────────────────────────────────
  modelSlots: ModelSlot[]
  upsertModelSlot: (slot: ModelSlot) => void
  removeModelSlot: (slot_name: string) => void

  // ── AGENT_HOME context ───────────────────────────────────────────────────────
  agentId:         string
  currentProject:  string
  setAgentId:      (id: string) => void
  setCurrentProject: (p: string) => void

  // ── Sandbox & Workspace ──────────────────────────────────────────────────────
  sandboxType:     'local' | 'remote'
  sandboxUrl:      string
  workspaceDir:    string
  setSandboxType:  (t: 'local' | 'remote') => void
  setSandboxUrl:   (u: string) => void
  setWorkspaceDir: (d: string) => void

  // ── Tool / skill selection ───────────────────────────────────────────────────
  enabledTools:  string[] | null   // null = all
  enabledSkills: string[] | null   // null = all
  setEnabledTools:  (t: string[] | null) => void
  setEnabledSkills: (s: string[] | null) => void

  // ── MCP servers ──────────────────────────────────────────────────────────────
  mcpServers: McpServerConfig[]
  setMcpServers:    (servers: McpServerConfig[]) => void
  addMcpServer:    (cfg: McpServerConfig) => void
  removeMcpServer: (id: string) => void
  updateMcpServer: (id: string, updates: Partial<McpServerConfig>) => void

  // ── Helpers ──────────────────────────────────────────────────────────────────
  /** Returns the primary model_id string for the main slot. */
  getMainModel:   () => string
  /** Serialises main slot into the legacy SlotConfig for the run request. */
  toSlotConfig:   () => SlotConfig
  /** Serialises all slots into a provider_config dict (_target_ format) for the run request. */
  toModelConfigPayload: () => Record<string, unknown>
  /** Returns an error string if config is incomplete, null if OK. */
  validateLaunch: () => string | null
  /** Resolve a slot's primary ModelDef, if any. */
  resolveSlot:    (slot_name: string) => ModelDef | undefined
}

// ─────────────────────────────────────────────────────────────────────────────

export const useSlotsStore = create<SlotsState>()(
  persist(
    (set, get) => ({

      // ── Metadata ─────────────────────────────────────────────────────────────
      vendors:    [],
      toolInfos:  [],
      skillInfos: [],
      setVendors:    (vendors)    => set({ vendors }),
      setToolInfos:  (toolInfos)  => set({ toolInfos }),
      setSkillInfos: (skillInfos) => {
        const { enabledSkills } = get()
        // When enabledSkills is a list (not null/all), auto-enable any newly
        // discovered skills so they are not silently filtered out by a stale
        // allowlist (e.g. after installing a plugin with new skills).
        if (enabledSkills !== null) {
          const known = new Set(enabledSkills)
          const added = skillInfos.filter((s) => !known.has(s.name)).map((s) => s.name)
          if (added.length > 0) {
            const merged = [...enabledSkills, ...added]
            // If all skills are now enabled, collapse back to null
            set({
              skillInfos,
              enabledSkills: merged.length >= skillInfos.length ? null : merged,
            })
            return
          }
        }
        set({ skillInfos })
      },

      // ── Model Registry ────────────────────────────────────────────────────────
      modelRegistry: [{ ...DEFAULT_MODEL_DEF }],

      addModelDef: (m) => {
        const def: ModelDef = normalizeModelDef({ ...m, id: genModelId() })
        set((s) => ({ modelRegistry: [...s.modelRegistry, def] }))
        return def
      },

      removeModelDef: (id) =>
        set((s) => ({
          modelRegistry: s.modelRegistry.filter((m) => m.id !== id),
          // Also scrub from all slots
          modelSlots: s.modelSlots.map((sl) => ({
            ...sl,
            model_ids: sl.model_ids.filter((mid) => mid !== id),
          })),
        })),

      updateModelDef: (id, updates) =>
        set((s) => ({
          modelRegistry: s.modelRegistry.map((m) =>
            m.id === id ? normalizeModelDef({ ...m, ...updates }) : m
          ),
        })),

      importModelConfig: (registry, slots) =>
        set(() => {
          const safeRegistry = (registry.length > 0 ? registry : [{ ...DEFAULT_MODEL_DEF }])
            .map((m) => normalizeModelDef(m))
          const knownIds = new Set(safeRegistry.map((m) => m.id))

          const safeSlots = slots
            .map((slot) => ({
              ...slot,
              model_ids: slot.model_ids.filter((id) => knownIds.has(id)),
            }))
            .filter((slot) => slot.model_ids.length > 0)

          const fallbackMainId = safeRegistry[0]?.id ?? DEFAULT_MODEL_ID
          const mainIdx = safeSlots.findIndex((slot) => slot.slot_name === 'main')

          if (mainIdx < 0) {
            safeSlots.unshift({
              slot_name: 'main',
              model_ids: [fallbackMainId],
              strategy: 'primary',
            })
          } else if (safeSlots[mainIdx].model_ids.length === 0) {
            safeSlots[mainIdx] = {
              ...safeSlots[mainIdx],
              model_ids: [fallbackMainId],
            }
          }

          return { modelRegistry: safeRegistry, modelSlots: safeSlots }
        }),

      // ── Model Slots ───────────────────────────────────────────────────────────
      modelSlots: [...DEFAULT_MODEL_SLOTS],

      upsertModelSlot: (slot) =>
        set((s) => {
          const exists = s.modelSlots.some((sl) => sl.slot_name === slot.slot_name)
          return {
            modelSlots: exists
              ? s.modelSlots.map((sl) => sl.slot_name === slot.slot_name ? slot : sl)
              : [...s.modelSlots, slot],
          }
        }),

      removeModelSlot: (slot_name) =>
        set((s) => ({
          modelSlots: s.modelSlots.filter((sl) => sl.slot_name !== slot_name),
        })),

      // ── AGENT_HOME context ────────────────────────────────────────────────────
      agentId:          'hxagent',
      currentProject:   'hxproject',
      setAgentId:         (agentId)         => set({ agentId }),
      setCurrentProject:  (currentProject)  => set({ currentProject }),

      // ── Sandbox & Workspace ───────────────────────────────────────────────────
      sandboxType:  'local',
      sandboxUrl:   '',
      workspaceDir: '',
      setSandboxType:  (sandboxType)  => set({ sandboxType }),
      setSandboxUrl:   (sandboxUrl)   => set({ sandboxUrl }),
      setWorkspaceDir: (workspaceDir) => set({ workspaceDir }),

      // ── Tool / Skill selection ────────────────────────────────────────────────
      enabledTools:     null,
      enabledSkills:    null,
      setEnabledTools:  (enabledTools)  => set({ enabledTools }),
      setEnabledSkills: (enabledSkills) => set({ enabledSkills }),

      // ── MCP servers ───────────────────────────────────────────────────────────
      mcpServers:      [],
      setMcpServers:   (mcpServers) => set({ mcpServers }),
      addMcpServer:    (cfg)     => set((s) => ({ mcpServers: [...s.mcpServers, cfg] })),
      removeMcpServer: (id)      => set((s) => ({ mcpServers: s.mcpServers.filter((m) => m.id !== id) })),
      updateMcpServer: (id, upd) => set((s) => ({
        mcpServers: s.mcpServers.map((m) => (m.id === id ? { ...m, ...upd } : m)),
      })),

      // ── Helpers ───────────────────────────────────────────────────────────────
      resolveSlot: (slot_name) => {
        const { modelSlots, modelRegistry } = get()
        const slot    = modelSlots.find((s) => s.slot_name === slot_name)
        const primary = slot?.model_ids[0]
        if (primary) return modelRegistry.find((m) => m.id === primary)
        // Any slot with no assigned models falls back to the main slot
        if (slot_name !== 'main') {
          const mainSlot = modelSlots.find((s) => s.slot_name === 'main')
          const mainId   = mainSlot?.model_ids[0]
          return mainId ? modelRegistry.find((m) => m.id === mainId) : undefined
        }
        return undefined
      },

      getMainModel: () => {
        const main = get().resolveSlot('main')
        return main?.model_id ?? 'claude-sonnet-4-6'
      },

      toSlotConfig: () => {
        const { enabledTools, enabledSkills, sandboxType, sandboxUrl } = get()
        return {
          enabled_tools:  enabledTools,
          enabled_skills: enabledSkills,
          sandbox_type:   sandboxType,
          sandbox_url:    sandboxUrl || null,
        }
      },

      toModelConfigPayload: () => {
        const { modelRegistry, modelSlots } = get()
        const ANTHROPIC_TARGET = 'harnessx.providers.anthropic_provider.AnthropicProvider'
        const OPENAI_TARGET    = 'harnessx.providers.openai_provider.OpenAIProvider'
        const LITELLM_TARGET   = 'harnessx.providers.litellm_provider.LiteLLMProvider'
        const GROUP_TARGET     = 'harnessx.providers.group.ProviderGroup'
        const targetForVendor = (vendor: string): string => {
          if (vendor === 'anthropic') return ANTHROPIC_TARGET
          if (vendor === 'openai') return OPENAI_TARGET
          return LITELLM_TARGET
        }
        const entryTypeForVendor = (vendor: string): 'anthropic' | 'openai' | 'litellm' => {
          if (vendor === 'anthropic') return 'anthropic'
          if (vendor === 'openai') return 'openai'
          return 'litellm'
        }

        const payload: Record<string, unknown> = {}

        for (const slot of modelSlots) {
          if (slot.model_ids.length === 0) continue
          const mods = slot.model_ids
            .map((id) => modelRegistry.find((m) => m.id === id))
            .filter((m): m is ModelDef => !!m)
          if (mods.length === 0) continue

          // Allow multiple models to share one provider connection config:
          // if a model leaves api_key/base_url empty, inherit from same-vendor
          // models in the same slot (first non-empty occurrence).
          const sharedByVendor = new Map<string, { api_key?: string; base_url?: string }>()
          for (const m of mods) {
            const prev = sharedByVendor.get(m.vendor) ?? {}
            const key = m.api_key?.trim()
            const url = m.base_url?.trim()
            if (!prev.api_key && key) prev.api_key = key
            if (!prev.base_url && url) prev.base_url = url
            sharedByVendor.set(m.vendor, prev)
          }
          const effApiKey = (m: ModelDef): string => {
            const own = m.api_key?.trim()
            if (own) return own
            return sharedByVendor.get(m.vendor)?.api_key ?? ''
          }
          const effBaseUrl = (m: ModelDef): string => {
            const own = m.base_url?.trim()
            if (own) return own
            return sharedByVendor.get(m.vendor)?.base_url ?? ''
          }

          if (mods.length === 1) {
            const m = mods[0]
            const spec: Record<string, unknown> = {
              _target_: targetForVendor(m.vendor),
            }
            const apiKey = effApiKey(m)
            const baseUrl = effBaseUrl(m)
            if (m.model_id) spec['model']    = m.model_id
            if (apiKey)  spec['api_key']  = apiKey
            if (baseUrl) {
              if (m.vendor === 'anthropic') spec['base_url'] = baseUrl
              else                          spec['api_base'] = baseUrl
            }
            if (m.extra_headers && Object.keys(m.extra_headers).length > 0) {
              if (m.vendor === 'anthropic') spec['default_headers'] = m.extra_headers
              else                          spec['extra_headers'] = m.extra_headers
            }
            if (m.vendor === 'anthropic') {
              if (m.extended_thinking) spec['extended_thinking'] = true
              if (typeof m.thinking_budget_tokens === 'number' && Number.isFinite(m.thinking_budget_tokens) && m.thinking_budget_tokens > 0) {
                spec['thinking_budget_tokens'] = Math.floor(m.thinking_budget_tokens)
              }
            } else if (m.reasoning_effort) {
              spec['reasoning_effort'] = m.reasoning_effort
            }
            if (m.vendor !== 'anthropic' && m.reasoning_summary) {
              spec['reasoning_summary'] = true
            }
            payload[slot.slot_name] = spec
          } else {
            const entries = mods.map((m, idx) => ({
              type:   entryTypeForVendor(m.vendor),
              ...(effApiKey(m)  ? { api_key:  effApiKey(m) }  : {}),
              ...(effBaseUrl(m) ? { api_base: effBaseUrl(m) } : {}),
              ...(m.extra_headers && Object.keys(m.extra_headers).length > 0
                ? (m.vendor === 'anthropic'
                    ? { default_headers: m.extra_headers }
                    : { extra_headers: m.extra_headers })
                : {}),
              ...(m.vendor === 'anthropic' && m.extended_thinking ? { extended_thinking: true } : {}),
              ...(m.vendor === 'anthropic' && typeof m.thinking_budget_tokens === 'number' && Number.isFinite(m.thinking_budget_tokens) && m.thinking_budget_tokens > 0
                ? { thinking_budget_tokens: Math.floor(m.thinking_budget_tokens) }
                : {}),
              ...(m.vendor !== 'anthropic' && m.reasoning_effort ? { reasoning_effort: m.reasoning_effort } : {}),
              ...(m.vendor !== 'anthropic' && m.reasoning_summary ? { reasoning_summary: true } : {}),
              models: [{ model: m.model_id, default: idx === 0 }],
            }))
            const spec: Record<string, unknown> = {
              _target_:    GROUP_TARGET,
              max_retries: 5,
              entries,
            }
            if (slot.strategy === 'round_robin') spec['strategy'] = 'round_robin'
            payload[slot.slot_name] = spec
          }
        }

        // Safety net: always send a usable main slot.
        // This prevents backend "ModelConfig requires a 'main' provider" failures
        // when local persisted state is temporarily empty/corrupted.
        if (!payload.main) {
          const mainSlot = modelSlots.find((s) => s.slot_name === 'main')
          const mainId = mainSlot?.model_ids[0]
          const resolvedMain =
            (mainId ? modelRegistry.find((m) => m.id === mainId) : undefined)
            ?? modelRegistry[0]
            ?? DEFAULT_MODEL_DEF

          const spec: Record<string, unknown> = {
            _target_: targetForVendor(resolvedMain.vendor),
          }
          if (resolvedMain.model_id) spec['model'] = resolvedMain.model_id
          if (resolvedMain.api_key) spec['api_key'] = resolvedMain.api_key
          if (resolvedMain.base_url) {
            if (resolvedMain.vendor === 'anthropic') spec['base_url'] = resolvedMain.base_url
            else                                     spec['api_base'] = resolvedMain.base_url
          }
          if (resolvedMain.extra_headers && Object.keys(resolvedMain.extra_headers).length > 0) {
            if (resolvedMain.vendor === 'anthropic') spec['default_headers'] = resolvedMain.extra_headers
            else                                     spec['extra_headers'] = resolvedMain.extra_headers
          }
          if (resolvedMain.vendor === 'anthropic') {
            if (resolvedMain.extended_thinking) spec['extended_thinking'] = true
            if (typeof resolvedMain.thinking_budget_tokens === 'number' && Number.isFinite(resolvedMain.thinking_budget_tokens) && resolvedMain.thinking_budget_tokens > 0) {
              spec['thinking_budget_tokens'] = Math.floor(resolvedMain.thinking_budget_tokens)
            }
          } else if (resolvedMain.reasoning_effort) {
            spec['reasoning_effort'] = resolvedMain.reasoning_effort
          }
          if (resolvedMain.vendor !== 'anthropic' && resolvedMain.reasoning_summary) {
            spec['reasoning_summary'] = true
          }
          payload.main = spec
        }

        return payload
      },

      validateLaunch: () => {
        const { sandboxType, sandboxUrl, vendors } = get()
        const main = get().resolveSlot('main')
        if (!main?.api_key?.trim()) {
          // Allow if the server-side env var for this vendor is present
          const vendorInfo = vendors.find((v) => v.id === main?.vendor)
          if (!vendorInfo?.env_key_set) {
            return 'API key is required for the main provider'
          }
        }
        if (sandboxType === 'remote' && !sandboxUrl.trim()) {
          return 'Sandbox URL is required for remote sandbox'
        }
        return null
      },
    }),
    {
      name: 'harness-slots',
      partialize: (s) => ({
        modelRegistry:  s.modelRegistry,
        modelSlots:     s.modelSlots,
        agentId:        s.agentId,
        currentProject: s.currentProject,
        sandboxType:    s.sandboxType,
        sandboxUrl:     s.sandboxUrl,
        workspaceDir:   s.workspaceDir,
        enabledTools:   s.enabledTools,
        enabledSkills:  s.enabledSkills,
        mcpServers:     s.mcpServers,
      }),
    },
  ),
)
