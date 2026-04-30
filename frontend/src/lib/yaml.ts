/**
 * Minimal YAML serializer/parser for HarnessConfig and ModelConfig.
 *
 * No external dependency — we only support the specific shapes we need.
 */
import type { HarnessConfig, ModelDef, ModelSlot, ModelCapability } from '../api/types'

export interface ModelConfig {
  model_registry: ModelDef[]
  model_slots:    ModelSlot[]
}

const HARNESS_HEADER = [
  '# HarnessX Harness Config',
  '# Python usage:',
  '#   from harnessx import HarnessConfig',
  '#   from harnessx.core.model_config import ModelConfig',
  '#   harness = HarnessConfig.from_yaml(open("this-file").read())',
  '#   model   = ModelConfig.from_yaml_file("model_config.yaml")',
  '#   agent   = model.agentic(harness)',
  '',
].join('\n')

// ── Scalar helpers ────────────────────────────────────────────────────────────

function needsQuotes(s: string): boolean {
  if (s === 'null' || s === 'true' || s === 'false') return true
  if (/^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(s)) return true
  if (/[:#\[\]{},|>&*!,'"%@`]/.test(s) || s.startsWith(' ') || s.endsWith(' ')) return true
  return false
}

function quoteStr(s: string): string {
  if (needsQuotes(s)) return `"${s.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`
  return s
}

function parseScalar(raw: string): unknown {
  const s = raw.trim()
  if (s === 'null' || s === '~' || s === '') return null
  if (s === 'true') return true
  if (s === 'false') return false
  if ((s.startsWith('{') && s.endsWith('}')) || (s.startsWith('[') && s.endsWith(']'))) {
    try { return JSON.parse(s) } catch { /* fall through */ }
  }
  if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) {
    return s.slice(1, -1).replace(/\\"/g, '"').replace(/\\\\/g, '\\')
  }
  if (/^-?\d+$/.test(s)) return parseInt(s, 10)
  if (/^-?\d+\.\d*$/.test(s) || /^-?\.\d+$/.test(s)) return parseFloat(s)
  return s
}

// ── Recursive YAML serializer ─────────────────────────────────────────────────

function serializeYaml(v: unknown, indent: number): string {
  const pad = '  '.repeat(indent)
  if (v === null || v === undefined) return 'null'
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  if (typeof v === 'number') return String(v)
  if (Array.isArray(v)) {
    if (v.length === 0) return '[]\n'
    return '\n' + v.map((item) => {
      const s = serializeYaml(item, indent + 1)
      // s starts with '\n' when item is an object/array
      if (s.startsWith('\n')) {
        const body = s.replace(/^\n/, '')
        const [firstLine, ...rest] = body.split('\n')
        return `${pad}- ${firstLine}\n${rest.map((l) => (l ? `${pad}  ${l}` : '')).join('\n')}`
      }
      return `${pad}- ${s}`
    }).join('\n') + '\n'
  }
  if (typeof v === 'object') {
    const entries = Object.entries(v as Record<string, unknown>)
    if (entries.length === 0) return '{}\n'
    return '\n' + entries.map(([k, val]) => {
      const s = serializeYaml(val, indent + 1)
      if (s.startsWith('\n')) return `${pad}${quoteStr(k)}:${s}`
      return `${pad}${quoteStr(k)}: ${s}`
    }).join('') // each entry already ends with \n
  }
  return quoteStr(String(v)) + '\n'
}

// ── HarnessConfig YAML export ─────────────────────────────────────────────────

export function harnessConfigToYaml(cfg: HarnessConfig): string {
  const procYaml = serializeYaml(cfg.processors, 1)
  const plugYaml = cfg.plugins == null ? 'null\n' : serializeYaml(cfg.plugins, 1)
  const mcpYaml = serializeYaml(cfg.mcp_config ?? { source: 'agent_home' }, 1)
  const spawnLine = `spawn_subagents: ${cfg.spawn_subagents !== false ? 'true' : 'false'}\n`
  return `${HARNESS_HEADER}processors:${procYaml}plugins: ${plugYaml}mcp_config:${mcpYaml}${spawnLine}`
}

// ── HarnessConfig YAML import — indent-based block parser ────────────────────

function getDepth(line: string): number {
  return line.length - line.trimStart().length
}

function parseBlock(lines: string[], startDepth: number): unknown {
  // Determine if this block is a list or a mapping
  const first = lines.find((l) => l.trim() && !l.trim().startsWith('#'))
  if (!first) return null
  if (first.trimStart().startsWith('- ')) {
    return parseListBlock(lines, startDepth)
  }
  return parseMappingBlock(lines, startDepth)
}

function parseListBlock(lines: string[], _startDepth: number): unknown[] {
  const result: unknown[] = []
  let itemLines: string[] = []

  const flush = () => {
    if (itemLines.length === 0) return
    const trimmed = itemLines[0].replace(/^(\s*)- /, '$1  ')
    const rest = itemLines.slice(1)
    const allLines = [trimmed, ...rest]
    const innerDepth = getDepth(trimmed)
    // Check if the item is inline (no children) or a sub-block
    const inlineVal = trimmed.trim()
    const hasColon = inlineVal.includes(':')
    if (allLines.every((l) => !l.trim() || getDepth(l) === innerDepth) && !hasColon) {
      result.push(parseScalar(inlineVal))
    } else {
      result.push(parseBlock(allLines, innerDepth))
    }
    itemLines = []
  }

  for (const line of lines) {
    const t = line.trim()
    if (!t || t.startsWith('#')) continue
    if (t.startsWith('- ') && getDepth(line) === _startDepth) {
      flush()
    }
    itemLines.push(line)
  }
  flush()
  return result
}

function parseMappingBlock(lines: string[], baseDepth: number): Record<string, unknown> {
  const result: Record<string, unknown> = {}
  let curKey: string | null = null
  let curLines: string[] = []

  const flush = () => {
    if (curKey === null) return
    if (curLines.length === 0) {
      result[curKey] = null
    } else if (curLines.length === 1 && !curLines[0].trim().startsWith('- ') && !_hasChild(curLines[0], baseDepth)) {
      result[curKey] = parseScalar(curLines[0].trim())
    } else {
      result[curKey] = parseBlock(curLines, getDepth(curLines[0]))
    }
    curKey = null
    curLines = []
  }

  for (const line of lines) {
    const t = line.trim()
    if (!t || t.startsWith('#')) continue
    const depth = getDepth(line)
    if (depth === baseDepth && !t.startsWith('- ')) {
      const ci = t.indexOf(':')
      if (ci !== -1) {
        flush()
        curKey = parseScalar(t.slice(0, ci)) as string
        const rest = t.slice(ci + 1).trim()
        if (rest) {
          // Inline value — check if it's a block start
          if (rest === '|' || rest === '>') {
            // block scalar — unsupported, skip
          } else {
            curLines = [rest]
          }
        }
        continue
      }
    }
    if (curKey !== null && depth > baseDepth) {
      curLines.push(line)
    }
  }
  flush()
  return result
}

function _hasChild(line: string, baseDepth: number): boolean {
  // A line "has a child" if the next non-empty line has more depth — irrelevant here, simplified
  void line; void baseDepth
  return false
}

export function yamlToHarnessConfig(text: string): Partial<HarnessConfig> {
  // Extract top-level sections
  const processorsLines: string[] = []
  const pluginsLines: string[] = []
  const mcpConfigLines: string[] = []
  let section: 'none' | 'processors' | 'plugins' | 'mcp_config' = 'none'
  let spawn_subagents: boolean | undefined = undefined

  for (const rawLine of text.split('\n')) {
    const line = rawLine.trimEnd()
    const t = line.trim()
    if (!t || t.startsWith('#')) continue
    const depth = getDepth(line)
    if (depth === 0) {
      if (t.startsWith('processors:')) {
        section = 'processors'
        const rest = t.slice('processors:'.length).trim()
        if (rest) processorsLines.push('  ' + rest)
        continue
      }
      if (t.startsWith('plugins:')) {
        section = 'plugins'
        const rest = t.slice('plugins:'.length).trim()
        if (rest) pluginsLines.push('  ' + rest)
        continue
      }
      if (t.startsWith('mcp_config:')) {
        section = 'mcp_config'
        const rest = t.slice('mcp_config:'.length).trim()
        if (rest) mcpConfigLines.push('  ' + rest)
        continue
      }
      if (t.startsWith('spawn_subagents:')) {
        const val = t.slice('spawn_subagents:'.length).trim().toLowerCase()
        spawn_subagents = val !== 'false' && val !== '0' && val !== 'no'
        section = 'none'
        continue
      }
      section = 'none'
      continue
    }
    if (section === 'processors') processorsLines.push(line)
    else if (section === 'plugins') pluginsLines.push(line)
    else if (section === 'mcp_config') mcpConfigLines.push(line)
  }

  let processors: Record<string, unknown>[] = []
  if (processorsLines.length > 0) {
    const parsed = parseBlock(processorsLines, getDepth(processorsLines[0]))
    if (Array.isArray(parsed)) {
      processors = parsed as Record<string, unknown>[]
    }
  }

  let plugins: unknown[] | null = null
  if (pluginsLines.length > 0) {
    const raw = pluginsLines[0].trim()
    if (raw !== 'null' && raw !== '~' && raw !== '') {
      const parsed = parseBlock(pluginsLines, getDepth(pluginsLines[0]))
      plugins = Array.isArray(parsed) ? parsed : null
    }
  }

  let mcp_config: HarnessConfig['mcp_config'] = { source: 'agent_home' }
  if (mcpConfigLines.length > 0) {
    const parsed = parseBlock(mcpConfigLines, getDepth(mcpConfigLines[0]))
    if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
      const sourceRaw = String((parsed as Record<string, unknown>).source ?? '').trim().toLowerCase()
      const source =
        sourceRaw === 'file' || sourceRaw === 'inline' || sourceRaw === 'disabled' || sourceRaw === 'agent_home'
          ? sourceRaw
          : 'agent_home'
      const path = (parsed as Record<string, unknown>).path
      mcp_config = {
        source,
        ...(typeof path === 'string' && path.trim() ? { path } : {}),
      }
    }
  }

  return {
    processors,
    plugins,
    mcp_config,
    ...(spawn_subagents !== undefined ? { spawn_subagents } : {}),
  }
}

// ── Model Config YAML v0.1 ────────────────────────────────────────────────────
//
// Format produced (current v0.1 format):
//
//   schema_version: 2
//   models:
//     - id: claude-sonnet-4-6
//       provider: anthropic
//       _target_: harnessx.providers.anthropic_provider.AnthropicProvider
//       model: claude-sonnet-4-6
//       api_key: sk-ant-...
//       _display_name: Claude Sonnet 4.6
//       _capabilities: [text, code, vl]
//   roles:
//     main:
//       default: claude-sonnet-4-6
//     compact:
//       default: claude-opus-4-6
//       model_ids: [claude-opus-4-6, claude-sonnet-4-6]
//       strategy: fallback
//
// Load in Python: ModelConfig.from_yaml_file(path)
// Current v0.1 and legacy pre-release formats are accepted on import.
//
// ─────────────────────────────────────────────────────────────────────────────

const MODEL_HEADER_V2 = [
  '# HarnessX Model Config v0.1',
  '# Python: ModelConfig.from_yaml_file(path)',
  '',
].join('\n')

const ANTHROPIC_TARGET = 'harnessx.providers.anthropic_provider.AnthropicProvider'
const OPENAI_TARGET    = 'harnessx.providers.openai_provider.OpenAIProvider'
const LITELLM_TARGET   = 'harnessx.providers.litellm_provider.LiteLLMProvider'

function yamlStr(s: string): string {
  if (!s) return '""'
  if (needsQuotes(s)) return `"${s.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`
  return s
}

function unquote(s: string): string {
  const t = s.trim()
  if ((t.startsWith('"') && t.endsWith('"')) || (t.startsWith("'") && t.endsWith("'"))) {
    return t.slice(1, -1).replace(/\\"/g, '"').replace(/\\\\/g, '\\')
  }
  return t
}

function vendorToTarget(vendor: string): string {
  if (vendor === 'anthropic') return ANTHROPIC_TARGET
  if (vendor === 'openai') return OPENAI_TARGET
  return LITELLM_TARGET
}

/** Infer vendor string from a _target_ class path. */
function targetToVendor(target: string): string {
  if (target.includes('AnthropicProvider')) return 'anthropic'
  if (target.includes('OpenAIProvider'))    return 'openai'
  if (target.includes('LiteLLMProvider'))   return 'litellm'  // best-guess; refined from model prefix
  return 'litellm'
}

/** Refine vendor based on model prefix, e.g. "gemini/..." → gemini. */
function refineVendor(vendor: string, modelId: string): string {
  const prefix = modelId.split('/')[0].toLowerCase()
  if (vendor === 'openai' || vendor === 'litellm') return vendor
  if (prefix === 'gemini')   return 'gemini'
  if (prefix === 'deepseek') return 'deepseek'
  if (prefix === 'openai')   return 'openai'
  return vendor
}

export function modelConfigToYaml(cfg: ModelConfig): string {
  const { model_registry, model_slots } = cfg
  const lines: string[] = [MODEL_HEADER_V2, 'schema_version: 2', '']

  // ── models section ──────────────────────────────────────────────────────────
  lines.push('models:')
  for (const m of model_registry) {
    const target = vendorToTarget(m.vendor)
    lines.push(`  - id: ${yamlStr(m.id)}`)
    lines.push(`    provider: ${m.vendor}`)
    lines.push(`    _target_: ${target}`)
    if (m.model_id)  lines.push(`    model: ${yamlStr(m.model_id)}`)
    if (m.api_key)   lines.push(`    api_key: ${yamlStr(m.api_key)}`)
    if (m.base_url)  lines.push(`    api_base: ${yamlStr(m.base_url)}`)
    if (m.extra_headers && Object.keys(m.extra_headers).length > 0)
      lines.push(`    extra_headers: ${JSON.stringify(m.extra_headers)}`)
    if (m.extended_thinking) lines.push('    extended_thinking: true')
    if (typeof m.thinking_budget_tokens === 'number' && Number.isFinite(m.thinking_budget_tokens))
      lines.push(`    thinking_budget_tokens: ${Math.floor(m.thinking_budget_tokens)}`)
    if (m.reasoning_effort)
      lines.push(`    reasoning_effort: ${yamlStr(m.reasoning_effort)}`)
    if (m.reasoning_summary)
      lines.push('    reasoning_summary: true')
    if (m.display_name && m.display_name !== m.model_id)
      lines.push(`    _display_name: ${yamlStr(m.display_name)}`)
    if (m.capabilities.length > 0)
      lines.push(`    _capabilities: [${m.capabilities.join(', ')}]`)
    lines.push('')
  }

  // ── roles section ───────────────────────────────────────────────────────────
  lines.push('roles:')
  for (const slot of model_slots) {
    if (slot.model_ids.length === 0) continue
    const primary = slot.model_ids[0]
    lines.push(`  ${slot.slot_name}:`)
    lines.push(`    default: ${yamlStr(primary)}`)
    if (slot.model_ids.length > 1) {
      lines.push(`    model_ids: [${slot.model_ids.map(yamlStr).join(', ')}]`)
      if (slot.strategy !== 'primary')
        lines.push(`    strategy: ${slot.strategy}`)
    }
  }

  lines.push('')
  return lines.join('\n')
}

// ── Import: parse YAML → {model_registry, model_slots} ─────────────────────
//
// Supports legacy pre-release (_target_ per-role) and current v0.1 (models registry + roles) formats.
// Format is detected automatically.

/** Short unique ID for imported registry entries. */
function genImportId(slotName: string, index: number): string {
  return `imp-${slotName}-${index}`
}

/**
 * Parse a Model Config YAML string into {model_registry, model_slots}.
 *
 * Current v0.1: produced by modelConfigToYaml(); has top-level `models:` and `roles:` keys.
 * Legacy pre-release: _target_-per-role format; still accepted for backward compatibility.
 */
export function yamlToModelConfig(text: string): Partial<ModelConfig> {
  const isV2 = /^models:/m.test(text) && /^roles:/m.test(text)
  return isV2 ? _parseV2(text) : _parseV1(text)
}

// ── current-format parser ─────────────────────────────────────────────────────

function _parseV2(text: string): Partial<ModelConfig> {
  const models: ModelDef[]  = []
  const slots:  ModelSlot[] = []

  type Section = 'none' | 'models' | 'roles'
  let section: Section = 'none'

  // Current model item being accumulated
  let curModel: Record<string, string> | null = null
  // Current role slot being accumulated
  let curRole:      string | null = null
  let curRoleProps: Record<string, string> = {}

  const flushModel = () => {
    if (!curModel) return
    const id          = unquote(curModel['id']           ?? '')
    const modelId     = unquote(curModel['model']        ?? '')
    const apiKey      = unquote(curModel['api_key']      ?? '')
    const baseUrl     = unquote(curModel['api_base']     ?? curModel['base_url'] ?? '')
    const displayName = unquote(curModel['_display_name'] ?? '') || modelId || id
    let extraHeaders: Record<string, string> | undefined
    const rawHeaders = curModel['extra_headers'] ?? curModel['default_headers'] ?? ''
    if (rawHeaders) {
      try {
        const parsed = JSON.parse(rawHeaders)
        if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
          extraHeaders = Object.fromEntries(
            Object.entries(parsed).map(([k, v]) => [String(k), String(v)])
          )
        }
      } catch {
        // keep undefined for invalid payload; parser should be forgiving
      }
    }
    const rawVendor   = unquote(curModel['provider']     ?? curModel['_vendor']  ?? '')
    const target      = unquote(curModel['_target_']     ?? '')
    const vendor      = rawVendor || refineVendor(targetToVendor(target), modelId)
    const extThinking = unquote(curModel['extended_thinking'] ?? '').toLowerCase() === 'true'
    const rawBudget = unquote(curModel['thinking_budget_tokens'] ?? '')
    const budget = rawBudget ? Number(rawBudget) : undefined
    const rawEffort = unquote(curModel['reasoning_effort'] ?? '')
    const reasonSummary = unquote(curModel['reasoning_summary'] ?? '').toLowerCase() === 'true'
    const rawCaps     = curModel['_capabilities'] ?? ''
    const capabilities = rawCaps
      ? rawCaps.replace(/^\[/, '').replace(/\]$/, '').split(',')
          .map((s) => s.trim()).filter(Boolean) as ModelCapability[]
      : []
    if (id) {
      models.push({ id, display_name: displayName,
                    vendor: refineVendor(vendor, modelId),
                    model_id: modelId, api_key: apiKey, base_url: baseUrl, extra_headers: extraHeaders, capabilities,
                    extended_thinking: extThinking || undefined,
                    thinking_budget_tokens: Number.isFinite(budget) ? budget : undefined,
                    reasoning_effort: (
                      rawEffort === 'low' || rawEffort === 'medium' || rawEffort === 'high'
                    ) ? rawEffort : undefined,
                    reasoning_summary: reasonSummary || undefined,
                  })
    }
    curModel = null
  }

  const flushRole = () => {
    if (!curRole) return
    const def = unquote(curRoleProps['default'] ?? '')
    if (def) {
      const rawIds = curRoleProps['model_ids'] ?? ''
      const modelIds = rawIds
        ? rawIds.replace(/^\[/, '').replace(/\]$/, '').split(',')
            .map((s) => unquote(s.trim())).filter(Boolean)
        : [def]
      const strategy = (curRoleProps['strategy'] as ModelSlot['strategy']) ?? 'primary'
      slots.push({ slot_name: curRole, model_ids: modelIds, strategy })
    }
    curRole = null
    curRoleProps = {}
  }

  for (const rawLine of text.split('\n')) {
    const line    = rawLine.trimEnd()
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) continue

    const depth = line.length - line.trimStart().length

    // Top-level section markers
    if (depth === 0) {
      if (trimmed === 'models:') { flushModel(); flushRole(); section = 'models'; continue }
      if (trimmed === 'roles:')  { flushModel(); flushRole(); section = 'roles';  continue }
      continue  // schema_version, comments, etc.
    }

    if (section === 'models') {
      if (depth === 2 && trimmed.startsWith('- ')) {
        flushModel()
        curModel = {}
        const rest = trimmed.slice(2)
        const ci = rest.indexOf(':')
        if (ci !== -1) curModel[rest.slice(0, ci).trim()] = rest.slice(ci + 1).trim()
        continue
      }
      if (depth === 4 && curModel) {
        const ci = trimmed.indexOf(':')
        if (ci !== -1) curModel[trimmed.slice(0, ci).trim()] = trimmed.slice(ci + 1).trim()
        continue
      }
    }

    if (section === 'roles') {
      if (depth === 2) {
        flushRole()
        const ci = trimmed.indexOf(':')
        if (ci !== -1) { curRole = trimmed.slice(0, ci).trim(); curRoleProps = {} }
        continue
      }
      if (depth === 4 && curRole) {
        const ci = trimmed.indexOf(':')
        if (ci !== -1) curRoleProps[trimmed.slice(0, ci).trim()] = trimmed.slice(ci + 1).trim()
        continue
      }
    }
  }
  flushModel()
  flushRole()

  return {
    model_registry: models.length > 0 ? models : undefined,
    model_slots:    slots.length  > 0 ? slots   : undefined,
  }
}

// ── legacy parser (_target_-per-role format) ─────────────────────────────────

function _parseV1(text: string): Partial<ModelConfig> {
  const models: ModelDef[] = []
  const slots: ModelSlot[]  = []

  // ── Phase 1: split top-level slot blocks ─────────────────────────────────
  // Top-level key pattern: /^[a-zA-Z_][a-zA-Z0-9_]*:/
  const blockLines: Record<string, string[]> = {}
  const blockOrder: string[] = []
  let curKey: string | null = null

  for (const rawLine of text.split('\n')) {
    const line    = rawLine.trimEnd()
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('#')) continue

    // Top-level key: no leading space, matches identifier:
    if (/^[a-zA-Z_][a-zA-Z0-9_]*:/.test(line)) {
      curKey = line.slice(0, line.indexOf(':')).trim()
      blockLines[curKey] = []
      blockOrder.push(curKey)
      const rest = line.slice(line.indexOf(':') + 1).trim()
      if (rest) blockLines[curKey].push(`  ${rest}`)
      continue
    }

    if (curKey && (line.startsWith(' ') || line.startsWith('\t'))) {
      blockLines[curKey].push(line)
    }
  }

  // ── Phase 2: parse each block ─────────────────────────────────────────────
  for (const slotName of blockOrder) {
    const lines = blockLines[slotName] ?? []
    if (lines.length === 0) continue

    // Extract flat key-value pairs (depth-1 only)
    const flat: Record<string, string> = {}
    for (const raw of lines) {
      const t = raw.trim()
      if (!t || t.startsWith('#') || t.startsWith('-')) continue
      const ci = t.indexOf(':')
      if (ci === -1) continue
      const k = t.slice(0, ci).trim()
      const v = t.slice(ci + 1).trim()
      if (v) flat[k] = v
    }

    const target = unquote(flat['_target_'] ?? '')

    // ── ProviderGroup ──────────────────────────────────────────────────────
    if (target.includes('ProviderGroup')) {
      const strategy: ModelSlot['strategy'] =
        flat['strategy'] === 'round_robin' ? 'round_robin' : 'fallback'
      const slotModelIds: string[] = []

      // Parse entries list
      // entries start at depth-1 as `  entries:` then `    - type: ...`
      let inEntries = false
      let curEntry: Record<string, string> | null = null
      let inModels  = false
      let entryIdx  = 0

      const flushEntry = () => {
        if (!curEntry) return
        const entryType   = curEntry['type'] ?? 'litellm'
        const entryVendor = entryType === 'anthropic'
          ? 'anthropic'
          : entryType === 'openai'
            ? 'openai'
            : 'litellm'
        const modelId     = unquote(curEntry['_model']        ?? '')
        const apiKey      = unquote(curEntry['api_key']       ?? '')
        const baseUrl     = unquote(curEntry['api_base']      ?? '')
        let extraHeaders: Record<string, string> | undefined
        const rawHeaders = curEntry['extra_headers'] ?? curEntry['default_headers'] ?? ''
        if (rawHeaders) {
          try {
            const parsed = JSON.parse(rawHeaders)
            if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
              extraHeaders = Object.fromEntries(
                Object.entries(parsed).map(([k, v]) => [String(k), String(v)])
              )
            }
          } catch {
            // ignore malformed headers
          }
        }
        const displayName = unquote(curEntry['_display_name'] ?? '') || modelId
        const extThinking = unquote(curEntry['extended_thinking'] ?? '').toLowerCase() === 'true'
        const rawBudget = unquote(curEntry['thinking_budget_tokens'] ?? '')
        const budget = rawBudget ? Number(rawBudget) : undefined
        const rawEffort = unquote(curEntry['reasoning_effort'] ?? '')
        const reasonSummary = unquote(curEntry['reasoning_summary'] ?? '').toLowerCase() === 'true'
        const rawCaps     = curEntry['_capabilities'] ?? ''
        const capabilities = rawCaps
          ? rawCaps.replace(/^\[/, '').replace(/\]$/, '').split(',').map((s) => s.trim()).filter(Boolean) as ModelCapability[]
          : []
        if (modelId) {
          const id = genImportId(slotName, entryIdx++)
          models.push({
            id,
            display_name: displayName,
            vendor:       refineVendor(entryVendor, modelId),
            model_id:     modelId,
            api_key:      apiKey,
            base_url:     baseUrl,
            extra_headers: extraHeaders,
            capabilities,
            extended_thinking: extThinking || undefined,
            thinking_budget_tokens: Number.isFinite(budget) ? budget : undefined,
            reasoning_effort: (
              rawEffort === 'low' || rawEffort === 'medium' || rawEffort === 'high'
            ) ? rawEffort : undefined,
            reasoning_summary: reasonSummary || undefined,
          })
          slotModelIds.push(id)
        }
        curEntry = null
        inModels = false
      }

      for (const raw of lines) {
        const t     = raw.trim()
        const depth = raw.length - raw.trimStart().length
        if (!t || t.startsWith('#')) continue

        if (depth === 2 && t === 'entries:') { inEntries = true; continue }
        if (!inEntries) continue

        // depth-2 list item → new entry
        if (depth === 2 && t.startsWith('- ')) {
          flushEntry()
          curEntry = {}
          const rest = t.slice(2)
          const ci = rest.indexOf(':')
          if (ci !== -1) {
            curEntry[rest.slice(0, ci).trim()] = rest.slice(ci + 1).trim()
          }
          continue
        }
        if (!curEntry) continue

        // depth-4 key-value inside an entry
        if (depth === 4) {
          const ci = t.indexOf(':')
          if (ci === -1) continue
          const k = t.slice(0, ci).trim()
          const v = t.slice(ci + 1).trim()
          if (k === 'models') { inModels = true; continue }
          if (v) curEntry[k] = v
        }
        // depth-4 list item inside models
        if (depth === 4 && t.startsWith('- ') && inModels) {
          const rest = t.slice(2)
          const ci = rest.indexOf(':')
          if (ci !== -1 && rest.slice(0, ci).trim() === 'model') {
            curEntry['_model'] = rest.slice(ci + 1).trim()
          }
          continue
        }
        // depth-6: model item properties (default, _display_name, _capabilities, …)
        if (depth === 6 && inModels) {
          const ci = t.indexOf(':')
          if (ci === -1) continue
          const k = t.slice(0, ci).trim()
          const v = t.slice(ci + 1).trim()
          if (k === 'model')         curEntry['_model']        = v
          if (k === '_display_name') curEntry['_display_name'] = v
          if (k === '_capabilities') curEntry['_capabilities'] = v
        }
      }
      flushEntry()

      if (slotModelIds.length > 0) {
        slots.push({ slot_name: slotName, model_ids: slotModelIds, strategy })
      }
      continue
    }

    // ── Simple provider (Anthropic / LiteLLM) ────────────────────────────
    if (target) {
      const modelId     = unquote(flat['model']          ?? '')
      const apiKey      = unquote(flat['api_key']        ?? '')
      const baseUrl     = unquote(flat['base_url']       ?? flat['api_base'] ?? '')
      let extraHeaders: Record<string, string> | undefined
      const rawHeaders = flat['extra_headers'] ?? flat['default_headers'] ?? ''
      if (rawHeaders) {
        try {
          const parsed = JSON.parse(rawHeaders)
          if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
            extraHeaders = Object.fromEntries(
              Object.entries(parsed).map(([k, v]) => [String(k), String(v)])
            )
          }
        } catch {
          // ignore malformed headers
        }
      }
      // Restore frontend metadata from _-prefixed keys (silently added on export)
      const displayName = unquote(flat['_display_name']  ?? '') || modelId || slotName
      const rawVendor   = unquote(flat['_vendor']        ?? '')
      const vendor      = rawVendor || refineVendor(targetToVendor(target), modelId)
      const extThinking = unquote(flat['extended_thinking'] ?? '').toLowerCase() === 'true'
      const rawBudget = unquote(flat['thinking_budget_tokens'] ?? '')
      const budget = rawBudget ? Number(rawBudget) : undefined
      const rawEffort = unquote(flat['reasoning_effort'] ?? '')
      const reasonSummary = unquote(flat['reasoning_summary'] ?? '').toLowerCase() === 'true'
      const rawCaps     = flat['_capabilities'] ?? ''
      const capabilities = rawCaps
        ? rawCaps.replace(/^\[/, '').replace(/\]$/, '').split(',').map((s) => s.trim()).filter(Boolean) as ModelCapability[]
        : []
      const id = genImportId(slotName, 0)
      models.push({
        id,
        display_name: displayName,
        vendor,
        model_id:     modelId,
        api_key:      apiKey,
        base_url:     baseUrl,
        extra_headers: extraHeaders,
        capabilities,
        extended_thinking: extThinking || undefined,
        thinking_budget_tokens: Number.isFinite(budget) ? budget : undefined,
        reasoning_effort: (
          rawEffort === 'low' || rawEffort === 'medium' || rawEffort === 'high'
        ) ? rawEffort : undefined,
        reasoning_summary: reasonSummary || undefined,
      })
      slots.push({
        slot_name: slotName,
        model_ids: [id],
        strategy:  'primary',
      })
    }
  }

  return {
    model_registry: models.length > 0 ? models : undefined,
    model_slots:    slots.length  > 0 ? slots   : undefined,
  }
}
