import type { CustomHarness, ExampleDescriptor } from '../api/types'

export const LAB_AGENT_ID = 'lab_agent'

function hashName(input: string): string {
  // FNV-1a 32-bit
  let h = 0x811c9dc5
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i)
    h = Math.imul(h, 0x01000193) >>> 0
  }
  return h.toString(36)
}

export function slugifyHarnessProject(name: string): string {
  const raw = (name || '').trim()
  const slug = raw
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80)
  if (slug) return slug
  return `harness-${hashName(raw || 'default')}`
}

export function workspaceFromHarnessName(harnessName: string): { agentId: string; project: string } {
  return {
    agentId: LAB_AGENT_ID,
    project: slugifyHarnessProject(harnessName),
  }
}

export function defaultCustomWorkspace(): { agent_id: string; project: string } {
  return {
    agent_id: LAB_AGENT_ID,
    project: 'default',
  }
}

export function resolveBuilderHarnessName(input: {
  selectedCustomId: string | null
  selectedExampleKey: string | null
  customHarnesses: CustomHarness[]
  examples: ExampleDescriptor[]
}): string {
  const { selectedCustomId, selectedExampleKey, customHarnesses, examples } = input
  if (selectedCustomId) {
    return customHarnesses.find((c) => c.id === selectedCustomId)?.name || selectedCustomId
  }
  if (selectedExampleKey) {
    const ex = examples.find((e) => e.key === selectedExampleKey)
    return ex?.label || selectedExampleKey
  }
  return 'cli-agent'
}
