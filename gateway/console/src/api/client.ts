import { api as labApi } from '@lab/api/client'
import type {
  AgentHarnessConfigResponse as WorkspaceHarnessConfigResponse,
  HarnessConfig,
  RunEvent,
} from '@lab/api/types'
import type {
  ChannelInfo,
  ChannelStatus,
  ChannelConfigResponse,
  GatewayHealth,
  GatewayConfig,
  ChannelTypeInfo,
  GatewaySessionMeta,
} from './types'

const BASE = '/api'
const GW   = '/gateway'

async function gw_get<T>(path: string): Promise<T> {
  const res = await fetch(`${GW}${path}`)
  if (!res.ok) throw new Error(`GET ${GW}${path} → ${res.status}`)
  return res.json() as Promise<T>
}

async function gw_post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${GW}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) { const t = await res.text(); throw new Error(`POST ${GW}${path} → ${res.status}: ${t}`) }
  return res.json() as Promise<T>
}

async function gw_put<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${GW}${path}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) { const t = await res.text(); throw new Error(`PUT ${GW}${path} → ${res.status}: ${t}`) }
  return res.json() as Promise<T>
}

async function gw_delete<T>(path: string): Promise<T> {
  const res = await fetch(`${GW}${path}`, { method: 'DELETE' })
  if (!res.ok) { const t = await res.text(); throw new Error(`DELETE ${GW}${path} → ${res.status}: ${t}`) }
  return res.json() as Promise<T>
}

const gwApi = {
  // ── Channel management ────────────────────────────────────────────────────

  listChannels(): Promise<ChannelInfo[]> {
    return gw_get('/channels')
  },

  getChannelStatus(name: string): Promise<ChannelStatus> {
    return gw_get(`/channels/${encodeURIComponent(name)}/status`)
  },

  getChannelConfig(name: string): Promise<ChannelConfigResponse> {
    return gw_get(`/channels/${encodeURIComponent(name)}/config`)
  },

  updateChannelConfig(name: string, config: Record<string, unknown>): Promise<{ ok: boolean; message: string }> {
    return gw_put(`/channels/${encodeURIComponent(name)}/config`, { config })
  },

  resetChannelSession(name: string, sessionId: string): Promise<{ ok: boolean; base_session_id: string; new_epoch: number }> {
    return gw_post(`/channels/${encodeURIComponent(name)}/reset_session`, { session_id: sessionId })
  },

  generatePairingCode(channel: string): Promise<{ code: string; ttl_seconds: number }> {
    return gw_post('/pairing/generate', { channel })
  },

  gatewayHealth(): Promise<GatewayHealth> {
    return gw_get('/health')
  },

  getGatewayConfig(): Promise<GatewayConfig> {
    return gw_get('/config')
  },

  updateGatewayConfig(gateway: Record<string, unknown>): Promise<{ ok: boolean }> {
    return gw_put('/config', { gateway })
  },

  listChannelTypes(): Promise<ChannelTypeInfo[]> {
    return gw_get('/channel-types')
  },

  createChannel(name: string, config: Record<string, unknown>, channelType?: string): Promise<{ ok: boolean; message: string }> {
    return gw_post('/channels/create', { name, channel_type: channelType ?? name, config })
  },

  restartChannel(name: string): Promise<{ ok: boolean; message: string }> {
    return gw_post(`/channels/${encodeURIComponent(name)}/restart`, {})
  },

  deleteChannel(name: string): Promise<{ ok: boolean }> {
    return gw_delete(`/channels/${encodeURIComponent(name)}`)
  },

  listGatewaySessions(channel?: string): Promise<GatewaySessionMeta[]> {
    const q = channel ? `?channel=${encodeURIComponent(channel)}` : ''
    return gw_get(`/sessions${q}`)
  },

  // ── IM workspace harness config ───────────────────────────────────────────

  getGwHarnessConfig(agentId: string, project: string): Promise<WorkspaceHarnessConfigResponse> {
    const p = new URLSearchParams({ workspace_base: 'im-workspaces', agent_id: agentId, project })
    return fetch(`${BASE}/home/harness-config?${p}`)
      .then(r => { if (!r.ok) throw new Error(`GET /home/harness-config → ${r.status}`); return r.json() })
  },

  saveGwHarnessConfig(agentId: string, project: string, config: HarnessConfig): Promise<WorkspaceHarnessConfigResponse> {
    const p = new URLSearchParams({ workspace_base: 'im-workspaces', agent_id: agentId, project })
    return fetch(`${BASE}/home/harness-config?${p}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    }).then(r => { if (!r.ok) throw new Error(`PUT /home/harness-config → ${r.status}`); return r.json() })
  },

  // ── Console chat (web_ui channel) ─────────────────────────────────────────

  /** Start a console chat run (project=web_ui). Returns { run_id, session_id }. */
  startConsoleRun(message: string, sessionId?: string): Promise<{ run_id: string; session_id: string }> {
    return gw_post('/console/run', { message, session_id: sessionId ?? null })
  },

  // ── Heartbeat ─────────────────────────────────────────────────────────────

  getHeartbeatConfig(): Promise<Record<string, unknown>> {
    return gw_get('/heartbeat/config')
  },

  updateHeartbeatConfig(cfg: Record<string, unknown>): Promise<{ ok: boolean }> {
    return gw_put('/heartbeat/config', cfg)
  },

  getHeartbeatState(): Promise<Record<string, unknown>> {
    return gw_get('/heartbeat')
  },

  // ── Cron jobs ─────────────────────────────────────────────────────────────

  listCronJobs(): Promise<Record<string, unknown>[]> {
    return gw_get('/cron/jobs')
  },

  createCronJob(spec: Record<string, unknown>): Promise<{ ok: boolean; id: string }> {
    return gw_post('/cron/jobs', spec)
  },

  getCronJob(id: string): Promise<Record<string, unknown>> {
    return gw_get(`/cron/jobs/${encodeURIComponent(id)}`)
  },

  updateCronJob(id: string, spec: Record<string, unknown>): Promise<{ ok: boolean; id: string }> {
    return gw_put(`/cron/jobs/${encodeURIComponent(id)}`, spec)
  },

  deleteCronJob(id: string): Promise<{ ok: boolean }> {
    return gw_delete(`/cron/jobs/${encodeURIComponent(id)}`)
  },

  runCronJobNow(id: string): Promise<{ ok: boolean; message: string }> {
    return gw_post(`/cron/jobs/${encodeURIComponent(id)}/run`, {})
  },

  pauseCronJob(id: string): Promise<{ ok: boolean }> {
    return gw_post(`/cron/jobs/${encodeURIComponent(id)}/pause`, {})
  },

  resumeCronJob(id: string): Promise<{ ok: boolean }> {
    return gw_post(`/cron/jobs/${encodeURIComponent(id)}/resume`, {})
  },

  // ── Gateway docs ──────────────────────────────────────────────────────────

  gwDocTree(lang = 'zh'): Promise<unknown> {
    return gw_get(`/docs?lang=${encodeURIComponent(lang)}`)
  },

  gwDocContent(path: string, lang = 'zh'): Promise<unknown> {
    return gw_get(`/docs/${path}?lang=${encodeURIComponent(lang)}`)
  },

  /** Connect to SSE stream; calls `onEvent` per event, returns cleanup fn. */
  streamRun(runId: string, onEvent: (e: RunEvent) => void): () => void {
    const es = new EventSource(`${BASE}/run/${runId}/stream`)
    es.onmessage = (msg) => {
      try {
        const event = JSON.parse(msg.data) as RunEvent
        onEvent(event)
        if (event.type === 'done' || event.type === 'error') es.close()
      } catch { /* ignore malformed frames */ }
    }
    es.onerror = () => { onEvent({ type: 'error', message: 'SSE connection lost' }); es.close() }
    return () => es.close()
  },
}

// Merge Lab UI api + gateway-specific api into one object.
// Console files import `api` from here; no need to import from @lab/api/client.
export const api = { ...labApi, ...gwApi }
