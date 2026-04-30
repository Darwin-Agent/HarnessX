// Re-export all Lab UI types so console files only need one import source
export * from '@lab/api/types'

// ── Gateway / IM Channels ─────────────────────────────────────────────────────

export interface ChannelInfo {
  name: string
  display_name: string
  enabled: boolean
  connection_state: 'online' | 'offline' | 'connecting' | 'error' | string
}

export interface ChannelStatus {
  name: string
  connection_state: string
  queue_size: number
  active_sessions: number
}

export interface ChannelConfigResponse {
  name: string
  config: Record<string, unknown>
  schema: {
    type: string
    required?: string[]
    properties?: Record<string, {
      type?: string
      title?: string
      format?: string
      default?: unknown
      items?: { type: string }
    }>
  }
}

export interface GatewayHealth {
  ok: boolean
  channels: Record<string, string>
}

export interface GatewayConfig {
  gateway: Record<string, unknown>
}

export interface ChannelTypeInfo {
  name: string
  display_name: string
  schema: {
    type?: string
    required?: string[]
    properties?: Record<string, {
      type?: string
      title?: string
      format?: string
      default?: unknown
      items?: { type: string }
    }>
  }
  available: boolean
  missing_dep?: string
}

export interface GatewaySessionMeta {
  session_id: string
  channel: string
  agent_id: string
  project: string
  first_query: string
  updated_at: string
  run_count: number
}
