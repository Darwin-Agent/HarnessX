import type {
  SchemaResponse,
  ExampleDescriptor,
  Provider,
  VendorInfo,
  ToolInfo,
  SkillInfo,
  RunRequest,
  RunResponse,
  RunEvent,
  FsListResponse,
  FsFileResponse,
  McpServerConfig,
  McpToolInfo,
  PluginInfo,
  CustomProcessorInfo,
  CustomProcessorScanResponse,
  CustomProcessorTestResponse,
  ValidateResponse,
  DocTree,
  DocContent,
  HomeInfo,
  AgentHarnessConfigResponse,
  HarnessConfig,
  ModelConfigResponse,
  ModelConfigSaveRequest,
  SessionListResponse,
  SessionMessagesResponse,
} from './types'

const BASE = '/api'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(`GET ${path} → ${res.status}`)
  return res.json() as Promise<T>
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`POST ${path} → ${res.status}: ${text}`)
  }
  return res.json() as Promise<T>
}

async function put<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`PUT ${path} → ${res.status}: ${text}`)
  }
  return res.json() as Promise<T>
}

async function patch<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`PATCH ${path} → ${res.status}: ${text}`)
  }
  return res.json() as Promise<T>
}

async function del(path: string): Promise<void> {
  const res = await fetch(`${BASE}${path}`, { method: 'DELETE' })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`DELETE ${path} → ${res.status}: ${text}`)
  }
}

// ── API calls ────────────────────────────────────────────────────────────────

export const api = {
  schema():    Promise<SchemaResponse>      { return get('/schema') },
  examples():  Promise<ExampleDescriptor[]> { return get('/examples') },
  providers(): Promise<Provider[]>          { return get('/providers') },
  vendors():         Promise<VendorInfo[]>        { return get('/vendors') },
  tools():           Promise<ToolInfo[]>          { return get('/tools') },
  skills():          Promise<SkillInfo[]>         { return get('/skills') },
  modelConfig():     Promise<ModelConfigResponse | null> { return get('/model-config') },
  saveModelConfig(req: ModelConfigSaveRequest): Promise<ModelConfigResponse> {
    return put('/model-config', req)
  },

  startRun(req: RunRequest): Promise<RunResponse> {
    return post('/run', req)
  },

  streamRun(runId: string, onEvent: (event: RunEvent) => void): () => void {
    const es = new EventSource(`${BASE}/run/${encodeURIComponent(runId)}/stream`)

    es.onmessage = (msg) => {
      if (!msg.data) return
      try {
        onEvent(JSON.parse(msg.data) as RunEvent)
      } catch {
        // Ignore malformed SSE payloads to keep stream alive.
      }
    }

    es.onerror = () => {
      // Backend closes stream after terminal events (done/error).
      // Keep behavior simple: client may close explicitly via returned stop().
    }

    return () => {
      es.close()
    }
  },

  cancelRun(runId: string): Promise<{ ok: boolean; run_id: string }> {
    return post(`/run/${encodeURIComponent(runId)}/cancel`, {})
  },

  // ── Filesystem ──────────────────────────────────────────────────────────────

  fsList(path: string): Promise<FsListResponse> {
    return get(`/fs?path=${encodeURIComponent(path)}`)
  },

  fsReadFile(path: string): Promise<FsFileResponse> {
    return get(`/fs/file?path=${encodeURIComponent(path)}`)
  },

  fsWriteFile(path: string, content: string): Promise<FsFileResponse> {
    return put('/fs/file', { path, content })
  },

  // ── MCP servers ─────────────────────────────────────────────────────────────

  mcpServers(): Promise<McpServerConfig[]> { return get('/mcp/servers') },

  mcpAddServer(cfg: Omit<McpServerConfig, 'id'>): Promise<McpServerConfig> {
    return post('/mcp/servers', cfg)
  },

  mcpUpdateServer(id: string, p: Partial<McpServerConfig>): Promise<McpServerConfig> {
    return patch(`/mcp/servers/${id}`, p)
  },

  mcpDeleteServer(id: string): Promise<void> { return del(`/mcp/servers/${id}`) },

  mcpServerTools(id: string): Promise<McpToolInfo[]> {
    return post(`/mcp/servers/${id}/tools`, {})
  },

  // ── Plugins ─────────────────────────────────────────────────────────────────

  plugins(): Promise<PluginInfo[]> { return get('/plugins') },

  pluginPatch(id: string, p: { enabled?: boolean }): Promise<PluginInfo> {
    return patch(`/plugins/${id}`, p)
  },

  pluginRemove(id: string): Promise<void> { return del(`/plugins/${id}`) },

  pluginImport(path: string): Promise<PluginInfo> {
    return post('/plugins/import', { path })
  },

  pluginScanDirs(): Promise<{ scan_dirs: string[] }> { return get('/plugins/scan-dirs') },

  pluginAddScanDir(path: string): Promise<{ scan_dirs: string[] }> {
    return post('/plugins/scan-dirs', { path })
  },

  async pluginRemoveScanDir(path: string): Promise<{ scan_dirs: string[] }> {
    const res = await fetch(`${BASE}/plugins/scan-dirs?path=${encodeURIComponent(path)}`, { method: 'DELETE' })
    if (!res.ok) { const t = await res.text(); throw new Error(`DELETE /plugins/scan-dirs → ${res.status}: ${t}`) }
    return res.json()
  },

  // ── Custom processors ───────────────────────────────────────────────────────

  customProcessors(): Promise<CustomProcessorInfo[]> {
    return get('/processors/custom')
  },

  customProcessorScanPath(path: string): Promise<CustomProcessorScanResponse> {
    return post('/processors/scan-path', { path })
  },

  customProcessorScanFile(filename: string, content: string): Promise<CustomProcessorScanResponse> {
    return post('/processors/scan-file', { filename, content })
  },

  customProcessorTest(req: {
    mode: 'path' | 'file'
    class_name: string
    path?: string
    file_path?: string
    filename?: string
    content?: string
  }): Promise<CustomProcessorTestResponse> {
    return post('/processors/test', req)
  },

  customProcessorImport(req: {
    mode: 'path' | 'file'
    class_name: string
    label?: string
    path?: string
    file_path?: string
    filename?: string
    content?: string
  }): Promise<CustomProcessorInfo> {
    return post('/processors/import', req)
  },

  customProcessorRemove(id: string): Promise<void> {
    return del(`/processors/custom/${encodeURIComponent(id)}`)
  },

  // ── AGENT_HOME ──────────────────────────────────────────────────────────────

  getHome(): Promise<HomeInfo> { return get('/home') },

  listAgents(): Promise<{ agents: string[] }> { return get('/home/agents') },

  listProjects(agentId: string): Promise<{ agent_id: string; projects: string[] }> {
    return get(`/home/agents/${encodeURIComponent(agentId)}/projects`)
  },

  getAgentHarnessConfig(
    agentId?: string,
    project?: string,
    persistDefault?: boolean,
  ): Promise<AgentHarnessConfigResponse> {
    const p = new URLSearchParams()
    if (agentId) p.set('agent_id', agentId)
    if (project) p.set('project', project)
    if (persistDefault) p.set('persist_default', 'true')
    return get(`/home/harness-config${p.toString() ? `?${p}` : ''}`)
  },

  saveAgentHarnessConfig(
    harnessConfig: HarnessConfig,
    agentId?: string,
    project?: string,
  ): Promise<AgentHarnessConfigResponse> {
    const p = new URLSearchParams()
    if (agentId) p.set('agent_id', agentId)
    if (project) p.set('project', project)
    return put(`/home/harness-config${p.toString() ? `?${p}` : ''}`, harnessConfig)
  },

  // ── Help / Docs ─────────────────────────────────────────────────────────────

  docTree(): Promise<DocTree> { return get('/help') },

  docContent(path: string): Promise<DocContent> { return get(`/help/${path}`) },

  // ── Session history ─────────────────────────────────────────────────────────

  listSessions(params: {
    workspace?: 'current' | 'all'
    agent_id?:  string
    project?:   string
    q?:         string
    page?:      number
    page_size?: number
  }): Promise<SessionListResponse> {
    const p = new URLSearchParams()
    if (params.workspace) p.set('workspace', params.workspace)
    if (params.agent_id)  p.set('agent_id',  params.agent_id)
    if (params.project)   p.set('project',   params.project)
    if (params.q)         p.set('q',         params.q)
    if (params.page)      p.set('page',      String(params.page))
    if (params.page_size) p.set('page_size', String(params.page_size))
    return get(`/sessions?${p}`)
  },

  getSessionMessages(
    sessionId: string,
    agentId:   string,
    project:   string,
    workspaceBase?: string,
  ): Promise<SessionMessagesResponse> {
    const p = new URLSearchParams({ agent_id: agentId, project })
    if (workspaceBase) p.set('workspace_base', workspaceBase)
    return get(`/sessions/${encodeURIComponent(sessionId)}/messages?${p}`)
  },

  deleteSession(sessionId: string, agentId: string, project: string, workspaceBase?: string): Promise<void> {
    const p = new URLSearchParams({ agent_id: agentId, project })
    if (workspaceBase) p.set('workspace_base', workspaceBase)
    return del(`/sessions/${encodeURIComponent(sessionId)}?${p}`)
  },

  /** Dry-run build: validate harness_config + slot config without starting a run. */
  validate(harnessConfig: unknown, slotConfig: unknown): Promise<ValidateResponse> {
    return post('/validate', { harness_config: harnessConfig, slot_config: slotConfig })
  },
}
