import { create } from 'zustand'
import type { SessionMeta } from '../api/types'
import { api } from '../api/client'

interface SessionsState {
  sessions:  SessionMeta[]
  total:     number
  page:      number
  page_size: number
  query:     string
  workspace: 'current' | 'all'
  loading:   boolean
  error:     string | null

  // Agent/project context — kept in sync with slots store
  agentId:  string
  project:  string

  setAgentContext: (agentId: string, project: string) => void
  setQuery:        (q: string) => void
  setWorkspace:    (w: 'current' | 'all') => void
  setPage:         (p: number) => void
  fetch:           () => Promise<void>
  removeSession:   (sessionId: string) => void
}

export const useSessionsStore = create<SessionsState>((set, get) => ({
  sessions:  [],
  total:     0,
  page:      1,
  page_size: 20,
  query:     '',
  workspace: 'current',
  loading:   false,
  error:     null,
  agentId:   'default',
  project:   'default',

  setAgentContext: (agentId, project) => set({ agentId, project }),

  setQuery: (q) => set({ query: q, page: 1 }),

  setWorkspace: (w) => {
    set({ workspace: w, page: 1 })
    get().fetch()
  },

  setPage: (p) => {
    set({ page: p })
    get().fetch()
  },

  fetch: async () => {
    const { query, workspace, page, page_size, agentId, project } = get()
    set({ loading: true, error: null })
    try {
      const res = await api.listSessions({
        workspace,
        agent_id:  agentId,
        project,
        q:         query || undefined,
        page,
        page_size,
      })
      set({ sessions: res.sessions, total: res.total, loading: false })
    } catch (e) {
      set({ error: String(e), loading: false })
    }
  },

  removeSession: (sessionId) =>
    set((s) => ({
      sessions: s.sessions.filter((x) => x.session_id !== sessionId),
      total:    Math.max(0, s.total - 1),
    })),
}))
