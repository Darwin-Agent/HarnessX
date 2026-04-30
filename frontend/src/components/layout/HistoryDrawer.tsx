import { useEffect, useRef, useState } from 'react'
import { X, Trash2, Clock, Search, ChevronLeft, ChevronRight } from 'lucide-react'
import { useSessionsStore } from '../../store/sessions'
import { useRunsStore } from '../../store/runs'
import { useLabStore } from '../../store/lab'
import { api } from '../../api/client'
import { resolveBuilderHarnessName, workspaceFromHarnessName } from '../../lib/labWorkspace'
import type { SessionMeta } from '../../api/types'

interface Props {
  isOpen: boolean
  onClose: () => void
}

function formatDate(iso: string): string {
  if (!iso) return '—'
  try {
    const d = new Date(iso)
    const date = d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
    const time = d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
    return `${date} · ${time}`
  } catch {
    return iso.slice(0, 10)
  }
}

// ── Session card ──────────────────────────────────────────────────────────────

function SessionCard({
  session,
  isLoading,
  onSelect,
  onDelete,
}: {
  session:   SessionMeta
  isLoading: boolean
  onSelect:  (s: SessionMeta) => void
  onDelete:  (s: SessionMeta) => void
}) {
  const [hovered, setHovered] = useState(false)

  return (
    <div
      onClick={() => !isLoading && onSelect(session)}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      style={{
        padding:      '9px 16px',
        cursor:       isLoading ? 'wait' : 'pointer',
        background:   hovered ? 'var(--bg-elevated)' : 'transparent',
        borderBottom: '1px solid var(--border)',
        transition:   'background 0.1s',
        opacity:      isLoading ? 0.6 : 1,
        position:     'relative',
      }}
    >
      {/* Top row: id / date / run count */}
      <div className="flex items-center gap-2 mb-0.5">
        <span
          style={{
            fontSize:   11,
            fontFamily: 'JetBrains Mono, monospace',
            color:      'var(--text-3)',
            flexShrink: 0,
          }}
        >
          {session.session_id.slice(0, 8)}
        </span>
        <span
          style={{
            fontSize: 11,
            color:    'var(--text-4)',
            flex:     1,
            textAlign: 'right',
            minWidth: 0,
            overflow: 'hidden',
            textOverflow: 'ellipsis',
            whiteSpace: 'nowrap',
          }}
        >
          {formatDate(session.updated_at)}
        </span>
        <span
          style={{
            fontSize:     10,
            color:        'var(--text-4)',
            background:   'var(--bg-elevated)',
            borderRadius: 4,
            padding:      '1px 5px',
            flexShrink:   0,
            fontFamily:   'JetBrains Mono, monospace',
          }}
        >
          ×{session.run_count}
        </span>
      </div>

      {/* First query */}
      <p
        style={{
          fontSize:     12,
          color:        'var(--text-2)',
          overflow:     'hidden',
          textOverflow: 'ellipsis',
          whiteSpace:   'nowrap',
          margin:       0,
          paddingRight: hovered ? 28 : 0,
        }}
        title={session.first_query}
      >
        {session.first_query || '(empty)'}
      </p>

      {/* Search match snippet */}
      {session.match_snippet && (
        <p
          style={{
            fontSize:     11,
            color:        'var(--text-3)',
            marginTop:    3,
            fontStyle:    'italic',
            overflow:     'hidden',
            textOverflow: 'ellipsis',
            whiteSpace:   'nowrap',
            margin:       '3px 0 0',
          }}
        >
          {session.match_snippet}
        </p>
      )}

      {/* Workspace badge for "all" mode */}
      {session.agent_id !== 'default' || session.project !== 'default' ? (
        <span
          style={{
            fontSize:     10,
            color:        'var(--text-4)',
            fontFamily:   'JetBrains Mono, monospace',
            marginTop:    2,
            display:      'block',
          }}
        >
          {session.agent_id}/{session.project}
        </span>
      ) : null}

      {/* Delete button — visible on hover */}
      {hovered && (
        <button
          onClick={(e) => { e.stopPropagation(); onDelete(session) }}
          style={{
            position:     'absolute',
            right:        10,
            top:          '50%',
            transform:    'translateY(-50%)',
            padding:      '4px 6px',
            borderRadius: 6,
            background:   'rgba(239,68,68,0.08)',
            color:        '#ef4444',
            border:       '1px solid rgba(239,68,68,0.2)',
          }}
          title="Delete"
        >
          <Trash2 size={11} />
        </button>
      )}
    </div>
  )
}

// ── Main drawer ───────────────────────────────────────────────────────────────

export function HistoryDrawer({ isOpen, onClose }: Props) {
  const {
    selectedCustomId, selectedExampleKey, customHarnesses, examples,
    startChat, savedWorkspaceConfig,
  } = useLabStore()
  const {
    sessions, total, page, page_size, query, workspace,
    loading,
    setAgentContext, setQuery, setWorkspace, setPage, fetch,
    removeSession,
  } = useSessionsStore()
  const { columns, loadHistorySession } = useRunsStore()

  const currentHarnessName = resolveBuilderHarnessName({
    selectedCustomId,
    selectedExampleKey,
    customHarnesses,
    examples,
  })
  const derivedWs = workspaceFromHarnessName(currentHarnessName)
  const preferStoreWorkspace = selectedCustomId !== null || selectedExampleKey !== null || columns.length <= 1
  const activeAgentId = preferStoreWorkspace
    ? savedWorkspaceConfig.agent_id
    : (columns[0]?.workspaceAgentId ?? derivedWs.agentId)
  const activeProject = preferStoreWorkspace
    ? savedWorkspaceConfig.project
    : (columns[0]?.workspaceProject ?? derivedWs.project)

  const [confirmDelete, setConfirmDelete] = useState<SessionMeta | null>(null)
  const [deleting, setDeleting]           = useState(false)
  const [loadingId, setLoadingId]         = useState<string | null>(null)
  const searchRef                         = useRef<HTMLInputElement>(null)
  const debounceRef                       = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Sync context and fetch on open
  useEffect(() => {
    if (!isOpen) return
    setAgentContext(activeAgentId, activeProject)
    fetch()
    setTimeout(() => searchRef.current?.focus(), 200)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, activeAgentId, activeProject])

  function handleQueryChange(q: string) {
    setQuery(q)
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => useSessionsStore.getState().fetch(), 400)
  }

  async function handleSelectSession(s: SessionMeta) {
    if (loadingId) return
    setLoadingId(s.session_id)
    try {
      const res = await api.getSessionMessages(s.session_id, s.agent_id, s.project)
      const colId = columns[0]?.id
      if (!colId) return
      loadHistorySession(colId, s.session_id, res.messages)
      // Switch builder to chat view
      startChat()
      onClose()
    } catch (e) {
      console.error('Failed to load session messages', e)
    } finally {
      setLoadingId(null)
    }
  }

  async function handleDeleteConfirm() {
    if (!confirmDelete) return
    setDeleting(true)
    try {
      await api.deleteSession(
        confirmDelete.session_id,
        confirmDelete.agent_id,
        confirmDelete.project,
      )
      removeSession(confirmDelete.session_id)
      setConfirmDelete(null)
    } catch (e) {
      console.error('Failed to delete session', e)
    } finally {
      setDeleting(false)
    }
  }

  const totalPages = Math.max(1, Math.ceil(total / page_size))

  return (
    <>
      {/* Backdrop */}
      {isOpen && (
        <div
          onClick={onClose}
          style={{
            position:   'fixed',
            inset:      0,
            zIndex:     40,
            background: 'rgba(0,0,0,0.3)',
          }}
        />
      )}

      {/* Drawer panel */}
      <div
        style={{
          position:      'fixed',
          top:           0,
          left:          0,
          bottom:        0,
          width:         '360px',
          zIndex:        50,
          background:    'var(--bg-card)',
          borderRight:   '1px solid var(--border)',
          display:       'flex',
          flexDirection: 'column',
          transform:     isOpen ? 'translateX(0)' : 'translateX(-100%)',
          transition:    'transform 0.22s cubic-bezier(0.4,0,0.2,1)',
          boxShadow:     isOpen ? '4px 0 24px rgba(0,0,0,0.18)' : 'none',
        }}
      >
        {/* Header */}
        <div
          style={{
            padding:      '12px 16px 10px',
            borderBottom: '1px solid var(--border)',
            flexShrink:   0,
          }}
        >
          <div className="flex items-center justify-between mb-3">
            <span
              className="font-semibold flex items-center gap-2"
              style={{ fontSize: 14, color: 'var(--text-1)' }}
            >
              <Clock size={14} style={{ color: 'var(--accent)' }} />
              History
            </span>
            <button
              onClick={onClose}
              className="p-1 rounded-lg transition-colors"
              style={{ color: 'var(--text-3)' }}
              onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--text-1)')}
              onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-3)')}
            >
              <X size={15} />
            </button>
          </div>

          {/* Workspace toggle */}
          <div
            className="flex rounded-lg overflow-hidden mb-3"
            style={{ border: '1px solid var(--border)' }}
          >
            {(['current', 'all'] as const).map((w) => (
              <button
                key={w}
                onClick={() => setWorkspace(w)}
                style={{
                  flex:       1,
                  padding:    '5px 0',
                  fontSize:   11,
                  fontWeight: workspace === w ? 600 : 400,
                  background: workspace === w ? 'var(--accent-bg)' : 'transparent',
                  color:      workspace === w ? 'var(--accent)' : 'var(--text-3)',
                  transition: 'all 0.15s',
                }}
              >
                {w === 'current' ? 'Current workspace' : 'All workspaces'}
              </button>
            ))}
          </div>

          {/* Search */}
          <div
            className="flex items-center gap-2 rounded-lg px-2.5"
            style={{ border: '1px solid var(--border)', background: 'var(--bg-elevated)' }}
          >
            <Search size={13} style={{ color: 'var(--text-3)', flexShrink: 0 }} />
            <input
              ref={searchRef}
              type="text"
              value={query}
              onChange={(e) => handleQueryChange(e.target.value)}
              placeholder="Search all user messages…"
              style={{
                flex:       1,
                padding:    '7px 0',
                background: 'transparent',
                border:     'none',
                outline:    'none',
                fontSize:   12,
                color:      'var(--text-1)',
              }}
            />
            {query && (
              <button
                onClick={() => handleQueryChange('')}
                style={{ color: 'var(--text-3)', lineHeight: 1, flexShrink: 0 }}
              >
                <X size={12} />
              </button>
            )}
          </div>
        </div>

        {/* Session list */}
        <div className="flex-1 overflow-y-auto">
          {loading && sessions.length === 0 && (
            <div className="text-center py-10" style={{ fontSize: 12, color: 'var(--text-3)' }}>
              Loading…
            </div>
          )}

          {!loading && sessions.length === 0 && (
            <div className="text-center py-10 px-6" style={{ fontSize: 12, color: 'var(--text-3)' }}>
              {query ? 'No sessions match your search.' : 'No conversations yet.'}
            </div>
          )}

          {sessions.map((s) => (
            <SessionCard
              key={s.session_id}
              session={s}
              isLoading={loadingId === s.session_id}
              onSelect={handleSelectSession}
              onDelete={(sess) => setConfirmDelete(sess)}
            />
          ))}
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div
            className="flex items-center justify-between px-4 py-2.5 shrink-0"
            style={{ borderTop: '1px solid var(--border)', fontSize: 11, color: 'var(--text-3)' }}
          >
            <button
              onClick={() => setPage(page - 1)}
              disabled={page <= 1}
              className="p-1 rounded disabled:opacity-30"
              style={{ color: 'var(--text-2)' }}
            >
              <ChevronLeft size={14} />
            </button>
            <span>{page} / {totalPages} · {total} total</span>
            <button
              onClick={() => setPage(page + 1)}
              disabled={page >= totalPages}
              className="p-1 rounded disabled:opacity-30"
              style={{ color: 'var(--text-2)' }}
            >
              <ChevronRight size={14} />
            </button>
          </div>
        )}
      </div>

      {/* Delete confirmation modal */}
      {confirmDelete && (
        <>
          <div
            onClick={() => !deleting && setConfirmDelete(null)}
            style={{
              position:   'fixed',
              inset:      0,
              zIndex:     60,
              background: 'rgba(0,0,0,0.45)',
            }}
          />
          <div
            style={{
              position:     'fixed',
              left:         '50%',
              top:          '50%',
              transform:    'translate(-50%,-50%)',
              zIndex:       70,
              background:   'var(--bg-card)',
              border:       '1px solid var(--border)',
              borderRadius: 16,
              padding:      24,
              width:        360,
              boxShadow:    '0 8px 32px rgba(0,0,0,0.28)',
            }}
          >
            <p className="font-semibold mb-2" style={{ fontSize: 14, color: 'var(--text-1)' }}>
              Delete this conversation?
            </p>
            <p
              style={{
                fontSize:   11,
                color:      'var(--text-3)',
                fontFamily: 'JetBrains Mono, monospace',
                marginBottom: 4,
              }}
            >
              {confirmDelete.session_id}
            </p>
            <p style={{ fontSize: 12, color: 'var(--text-2)', margin: '4px 0 0' }}>
              {confirmDelete.first_query
                ? `"${confirmDelete.first_query.slice(0, 80)}${confirmDelete.first_query.length > 80 ? '…' : ''}"`
                : '(empty)'}
            </p>
            <p style={{ fontSize: 11, color: '#ef4444', margin: '12px 0 16px' }}>
              This permanently deletes all conversation data and cannot be undone.
            </p>
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setConfirmDelete(null)}
                disabled={deleting}
                className="px-3 py-1.5 rounded-lg disabled:opacity-50"
                style={{ fontSize: 12, border: '1px solid var(--border)', color: 'var(--text-2)' }}
              >
                Cancel
              </button>
              <button
                onClick={handleDeleteConfirm}
                disabled={deleting}
                className="px-3 py-1.5 rounded-lg font-semibold disabled:opacity-60"
                style={{ fontSize: 12, background: '#ef4444', color: '#fff' }}
              >
                {deleting ? 'Deleting…' : 'Delete'}
              </button>
            </div>
          </div>
        </>
      )}
    </>
  )
}
