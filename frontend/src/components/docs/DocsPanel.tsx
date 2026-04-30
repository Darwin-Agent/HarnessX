import { useEffect, useState, useCallback, useRef } from 'react'
import { X, BookOpen, ChevronRight, Home, Loader, ArrowLeft } from 'lucide-react'
import { marked } from 'marked'
import { useDocsStore } from '../../store/docs'
import { api } from '../../api/client'
import type { DocTree, DocSection, DocEntry } from '../../api/types'

// Configure marked for GFM
marked.setOptions({ gfm: true, breaks: false })

// ── Doc tree nav ──────────────────────────────────────────────────────────────

function DocNav({
  tree,
  currentPath,
  onNavigate,
}: {
  tree: DocTree
  currentPath: string | null
  onNavigate: (path: string) => void
}) {
  return (
    <nav
      className="flex flex-col gap-5 py-4"
      style={{ overflowY: 'auto', overflowX: 'hidden' }}
    >
      {/* Home */}
      <button
        onClick={() => onNavigate('__index__')}
        className="flex items-center gap-2 px-3 py-1.5 rounded-lg mx-2 text-left transition-colors"
        style={{
          fontSize: '13px',
          color: currentPath === '__index__' || currentPath === null
            ? 'var(--accent)'
            : 'var(--text-2)',
          background: currentPath === '__index__' || currentPath === null
            ? 'var(--accent-bg)'
            : 'transparent',
          fontWeight: currentPath === '__index__' || currentPath === null ? 600 : 400,
        }}
      >
        <Home size={13} style={{ flexShrink: 0 }} />
        Overview
      </button>

      {tree.sections.map((section: DocSection) => (
        <div key={section.name}>
          <div
            className="px-3 mb-1"
            style={{
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '10px',
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: 'var(--text-4)',
              fontWeight: 600,
            }}
          >
            {section.name}
          </div>
          {section.items.map((item: DocEntry) => (
            <button
              key={item.path}
              onClick={() => onNavigate(item.path)}
              className="w-full flex items-center gap-2 px-3 py-1.5 rounded-lg mx-2 text-left transition-colors"
              style={{
                width: 'calc(100% - 16px)',
                fontSize: '13px',
                color: currentPath === item.path ? 'var(--accent)' : 'var(--text-2)',
                background: currentPath === item.path ? 'var(--accent-bg)' : 'transparent',
                fontWeight: currentPath === item.path ? 600 : 400,
              }}
              onMouseEnter={(e) => {
                if (currentPath !== item.path) {
                  e.currentTarget.style.background = 'var(--bg-elevated)'
                  e.currentTarget.style.color = 'var(--text-1)'
                }
              }}
              onMouseLeave={(e) => {
                if (currentPath !== item.path) {
                  e.currentTarget.style.background = 'transparent'
                  e.currentTarget.style.color = 'var(--text-2)'
                }
              }}
            >
              {currentPath === item.path
                ? <ChevronRight size={11} style={{ flexShrink: 0, color: 'var(--accent)' }} />
                : <span style={{ width: 11, flexShrink: 0 }} />
              }
              {item.title}
            </button>
          ))}
        </div>
      ))}
    </nav>
  )
}

// ── Doc index (overview) ──────────────────────────────────────────────────────

function DocIndex({
  tree,
  onNavigate,
}: {
  tree: DocTree
  onNavigate: (path: string) => void
}) {
  return (
    <div className="px-8 py-8" style={{ maxWidth: '640px' }}>
      <div className="flex items-center gap-3 mb-6">
        <div
          className="flex items-center justify-center rounded-xl"
          style={{ width: 36, height: 36, background: 'var(--accent-bg)', color: 'var(--accent)' }}
        >
          <BookOpen size={18} />
        </div>
        <div>
          <h1 style={{ fontSize: '20px', fontWeight: 700, color: 'var(--text-1)', lineHeight: 1.2 }}>
            Documentation
          </h1>
          <p style={{ fontSize: '13px', color: 'var(--text-3)', marginTop: '2px' }}>
            HarnessX Harness Lab
          </p>
        </div>
      </div>

      <div className="flex flex-col gap-6">
        {tree.sections.map((section) => (
          <div key={section.name}>
            <div
              style={{
                fontSize: '11px',
                fontFamily: 'JetBrains Mono, monospace',
                letterSpacing: '0.08em',
                textTransform: 'uppercase',
                color: 'var(--text-3)',
                fontWeight: 600,
                marginBottom: '8px',
              }}
            >
              {section.name}
            </div>
            <div className="flex flex-col gap-1">
              {section.items.map((item) => (
                <button
                  key={item.path}
                  onClick={() => onNavigate(item.path)}
                  className="flex items-center justify-between px-3 py-2.5 rounded-xl text-left transition-colors"
                  style={{
                    border: '1px solid var(--border)',
                    background: 'var(--bg-card)',
                    color: 'var(--text-1)',
                    fontSize: '13px',
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = 'var(--accent-ring)'
                    e.currentTarget.style.background = 'var(--accent-bg)'
                    e.currentTarget.style.color = 'var(--accent)'
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = 'var(--border)'
                    e.currentTarget.style.background = 'var(--bg-card)'
                    e.currentTarget.style.color = 'var(--text-1)'
                  }}
                >
                  <span style={{ fontWeight: 500 }}>{item.title}</span>
                  <ChevronRight size={13} style={{ flexShrink: 0, color: 'var(--text-4)' }} />
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Main DocsPanel ────────────────────────────────────────────────────────────

export function DocsPanel() {
  const { isOpen, path, close, navigate } = useDocsStore()
  const [tree, setTree]         = useState<DocTree | null>(null)
  const [content, setContent]   = useState<string>('')
  const [title, setTitle]       = useState<string>('')
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState<string | null>(null)
  const [history, setHistory]   = useState<string[]>([])
  const contentRef = useRef<HTMLDivElement>(null)

  // Fetch tree on first open
  useEffect(() => {
    if (isOpen && !tree) {
      api.docTree()
        .then(setTree)
        .catch((e) => console.error('docTree error:', e))
    }
  }, [isOpen, tree])

  // Fetch content when path changes
  const loadContent = useCallback(async (docPath: string) => {
    if (docPath === '__index__') {
      setContent('')
      setTitle('')
      setError(null)
      return
    }
    setLoading(true)
    setError(null)
    try {
      const doc = await api.docContent(docPath)
      setTitle(doc.title)
      setContent(doc.content)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (isOpen && path) {
      loadContent(path)
    }
  }, [isOpen, path, loadContent])

  // Reset content when panel closes
  useEffect(() => {
    if (!isOpen) {
      setHistory([])
    }
  }, [isOpen])

  function handleNavigate(newPath: string) {
    if (path) setHistory((h) => [...h, path])
    navigate(newPath)
    // Scroll content to top
    if (contentRef.current) contentRef.current.scrollTop = 0
  }

  function handleBack() {
    if (history.length === 0) return
    const prev = history[history.length - 1]
    setHistory((h) => h.slice(0, -1))
    navigate(prev)
  }

  // Handle internal doc links inside rendered markdown
  function handleContentClick(e: React.MouseEvent<HTMLDivElement>) {
    const target = e.target as HTMLElement
    const anchor = target.closest('a')
    if (!anchor) return
    const href = anchor.getAttribute('href')
    if (!href) return
    // Internal doc link: relative paths like "../feats/plugins" or "feats/plugins"
    if (!href.startsWith('http') && !href.startsWith('#')) {
      e.preventDefault()
      // Resolve relative to current path
      let resolved = href
      if (href.startsWith('../') && path) {
        const parts = path.split('/')
        const hrefParts = href.replace(/^\.\.\//, '').split('/')
        resolved = [...parts.slice(0, -1), ...hrefParts].join('/')
        // Strip .md extension if present
        resolved = resolved.replace(/\.md$/, '')
      }
      handleNavigate(resolved)
    }
  }

  if (!isOpen) return null

  const showIndex = !path || path === '__index__'

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-[60]"
        style={{ background: 'rgba(0,0,0,0.25)' }}
        onClick={close}
      />

      {/* Panel */}
      <div
        className="fixed inset-y-0 right-0 z-[61] flex flex-col"
        style={{
          width: '740px',
          background: 'var(--bg-base)',
          borderLeft: '1px solid var(--border)',
          animation: 'slideInRight 0.18s ease',
          boxShadow: '-4px 0 24px rgba(0,0,0,0.12)',
        }}
      >
        {/* Header */}
        <div
          className="flex items-center gap-3 px-4 shrink-0"
          style={{
            height: '48px',
            background: 'var(--bg-card)',
            borderBottom: '1px solid var(--border)',
          }}
        >
          {/* Back button */}
          {history.length > 0 && (
            <button
              onClick={handleBack}
              className="p-1.5 rounded-lg transition-colors"
              style={{ color: 'var(--text-3)' }}
              onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--bg-elevated)'; e.currentTarget.style.color = 'var(--text-1)' }}
              onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-3)' }}
            >
              <ArrowLeft size={15} />
            </button>
          )}

          <BookOpen size={15} style={{ color: 'var(--accent)', flexShrink: 0 }} />
          <span style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-1)', flex: 1 }}>
            {showIndex ? 'Documentation' : (title || 'Documentation')}
          </span>

          <button
            onClick={close}
            className="p-1.5 rounded-lg transition-colors ml-auto"
            style={{ color: 'var(--text-3)' }}
            onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--bg-elevated)'; e.currentTarget.style.color = 'var(--text-1)' }}
            onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-3)' }}
          >
            <X size={16} />
          </button>
        </div>

        {/* Body */}
        <div className="flex flex-1 min-h-0">
          {/* Left nav */}
          <div
            className="shrink-0 overflow-y-auto"
            style={{
              width: '192px',
              borderRight: '1px solid var(--border)',
              background: 'var(--bg-card)',
            }}
          >
            {tree ? (
              <DocNav
                tree={tree}
                currentPath={path}
                onNavigate={handleNavigate}
              />
            ) : (
              <div className="flex items-center justify-center py-8">
                <Loader size={14} className="animate-spin" style={{ color: 'var(--text-4)' }} />
              </div>
            )}
          </div>

          {/* Right content */}
          <div
            ref={contentRef}
            className="flex-1 overflow-y-auto"
            style={{ background: 'var(--bg-base)' }}
            onClick={handleContentClick}
          >
            {loading && (
              <div className="flex items-center justify-center py-12">
                <Loader size={18} className="animate-spin" style={{ color: 'var(--text-4)' }} />
              </div>
            )}

            {error && !loading && (
              <div className="px-8 py-8">
                <p style={{ fontSize: '13px', color: '#ef4444' }}>{error}</p>
              </div>
            )}

            {!loading && !error && showIndex && tree && (
              <DocIndex tree={tree} onNavigate={handleNavigate} />
            )}

            {!loading && !error && !showIndex && content && (
              <div
                className="doc-content px-8 py-8"
                style={{ maxWidth: '680px' }}
                dangerouslySetInnerHTML={{ __html: marked.parse(content) as string }}
              />
            )}
          </div>
        </div>
      </div>
    </>
  )
}
