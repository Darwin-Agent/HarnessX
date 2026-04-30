import { useEffect, useState, useCallback, useRef } from 'react'
import { BookOpen, ChevronRight, Home, Loader, ArrowLeft } from 'lucide-react'
import { marked } from 'marked'
import { api } from '@gw/api/client'
import { useUIStore } from '@lab/store/ui'

marked.setOptions({ gfm: true, breaks: false })

interface DocEntry  { path: string; title: string }
interface DocSection { name: string; items: DocEntry[] }
interface DocTree    { sections: DocSection[] }
interface DocContent { path: string; title: string; content: string }

// ── Left sidebar nav ──────────────────────────────────────────────────────────

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
    <nav className="flex flex-col gap-5 py-4" style={{ overflowY: 'auto', overflowX: 'hidden' }}>
      <button
        onClick={() => onNavigate('__index__')}
        className="flex items-center gap-2 px-3 py-1.5 rounded-lg mx-2 text-left transition-colors"
        style={{
          fontSize: '13px',
          color: !currentPath || currentPath === '__index__' ? 'var(--accent)' : 'var(--text-2)',
          background: !currentPath || currentPath === '__index__' ? 'var(--accent-bg)' : 'transparent',
          fontWeight: !currentPath || currentPath === '__index__' ? 600 : 400,
        }}
      >
        <Home size={13} style={{ flexShrink: 0 }} />
        Overview
      </button>

      {tree.sections.map((section) => (
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
          {section.items.map((item) => (
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

// ── Overview index ────────────────────────────────────────────────────────────

function DocIndex({ tree, onNavigate }: { tree: DocTree; onNavigate: (path: string) => void }) {
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
            Gateway Docs
          </h1>
          <p style={{ fontSize: '13px', color: 'var(--text-3)', marginTop: '2px' }}>
            HarnessX IM Gateway
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

// ── DocsPage ──────────────────────────────────────────────────────────────────

export function DocsPage() {
  const lang = useUIStore((s) => s.lang)

  const [tree, setTree]       = useState<DocTree | null>(null)
  const [path, setPath]       = useState<string | null>(null)
  const [content, setContent] = useState<string>('')
  const [title, setTitle]     = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState<string | null>(null)
  const [history, setHistory] = useState<string[]>([])
  const contentRef = useRef<HTMLDivElement>(null)

  // Reload tree when lang changes; reset nav state
  useEffect(() => {
    setTree(null)
    setPath(null)
    setContent('')
    setTitle('')
    setHistory([])
    api.gwDocTree(lang)
      .then((t) => setTree(t as DocTree))
      .catch((e) => console.error('gwDocTree error:', e))
  }, [lang])

  const loadContent = useCallback(async (docPath: string, docLang: string) => {
    if (docPath === '__index__') {
      setContent('')
      setTitle('')
      setError(null)
      return
    }
    setLoading(true)
    setError(null)
    try {
      const doc = await api.gwDocContent(docPath, docLang) as DocContent
      setTitle(doc.title)
      setContent(doc.content)
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }, [])

  function handleNavigate(newPath: string) {
    if (path) setHistory((h) => [...h, path])
    setPath(newPath)
    loadContent(newPath, lang)
    if (contentRef.current) contentRef.current.scrollTop = 0
  }

  function handleBack() {
    if (history.length === 0) return
    const prev = history[history.length - 1]
    setHistory((h) => h.slice(0, -1))
    setPath(prev)
    loadContent(prev, lang)
  }

  function handleContentClick(e: React.MouseEvent<HTMLDivElement>) {
    const anchor = (e.target as HTMLElement).closest('a')
    if (!anchor) return
    const href = anchor.getAttribute('href')
    if (!href || href.startsWith('http') || href.startsWith('#')) return
    e.preventDefault()
    let resolved = href.replace(/\.md$/, '')
    if (href.startsWith('../') && path) {
      const parts = path.split('/')
      resolved = [...parts.slice(0, -1), ...href.replace(/^\.\.\//, '').split('/')].join('/').replace(/\.md$/, '')
    }
    handleNavigate(resolved)
  }

  const showIndex = !path || path === '__index__'

  return (
    <div className="flex flex-1 min-h-0 overflow-hidden">
      {/* Left sidebar */}
      <div
        className="shrink-0 overflow-y-auto"
        style={{
          width: '200px',
          borderRight: '1px solid var(--border)',
          background: 'var(--bg-card)',
        }}
      >
        {tree ? (
          <DocNav tree={tree} currentPath={path} onNavigate={handleNavigate} />
        ) : (
          <div className="flex items-center justify-center py-8">
            <Loader size={14} className="animate-spin" style={{ color: 'var(--text-4)' }} />
          </div>
        )}
      </div>

      {/* Content area */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* Content header (breadcrumb / back) */}
        <div
          className="flex items-center gap-3 px-5 shrink-0"
          style={{
            height: '40px',
            borderBottom: '1px solid var(--border)',
            background: 'var(--bg-card)',
          }}
        >
          {history.length > 0 && (
            <button
              onClick={handleBack}
              className="p-1 rounded-lg transition-colors flex items-center gap-1.5"
              style={{ color: 'var(--text-3)', fontSize: '12px' }}
              onMouseEnter={(e) => { e.currentTarget.style.color = 'var(--text-1)' }}
              onMouseLeave={(e) => { e.currentTarget.style.color = 'var(--text-3)' }}
            >
              <ArrowLeft size={13} />
              Back
            </button>
          )}
          <span style={{ fontSize: '13px', color: showIndex ? 'var(--text-3)' : 'var(--text-1)', fontWeight: showIndex ? 400 : 500 }}>
            {showIndex ? 'Gateway Documentation' : title}
          </span>
        </div>

        {/* Scrollable content */}
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
            <div style={{ maxWidth: '800px', margin: '0 auto', width: '100%' }}>
              <DocIndex tree={tree} onNavigate={handleNavigate} />
            </div>
          )}

          {!loading && !error && !showIndex && content && (
            <div className="px-8 py-8" style={{ maxWidth: '800px', margin: '0 auto', width: '100%' }}>
              <div
                className="doc-content"
                dangerouslySetInnerHTML={{ __html: marked.parse(content) as string }}
              />
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
