import { BookOpen } from 'lucide-react'
import { useDocsStore } from '../../store/docs'
import { useT } from '../../i18n'

interface DocLinkProps {
  /** Doc path to open, e.g. "feats/plugins". Opens the index if omitted. */
  path?: string
  /** Optional tooltip text */
  title?: string
  /** Small extra label beside the icon */
  label?: string
}

/**
 * Small contextual button that opens the DocsPanel at a specific page.
 * Place near section headers on Settings pages.
 */
export function DocLink({ path, title, label }: DocLinkProps) {
  const open = useDocsStore((s) => s.open)
  const t = useT()

  return (
    <button
      onClick={() => open(path ?? null)}
      title={title || t('docs.open')}
      className="inline-flex items-center gap-1 rounded-lg transition-colors"
      style={{
        padding: label ? '3px 8px' : '3px 5px',
        fontSize: '11px',
        color: 'var(--text-3)',
        border: '1px solid var(--border)',
        background: 'transparent',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.color = 'var(--accent)'
        e.currentTarget.style.borderColor = 'var(--accent-ring)'
        e.currentTarget.style.background = 'var(--accent-bg)'
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.color = 'var(--text-3)'
        e.currentTarget.style.borderColor = 'var(--border)'
        e.currentTarget.style.background = 'transparent'
      }}
    >
      <BookOpen size={11} />
      {label && <span style={{ fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.02em' }}>{label}</span>}
    </button>
  )
}
