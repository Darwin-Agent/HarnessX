interface BadgeProps {
  children: React.ReactNode
  variant?: 'green' | 'red' | 'yellow' | 'blue' | 'gray'
  size?: 'sm' | 'xs'
}

const variantStyles: Record<string, React.CSSProperties> = {
  green:  { background: 'rgba(16,185,129,0.10)',  color: '#10b981', border: '1px solid rgba(16,185,129,0.25)' },
  red:    { background: 'rgba(239,68,68,0.10)',   color: '#ef4444', border: '1px solid rgba(239,68,68,0.25)' },
  yellow: { background: 'rgba(245,158,11,0.10)',  color: '#d97706', border: '1px solid rgba(245,158,11,0.25)' },
  blue:   { background: 'var(--accent-bg)',        color: 'var(--accent)', border: '1px solid var(--accent-ring)' },
  gray:   { background: 'var(--bg-elevated)',      color: 'var(--text-2)', border: '1px solid var(--border)' },
}

export function Badge({ children, variant = 'gray', size = 'sm' }: BadgeProps) {
  const sz = size === 'xs'
    ? { fontSize: '10px', padding: '1px 6px' }
    : { fontSize: '11px', padding: '2px 7px' }

  return (
    <span
      className="inline-flex items-center rounded-full font-mono font-medium"
      style={{ ...variantStyles[variant], ...sz, letterSpacing: '0.02em' }}
    >
      {children}
    </span>
  )
}
