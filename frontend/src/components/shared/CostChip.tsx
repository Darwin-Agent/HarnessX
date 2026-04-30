interface CostChipProps {
  usd: number
}

export function CostChip({ usd }: CostChipProps) {
  const text = usd < 0.01
    ? `${(usd * 100).toFixed(2)}¢`
    : `$${usd.toFixed(3)}`

  const style: React.CSSProperties = usd < 0.01
    ? { background: 'rgba(16,185,129,0.1)', color: '#34d399', border: '1px solid rgba(16,185,129,0.15)' }
    : usd < 0.10
    ? { background: 'rgba(245,158,11,0.1)', color: '#fbbf24', border: '1px solid rgba(245,158,11,0.15)' }
    : { background: 'rgba(239,68,68,0.1)',  color: '#f87171', border: '1px solid rgba(239,68,68,0.15)' }

  return (
    <span
      className="inline-flex items-center font-mono font-medium rounded"
      style={{ ...style, fontSize: '10px', padding: '1px 5px', letterSpacing: '0.02em' }}
    >
      {text}
    </span>
  )
}
