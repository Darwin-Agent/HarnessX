import { useEffect, useRef } from 'react'

interface StreamOutputProps {
  tokens: string
  autoScroll?: boolean
}

export function StreamOutput({ tokens, autoScroll = true }: StreamOutputProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (autoScroll) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
    }
  }, [tokens, autoScroll])

  return (
    <div className="flex-1 overflow-y-auto bg-gray-950 rounded-lg p-3">
      <pre className="text-xs text-gray-200 font-mono whitespace-pre-wrap leading-relaxed">
        {tokens || <span className="text-gray-600 italic">Waiting for output…</span>}
      </pre>
      <div ref={bottomRef} />
    </div>
  )
}
