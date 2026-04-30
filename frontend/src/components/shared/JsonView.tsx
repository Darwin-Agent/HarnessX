import { useState } from 'react'
import { ChevronDown, ChevronRight } from 'lucide-react'

interface JsonViewProps {
  data: unknown
  label?: string
  defaultOpen?: boolean
}

export function JsonView({ data, label, defaultOpen = false }: JsonViewProps) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <div className="text-xs font-mono">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1 text-gray-500 hover:text-gray-700 mb-1"
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <span className="font-medium">{label ?? 'JSON'}</span>
      </button>
      {open && (
        <pre className="bg-gray-50 border border-gray-200 rounded p-2 overflow-x-auto text-gray-700 leading-relaxed">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  )
}
