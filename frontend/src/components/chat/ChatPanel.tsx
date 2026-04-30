import { useEffect, useRef } from 'react'
import type { ChatMessage } from '../../api/types'
import { ChatBubble } from './ChatBubble'
import { useT } from '../../i18n'

interface Props {
  messages:   ChatMessage[]
  autoScroll: boolean
}

export function ChatPanel({ messages, autoScroll }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const t = useT()

  useEffect(() => {
    if (autoScroll) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, autoScroll])

  if (messages.length === 0) {
    return (
      <div
        className="flex-1 flex items-center justify-center text-xs select-none font-mono"
        style={{ color: 'var(--text-4)' }}
      >
        {t('chat.empty')}
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-3 min-h-0">
      {messages.map((msg, i) => (
        <ChatBubble key={i} message={msg} />
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
