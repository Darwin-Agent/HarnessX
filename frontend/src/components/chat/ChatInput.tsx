import { useState, useRef, useEffect } from 'react'
import { Send, RotateCcw, ChevronDown, Paperclip, X } from 'lucide-react'
import { useLabStore } from '../../store/lab'
import { useT } from '../../i18n'
import type { Attachment } from '../../api/types'

interface Props {
  onSend:       (text: string, attachments: Attachment[]) => void
  onNewChat?:   () => void
  disabled?:    boolean
  placeholder?: string
}

export function ChatInput({ onSend, onNewChat, disabled, placeholder }: Props) {
  const [text, setText] = useState('')
  const [attachments, setAttachments] = useState<Attachment[]>([])
  const [advOpen, setAdvOpen] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const { successCriteria, setSuccessCriteria } = useLabStore()
  const t = useT()

  useEffect(() => {
    const el = textareaRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`
  }, [text])

  async function addImageFile(file: File) {
    if (!file.type.startsWith('image/')) return
    if (file.size > 5 * 1024 * 1024) return  // silent drop >5 MB
    const dataUrl = await new Promise<string>((resolve) => {
      const r = new FileReader()
      r.onload = () => resolve(r.result as string)
      r.readAsDataURL(file)
    })
    const [header, data] = dataUrl.split(',')
    const media_type = header.match(/data:([^;]+)/)?.[1] ?? 'image/png'
    setAttachments((prev) => [...prev, { type: 'image', media_type, data, name: file.name }])
  }

  function removeAttachment(idx: number) {
    setAttachments((prev) => prev.filter((_, i) => i !== idx))
  }

  function send() {
    const trimmed = text.trim()
    if ((!trimmed && attachments.length === 0) || disabled) return
    onSend(trimmed, attachments)
    setText('')
    setAttachments([])
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  async function handlePaste(e: React.ClipboardEvent<HTMLTextAreaElement>) {
    const items = Array.from(e.clipboardData?.items ?? [])
    const imageItem = items.find((i) => i.type.startsWith('image/'))
    if (!imageItem) return
    e.preventDefault()
    const file = imageItem.getAsFile()
    if (file) await addImageFile(file)
  }

  async function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setDragOver(false)
    const files = Array.from(e.dataTransfer.files).filter((f) => f.type.startsWith('image/'))
    for (const f of files) await addImageFile(f)
  }

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files ?? [])
    for (const f of files) await addImageFile(f)
    e.target.value = ''  // reset so same file can be picked again
  }

  const canSend = (!!text.trim() || attachments.length > 0) && !disabled

  return (
    <div
      className="px-4 pt-3 pb-4 shrink-0"
      style={{ borderTop: '1px solid var(--border)', background: 'var(--bg-card)' }}
    >
      <div
        className="rounded-2xl overflow-hidden transition-all duration-150"
        style={{
          border: `1px solid ${dragOver ? 'var(--accent)' : 'var(--border)'}`,
          background: 'var(--bg-base)',
          boxShadow: dragOver ? '0 0 0 3px var(--accent-ring)' : '0 1px 6px rgba(0,0,0,0.04)',
        }}
        onFocusCapture={(e) => {
          if (e.target.tagName === 'TEXTAREA') {
            (e.currentTarget as HTMLElement).style.borderColor = 'var(--accent)'
            ;(e.currentTarget as HTMLElement).style.boxShadow = '0 0 0 3px var(--accent-ring)'
          }
        }}
        onBlurCapture={(e) => {
          if (e.target.tagName === 'TEXTAREA') {
            ;(e.currentTarget as HTMLElement).style.borderColor = 'var(--border)'
            ;(e.currentTarget as HTMLElement).style.boxShadow = '0 1px 6px rgba(0,0,0,0.04)'
          }
        }}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        {/* Attachment preview strip */}
        {attachments.length > 0 && (
          <div className="flex flex-wrap gap-2 px-4 pt-3">
            {attachments.map((att, i) => (
              <div key={i} className="relative shrink-0" style={{ width: 48, height: 48 }}>
                <img
                  src={`data:${att.media_type};base64,${att.data}`}
                  alt={att.name ?? 'image'}
                  className="rounded-lg object-cover w-full h-full"
                  style={{ border: '1px solid var(--border)' }}
                />
                <button
                  onClick={() => removeAttachment(i)}
                  className="absolute -top-1.5 -right-1.5 rounded-full flex items-center justify-center"
                  style={{
                    width: 16, height: 16,
                    background: 'var(--bg-elevated)',
                    border: '1px solid var(--border)',
                    color: 'var(--text-2)',
                    cursor: 'pointer',
                    padding: 0,
                  }}
                  title="Remove"
                >
                  <X size={10} />
                </button>
              </div>
            ))}
          </div>
        )}

        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          onPaste={handlePaste}
          disabled={disabled}
          placeholder={placeholder ?? t('chat.placeholder')}
          rows={2}
          className="w-full resize-none disabled:opacity-50"
          style={{
            background: 'transparent',
            border: 'none',
            outline: 'none',
            color: 'var(--text-1)',
            fontSize: '15px',
            lineHeight: '1.6',
            padding: attachments.length > 0 ? '8px 16px 8px' : '14px 16px 8px',
            fontFamily: 'inherit',
            boxShadow: 'none',
          }}
        />

        {/* Action bar */}
        <div className="flex items-center gap-2 px-3 pb-2.5">
          {/* Attach image */}
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={disabled}
            className="p-1.5 rounded-lg transition-colors disabled:opacity-30"
            style={{ color: 'var(--text-4)' }}
            title="Attach image"
            onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--text-2)')}
            onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-4)')}
          >
            <Paperclip size={15} />
          </button>

          {/* Advanced toggle */}
          <button
            onClick={() => setAdvOpen((o) => !o)}
            className="flex items-center gap-1 transition-colors rounded-lg px-2 py-1"
            style={{ color: advOpen ? 'var(--accent)' : 'var(--text-4)', fontSize: '12px' }}
            onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--text-2)')}
            onMouseLeave={(e) => (e.currentTarget.style.color = advOpen ? 'var(--accent)' : 'var(--text-4)')}
          >
            <ChevronDown
              size={12}
              style={{ transform: advOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.15s' }}
            />
            {t('chat.advanced')}
          </button>

          <div className="flex-1" />

          {/* New chat */}
          {onNewChat && (
            <button
              onClick={onNewChat}
              disabled={disabled}
              className="p-2 rounded-xl shrink-0 disabled:opacity-30 transition-colors"
              style={{ color: 'var(--text-3)' }}
              title={t('status.new_chat')}
              onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--text-1)')}
              onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--text-3)')}
            >
              <RotateCcw size={16} />
            </button>
          )}

          {/* Send */}
          <button
            onClick={send}
            disabled={!canSend}
            className="p-2.5 rounded-xl shrink-0 transition-all duration-150 disabled:opacity-25 disabled:cursor-not-allowed"
            style={{
              background: canSend ? 'var(--accent)' : 'var(--bg-elevated)',
              color: canSend ? '#fff' : 'var(--text-3)',
              boxShadow: canSend ? 'var(--accent-glow)' : 'none',
            }}
            title="Send"
          >
            <Send size={16} />
          </button>
        </div>
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        className="hidden"
        onChange={handleFileChange}
      />

      {/* Advanced panel */}
      {advOpen && (
        <div className="mt-3 grid grid-cols-2 gap-3">
          <div>
            <label className="block mb-1.5" style={{ fontSize: '12px', color: 'var(--text-3)', fontFamily: 'JetBrains Mono, monospace', letterSpacing: '0.04em', textTransform: 'uppercase' }}>
              {t('chat.success_criteria')}
            </label>
            <input
              type="text"
              value={successCriteria}
              onChange={(e) => setSuccessCriteria(e.target.value)}
              placeholder={t('chat.success_placeholder')}
              className="w-full rounded-lg px-3 py-2"
              style={{ fontSize: '13px' }}
            />
          </div>
        </div>
      )}
    </div>
  )
}
