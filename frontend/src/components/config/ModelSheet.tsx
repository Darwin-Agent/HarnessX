import { useEffect } from 'react'
import { ArrowLeft, Sun, Moon, Type } from 'lucide-react'
import { useUIStore } from '../../store/ui'
import { useT } from '../../i18n'
import { ModelPage } from '../settings/ModelPage'

export function ModelSheet() {
  const t = useT()
  const { modelOpen, setModelOpen, theme, lang, toggleTheme, setLang, fontSize, setFontSize } = useUIStore()

  useEffect(() => {
    if (!modelOpen) return
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') setModelOpen(false) }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [modelOpen, setModelOpen])

  if (!modelOpen) return null

  return (
    <div
      className="fixed inset-0 z-50 flex flex-col"
      style={{
        background: 'var(--bg-base)',
        animation: 'slideInUp 0.20s cubic-bezier(0.16, 1, 0.3, 1)',
      }}
    >
      {/* ── Top header ── */}
      <div
        className="flex items-center gap-3 px-5 shrink-0"
        style={{ borderBottom: '1px solid var(--border)', background: 'var(--bg-card)', height: '52px' }}
      >
        <button
          onClick={() => setModelOpen(false)}
          className="flex items-center gap-2 px-3 py-1.5 rounded-xl font-medium transition-all duration-150"
          style={{ fontSize: '14px', color: 'var(--text-2)', border: '1px solid var(--border)', background: 'var(--bg-elevated)' }}
          onMouseEnter={(e) => {
            e.currentTarget.style.color = 'var(--accent)'
            e.currentTarget.style.borderColor = 'var(--accent-ring)'
            e.currentTarget.style.background = 'var(--accent-bg)'
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.color = 'var(--text-2)'
            e.currentTarget.style.borderColor = 'var(--border)'
            e.currentTarget.style.background = 'var(--bg-elevated)'
          }}
        >
          <ArrowLeft size={14} />
          {t('settings.back')}
        </button>

        <span style={{ color: 'var(--border)', fontSize: '16px' }}>/</span>
        <span className="font-semibold" style={{ fontSize: '15px', color: 'var(--text-1)', letterSpacing: '-0.02em' }}>
          {t('topbar.model')}
        </span>

        <div className="flex-1" />

        {/* Font size scaler */}
        <div className="flex items-center gap-2 mr-2">
          <Type size={11} style={{ color: 'var(--text-3)' }} />
          <input
            type="range"
            min={0.80}
            max={1.30}
            step={0.05}
            value={fontSize ?? 1}
            onChange={(e) => setFontSize(parseFloat(e.target.value))}
            className="range-slider"
            style={{ width: '72px' }}
          />
          <span style={{ fontSize: '11px', color: 'var(--accent)', fontFamily: 'JetBrains Mono, monospace', minWidth: '30px' }}>
            {Math.round((fontSize ?? 1) * 100)}%
          </span>
        </div>

        {/* Language toggle */}
        <div className="flex items-center gap-0.5" style={{ borderRadius: '8px', border: '1px solid var(--border)', padding: '2px', background: 'var(--bg-elevated)' }}>
          {(['en', 'zh'] as const).map((l) => (
            <button
              key={l}
              onClick={() => setLang(l)}
              className="px-2 py-1 rounded-md transition-all duration-150 font-medium"
              style={{
                fontSize: '12px',
                color: lang === l ? 'var(--accent)' : 'var(--text-3)',
                background: lang === l ? 'var(--bg-card)' : 'transparent',
                fontFamily: l === 'zh' ? 'system-ui, sans-serif' : 'JetBrains Mono, monospace',
              }}
            >
              {l === 'en' ? 'EN' : '中'}
            </button>
          ))}
        </div>

        {/* Theme toggle */}
        <button
          onClick={toggleTheme}
          className="p-2 rounded-xl transition-all duration-150"
          style={{ color: 'var(--text-3)' }}
          onMouseEnter={(e) => { e.currentTarget.style.background = 'var(--bg-elevated)'; e.currentTarget.style.color = 'var(--text-1)' }}
          onMouseLeave={(e) => { e.currentTarget.style.background = 'transparent'; e.currentTarget.style.color = 'var(--text-3)' }}
        >
          {theme === 'dark' ? <Sun size={15} /> : <Moon size={15} />}
        </button>
      </div>

      {/* ── Content ── */}
      <div className="flex-1 min-h-0 overflow-y-auto">
        <div className="px-8 py-7 max-w-4xl">
          <div className="mb-7">
            <h1 className="font-bold" style={{ fontSize: '22px', color: 'var(--text-1)', letterSpacing: '-0.03em' }}>
              {lang === 'zh' ? '模型' : 'Model'}
            </h1>
            <p className="mt-1.5" style={{ fontSize: '14px', color: 'var(--text-3)', lineHeight: 1.6 }}>
              {lang === 'zh'
                ? '配置模型提供商、API 密钥和端点，并将模型分配到命名插槽。支持导入/导出 YAML。'
                : 'Configure model providers, API keys, and endpoints. Assign models to named slots. Supports YAML import/export.'}
            </p>
          </div>
          <ModelPage />
        </div>
      </div>
    </div>
  )
}
