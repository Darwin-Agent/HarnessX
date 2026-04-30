import { useSlotsStore } from '../../store/slots'
import { useT } from '../../i18n'

export function ToolsSection() {
  const t = useT()
  const { toolInfos, enabledTools, setEnabledTools } = useSlotsStore()

  const allNames = toolInfos.map((t) => t.name)
  const active = enabledTools ?? allNames

  function toggle(name: string) {
    const next = active.includes(name)
      ? active.filter((n) => n !== name)
      : [...active, name]
    setEnabledTools(next.length === allNames.length ? null : next)
  }

  const groups = ['filesystem', 'web'] as const

  return (
    <div className="space-y-2">
      <div className="flex gap-3 mb-1">
        <button
          onClick={() => setEnabledTools(null)}
          className="text-xs transition-colors"
          style={{ color: 'var(--accent)' }}
        >
          {t('tools.enable_all')}
        </button>
        <button
          onClick={() => setEnabledTools([])}
          className="text-xs transition-colors"
          style={{ color: 'var(--text-3)' }}
        >
          {t('tools.disable_all')}
        </button>
      </div>

      {groups.map((group) => {
        const items = toolInfos.filter((ti) => ti.group === group)
        if (items.length === 0) return null
        return (
          <div key={group}>
            <p className="label-mono mb-1.5">{group}</p>
            <div className="space-y-1">
              {items.map((ti) => (
                <label key={ti.name} className="flex items-start gap-2 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={active.includes(ti.name)}
                    onChange={() => toggle(ti.name)}
                    className="mt-0.5"
                  />
                  <div>
                    <span className="text-xs font-medium" style={{ color: 'var(--text-1)' }}>{ti.name}</span>
                    <span className="text-xs ml-1" style={{ color: 'var(--text-3)' }}>— {ti.description}</span>
                  </div>
                </label>
              ))}
            </div>
          </div>
        )
      })}
    </div>
  )
}
