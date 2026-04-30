import { useLabStore } from '../../store/lab'
import { useSlotsStore } from '../../store/slots'
import { useT } from '../../i18n'

const SKILL_LOAD_TARGET = 'harnessx.processors.tools.skill_loader.ProgressiveSkillLoader'

export function SkillsSection() {
  const t = useT()
  const { harnessConfig, updateProcessors } = useLabStore()
  const { skillInfos, setEnabledSkills } = useSlotsStore()
  const enabled = harnessConfig.processors.some((p) => p._target_ === SKILL_LOAD_TARGET)

  function setEnabled(on: boolean) {
    if (on) {
      updateProcessors([...harnessConfig.processors, { _target_: SKILL_LOAD_TARGET }])
      // Sync enabledSkills: null = all enabled, consistent with SkillsPage toggle ON
      setEnabledSkills(null)
    } else {
      updateProcessors(harnessConfig.processors.filter((p) => p._target_ !== SKILL_LOAD_TARGET))
      // Sync enabledSkills: [] = none, consistent with SkillsPage toggle OFF
      setEnabledSkills([])
    }
  }

  return (
    <div className="space-y-2">
      <label className="flex items-center gap-2 cursor-pointer">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => setEnabled(e.target.checked)}
        />
        <span className="text-xs font-medium" style={{ color: 'var(--text-1)' }}>
          {t('skills.enable')}
        </span>
      </label>

      <p className="text-xs" style={{ color: 'var(--text-3)' }}>
        {enabled ? t('skills.auto_inject') : t('skills.available')}
      </p>

      <ul className="space-y-1 max-h-48 overflow-y-auto pr-1">
        {skillInfos.length === 0 ? (
          <li className="text-xs italic" style={{ color: 'var(--text-4)' }}>{t('skills.loading')}</li>
        ) : (
          skillInfos.map((s) => (
            <li key={s.name}>
              <span className="text-xs font-medium" style={{ color: enabled ? 'var(--text-1)' : 'var(--text-3)' }}>
                {s.name}
              </span>
              {s.description && (
                <span className="text-xs ml-1" style={{ color: 'var(--text-4)' }}>
                  — {s.description.length > 60 ? s.description.slice(0, 57) + '…' : s.description}
                </span>
              )}
            </li>
          ))
        )}
      </ul>
    </div>
  )
}
