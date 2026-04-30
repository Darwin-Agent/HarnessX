import { BookOpen } from 'lucide-react'
import { useSlotsStore } from '../../store/slots'
import { useLabStore } from '../../store/lab'
import { useT } from '../../i18n'

export function SkillsPage() {
  const t = useT()
  const skillInfos  = useSlotsStore((s) => s.skillInfos)
  const enabledSkills = useSlotsStore((s) => s.enabledSkills)
  const setEnabledSkills = useSlotsStore((s) => s.setEnabledSkills)

  const { harnessConfig, updateProcessors, saveCurrentHarness } = useLabStore()
  const SKILL_LOAD_TARGET = 'harnessx.processors.tools.skill_loader.ProgressiveSkillLoader'
  const skillLoading = harnessConfig.processors.some((p) => p._target_ === SKILL_LOAD_TARGET)

  function toggleSkillLoading() {
    if (skillLoading) {
      // Turning OFF: remove loader processor AND set enabled_skills=[] so
      // DefaultSystemPromptBuilder also stops listing skills in system prompt
      updateProcessors(harnessConfig.processors.filter((p) => p._target_ !== SKILL_LOAD_TARGET))
      setEnabledSkills([])
    } else {
      // Turning ON: add loader processor AND restore enabled_skills=null (all)
      updateProcessors([...harnessConfig.processors, { _target_: SKILL_LOAD_TARGET }])
      setEnabledSkills(null)
    }
    // Auto-save so the next run immediately uses the updated config
    setTimeout(() => saveCurrentHarness(), 0)
  }

  function isSkillEnabled(name: string) {
    return enabledSkills === null || enabledSkills.includes(name)
  }

  function toggleSkill(name: string) {
    if (enabledSkills === null) {
      setEnabledSkills(skillInfos.map((s) => s.name).filter((n) => n !== name))
    } else {
      const next = isSkillEnabled(name)
        ? enabledSkills.filter((n) => n !== name)
        : [...enabledSkills, name]
      setEnabledSkills(next.length === skillInfos.length ? null : next)
    }
  }

  return (
    <div className="space-y-5">
      {/* Global auto-inject toggle */}
      <div
        className="flex items-center justify-between rounded-2xl px-5 py-4"
        style={{ border: '1px solid var(--border)', background: 'var(--bg-card)' }}
      >
        <div>
          <div className="font-medium" style={{ fontSize: '15px', color: 'var(--text-1)' }}>
            {t('skills.enable')}
          </div>
          <p className="mt-1" style={{ fontSize: '13px', color: 'var(--text-3)', lineHeight: 1.5 }}>
            {skillLoading ? t('skills.auto_inject') : t('skills.available')}
          </p>
        </div>
        <button
          onClick={toggleSkillLoading}
          className="px-4 py-2 rounded-xl font-medium transition-all duration-150 shrink-0"
          style={{
            fontSize: '13px',
            ...(skillLoading
              ? { background: 'rgba(16,185,129,0.08)', color: '#10b981', border: '1px solid rgba(16,185,129,0.2)' }
              : { background: 'var(--bg-elevated)', color: 'var(--text-3)', border: '1px solid var(--border)' }),
          }}
        >
          {skillLoading ? 'ON' : 'OFF'}
        </button>
      </div>

      {/* Skill cards */}
      <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))' }}>
        {skillInfos.length === 0 && (
          <p style={{ fontSize: '14px', color: 'var(--text-4)', gridColumn: '1/-1' }}>{t('skills.loading')}</p>
        )}
        {skillInfos.map((skill) => {
          const on = isSkillEnabled(skill.name)
          return (
            <div
              key={skill.name}
              className="rounded-2xl p-4 transition-all duration-150"
              style={{
                border: on ? '1px solid var(--accent-ring)' : '1px solid var(--border)',
                background: on ? 'var(--accent-bg)' : 'var(--bg-elevated)',
                opacity: skillLoading ? 1 : 0.65,
              }}
            >
              <div className="flex items-start gap-3">
                <div
                  className="rounded-xl p-2 shrink-0"
                  style={{ background: on ? 'var(--accent-bg)' : 'var(--bg-card)', border: '1px solid var(--border)' }}
                >
                  <BookOpen size={14} style={{ color: on ? 'var(--accent)' : 'var(--text-3)' }} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="font-medium truncate" style={{ fontSize: '14px', color: on ? 'var(--accent)' : 'var(--text-1)' }}>
                    {skill.name}
                  </div>
                  <p className="mt-1 line-clamp-2" style={{ fontSize: '12px', color: 'var(--text-3)', lineHeight: 1.5 }}>
                    {skill.description}
                  </p>
                </div>
                <button
                  onClick={() => skillLoading && toggleSkill(skill.name)}
                  disabled={!skillLoading}
                  className="px-2.5 py-1 rounded-lg font-medium transition-all duration-150 shrink-0 disabled:cursor-default"
                  style={{
                    fontSize: '11px',
                    ...(on
                      ? { background: 'rgba(16,185,129,0.08)', color: '#10b981', border: '1px solid rgba(16,185,129,0.2)' }
                      : { background: 'var(--bg-card)', color: 'var(--text-4)', border: '1px solid var(--border)' }),
                  }}
                >
                  {on ? 'ON' : 'OFF'}
                </button>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
