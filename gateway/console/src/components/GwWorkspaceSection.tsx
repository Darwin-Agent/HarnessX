import { useEffect, useState } from 'react'
import { Folder } from 'lucide-react'
import { useSlotsStore } from '@lab/store/slots'
import { useT } from '@gw/i18n'
import { api } from '@gw/api/client'

export function GwWorkspaceSection() {
  const t = useT()
  const agentId = useSlotsStore((s) => s.agentId)
  const [homePath, setHomePath] = useState<string>('')

  useEffect(() => {
    api.getHome().then((h) => setHomePath(h.home)).catch(console.error)
  }, [])

  const wsPath = homePath
    ? `${homePath}/im-workspaces/${agentId}/web_ui`
    : `~/.harnessx/im-workspaces/${agentId}/web_ui`

  return (
    <div className="px-6 py-4 max-w-lg" style={{ borderBottom: '1px solid var(--border)' }}>
      <label
        className="font-semibold block mb-1"
        style={{ fontSize: 11, color: 'var(--text-3)', textTransform: 'uppercase', letterSpacing: '0.06em' }}
      >
        {t('gw.chat.workspace')}
      </label>
      <p style={{ fontSize: 12, color: 'var(--text-4)', marginBottom: 8 }}>
        {t('gw.settings.agent_id_hint')}
      </p>
      <div
        className="flex items-center gap-2 px-3 py-2 rounded-lg font-mono"
        style={{ fontSize: 12, background: 'var(--bg-elevated)', border: '1px solid var(--border)', color: 'var(--text-2)' }}
      >
        <Folder size={13} style={{ color: 'var(--text-4)', flexShrink: 0 }} />
        <span className="truncate">{wsPath}</span>
      </div>
    </div>
  )
}
