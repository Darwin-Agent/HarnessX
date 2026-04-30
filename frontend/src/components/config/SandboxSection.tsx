import { useSlotsStore } from '../../store/slots'
import { useT } from '../../i18n'

export function SandboxSection() {
  const t = useT()
  const {
    sandboxType, sandboxUrl, workspaceDir,
    setSandboxType, setSandboxUrl, setWorkspaceDir,
  } = useSlotsStore()

  return (
    <div className="space-y-3">
      {/* Sandbox type */}
      <div className="flex gap-3">
        {(['local', 'remote'] as const).map((type) => (
          <label key={type} className="flex items-center gap-1.5 cursor-pointer">
            <input
              type="radio"
              name="sandbox"
              value={type}
              checked={sandboxType === type}
              onChange={() => setSandboxType(type)}
            />
            <span className="text-xs capitalize" style={{ color: 'var(--text-2)' }}>{type}</span>
          </label>
        ))}
      </div>

      {/* Remote sandbox URL */}
      {sandboxType === 'remote' && (
        <div>
          <label className="block mb-1 label-mono">Sandbox URL</label>
          <input
            type="text"
            value={sandboxUrl}
            onChange={(e) => setSandboxUrl(e.target.value)}
            placeholder="https://sandbox.example.com"
            className="w-full text-xs rounded px-2 py-1.5"
          />
        </div>
      )}

      {/* Workspace directory (local sandbox) */}
      {sandboxType === 'local' && (
        <div>
          <label className="block mb-1 label-mono">{t('sandbox.workspace_dir')}</label>
          <input
            type="text"
            value={workspaceDir}
            onChange={(e) => setWorkspaceDir(e.target.value)}
            placeholder="~/.harnessx/workspace"
            className="w-full text-xs rounded px-2 py-1.5"
          />
          <p className="text-xs mt-0.5" style={{ color: 'var(--text-4)' }}>
            {t('sandbox.workspace_hint')}
          </p>
        </div>
      )}
    </div>
  )
}
