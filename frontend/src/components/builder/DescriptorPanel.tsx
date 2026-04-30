import { useLabStore } from '../../store/lab'
import { DimensionCard } from './DimensionCard'
import { useT } from '../../i18n'

type ProcessorList = Record<string, unknown>[]

interface DescriptorPanelProps {
  processors?:          ProcessorList
  onProcessorsChange?:  (p: ProcessorList) => void
  compact?:             boolean
  readOnly?:            boolean
}

export function DescriptorPanel(props: DescriptorPanelProps = {}) {
  const store = useLabStore()
  const t = useT()
  const dimensions         = store.dimensions
  const processors         = props.processors         ?? store.harnessConfig.processors
  const onProcessorsChange = props.onProcessorsChange ?? store.updateProcessors
  const compact            = props.compact ?? false
  const readOnly           = props.readOnly ?? false

  if (dimensions.length === 0) {
    return (
      <div
        className="flex-1 flex items-center justify-center text-xs font-mono"
        style={{ color: 'var(--text-4)' }}
      >
        {t('builder.loading_dims')}
      </div>
    )
  }

  const gridCls = compact
    ? 'grid grid-cols-1 sm:grid-cols-2 gap-2'
    : 'grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3'

  return (
    <div
      className={compact ? 'p-3' : 'flex-1 overflow-y-auto p-4'}
      style={{ background: 'var(--bg-base)' }}
    >
      <div className={gridCls}>
        {dimensions.map((dim, i) => (
          <div
            key={dim.key}
            className="animate-card-in"
            style={{ animationDelay: `${i * 45}ms` }}
          >
            <DimensionCard
              dimension={dim}
              processors={processors}
              onProcessorsChange={onProcessorsChange}
              readOnly={readOnly}
            />
          </div>
        ))}
      </div>
    </div>
  )
}
