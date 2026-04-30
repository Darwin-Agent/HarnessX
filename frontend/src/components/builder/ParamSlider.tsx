import type { DimensionParam } from '../../api/types'

interface ParamSliderProps {
  param: DimensionParam
  value: number
  onChange: (v: number) => void
}

export function ParamSlider({ param, value, onChange }: ParamSliderProps) {
  const step = param.step ?? (param.type === 'int' ? 1 : 0.1)

  return (
    <div className="flex items-center gap-2 mt-2">
      <label className="text-xs w-28 shrink-0 leading-tight" style={{ color: 'var(--text-3)' }}>{param.label}</label>
      <input
        type="range"
        min={param.min}
        max={param.max}
        step={step}
        value={value}
        onChange={(e) => {
          const raw = parseFloat(e.target.value)
          onChange(param.type === 'int' ? Math.round(raw) : raw)
        }}
        className="range-slider flex-1"
      />
      <span
        className="text-xs font-mono w-10 text-right tabular-nums"
        style={{ color: 'var(--accent)', fontWeight: 500 }}
      >
        {value}
      </span>
    </div>
  )
}
