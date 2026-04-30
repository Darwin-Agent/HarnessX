import type { DimensionOption } from '../../api/types'

interface OptionChipProps {
  option: DimensionOption
  selected: boolean
  onSelect: () => void
}

export function OptionChip({ option, selected, onSelect }: OptionChipProps) {
  return (
    <button
      onClick={onSelect}
      title={option.description}
      className={selected ? 'chip chip-selected' : 'chip chip-default'}
    >
      {option.label}
    </button>
  )
}
