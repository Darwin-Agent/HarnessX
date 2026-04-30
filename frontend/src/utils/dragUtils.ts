import type { Dimension, DimensionDragPayload } from '../api/types'

export const DRAG_TYPE = 'application/x-dimension-patch'

/**
 * Collect the active processors that belong to a given dimension.
 * "Belong" means the processor's _target_ appears in at least one option
 * of that dimension.
 */
export function extractDimensionProcessors(
  dim: Dimension,
  processors: Record<string, unknown>[],
): Record<string, unknown>[] {
  const ownedTargets = new Set(
    dim.options.flatMap((o) => o.processors.map((p) => p._target_ as string)),
  )
  return processors.filter((p) => ownedTargets.has(p._target_ as string))
}

export function buildDragPayload(
  dim: Dimension,
  processors: Record<string, unknown>[],
): DimensionDragPayload {
  return {
    dimensionKey:   dim.key,
    dimensionLabel: dim.label,
    processors:     extractDimensionProcessors(dim, processors),
  }
}

export function parseDragPayload(e: { dataTransfer: DataTransfer | null }): DimensionDragPayload | null {
  try {
    const raw = e.dataTransfer?.getData(DRAG_TYPE)
    if (!raw) return null
    return JSON.parse(raw) as DimensionDragPayload
  } catch {
    return null
  }
}
