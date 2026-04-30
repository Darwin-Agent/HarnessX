import { create } from 'zustand'

interface DocsState {
  isOpen: boolean
  /** Current doc path, e.g. "feats/plugins". null = show index/home. */
  path:   string | null
  open:   (path?: string | null) => void
  close:  () => void
  navigate: (path: string) => void
}

export const useDocsStore = create<DocsState>()((set) => ({
  isOpen: false,
  path:   null,
  open:   (path = null) => set({ isOpen: true, path }),
  close:  () => set({ isOpen: false }),
  navigate: (path) => set({ path }),
}))
