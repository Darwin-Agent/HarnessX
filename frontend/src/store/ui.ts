import { create } from 'zustand'
import { persist } from 'zustand/middleware'

type Theme = 'light' | 'dark'
type Lang  = 'en' | 'zh'
type ThinkingPostStream = 'collapse' | 'keep'

interface UIState {
  theme:        Theme
  lang:         Lang
  settingsOpen: boolean
  modelOpen:    boolean
  fontSize:     number   // scale factor: 0.85 – 1.20 (default 1.0)
  thinkingPostStream: ThinkingPostStream
  toggleTheme:     () => void
  setLang:         (l: Lang) => void
  setSettingsOpen: (open: boolean) => void
  setModelOpen:    (open: boolean) => void
  setFontSize:     (scale: number) => void
  setThinkingPostStream: (mode: ThinkingPostStream) => void
}

export const useUIStore = create<UIState>()(
  persist(
    (set, get) => ({
      theme:        'light',
      lang:         'en',
      settingsOpen: false,
      modelOpen:    false,
      fontSize:     1.0,
      thinkingPostStream: 'collapse',
      toggleTheme:     () => set({ theme: get().theme === 'light' ? 'dark' : 'light' }),
      setLang:         (lang) => set({ lang }),
      setSettingsOpen: (settingsOpen) => set({ settingsOpen }),
      setModelOpen:    (modelOpen) => set({ modelOpen }),
      setFontSize:     (fontSize) => set({ fontSize }),
      setThinkingPostStream: (thinkingPostStream) => set({ thinkingPostStream }),
    }),
    {
      name: 'harness-ui',
      partialize: (s) => ({
        theme: s.theme,
        lang: s.lang,
        fontSize: s.fontSize,
        thinkingPostStream: s.thinkingPostStream,
      }),
    },
  ),
)
