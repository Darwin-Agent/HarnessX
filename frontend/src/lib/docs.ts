const recipePages: Record<string, string> = {
  'recipes/coevolution': 'recipe/coevolution/README.md',
  'recipes/verl': 'recipe/verl/README.md',
}

export function resolveDocLink(href: string, currentPath: string | null) {
  if (href.startsWith('#')) return null
  const source = recipePages[currentPath || ''] || `docs/${currentPath || 'index'}.md`
  const base = 'https://harnessx.invalid/'
  const target = new URL(href, new URL(source, base))
  if (target.origin !== new URL(base).origin) return null
  const filename = target.pathname.slice(1)
  if (!filename.endsWith('.md')) return null
  const isChinese = filename.endsWith('.zh-CN.md')
  const canonical = filename.replace(/\.zh-CN\.md$/, '.md')
  const recipePath = Object.entries(recipePages).find(([, value]) => value === canonical)?.[0]
  const path = recipePath || (canonical.startsWith('docs/') ? canonical.slice(5, -3) : null)
  if (!path) return null
  const lang = isChinese ? 'zh' : path === currentPath ? 'en' : undefined
  return { path, lang } as { path: string; lang?: 'en' | 'zh' }
}
