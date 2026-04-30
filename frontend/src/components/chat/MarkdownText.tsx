import { Fragment, useMemo, type ReactNode } from 'react'
import { marked, type Token, type Tokens } from 'marked'

const MARKDOWN_OPTIONS = { gfm: true, breaks: true } as const
const SAFE_HREF_RE = /^(https?:|mailto:|\/|#)/i

interface Props {
  content: string
  className?: string
}

function safeHref(href: string): string | null {
  const normalized = href.trim()
  if (!normalized) return null
  return SAFE_HREF_RE.test(normalized) ? normalized : null
}

function renderInline(tokens: readonly Token[], keyPrefix: string): ReactNode[] {
  return tokens.map((token, idx) => {
    const key = `${keyPrefix}-in-${idx}`

    switch (token.type) {
      case 'text':
        if (token.tokens && token.tokens.length > 0) {
          return <Fragment key={key}>{renderInline(token.tokens, key)}</Fragment>
        }
        return <Fragment key={key}>{token.text}</Fragment>
      case 'escape':
        return <Fragment key={key}>{token.text}</Fragment>
      case 'strong':
        return <strong key={key}>{renderInline(token.tokens ?? [], key)}</strong>
      case 'em':
        return <em key={key}>{renderInline(token.tokens ?? [], key)}</em>
      case 'del':
        return <del key={key}>{renderInline(token.tokens ?? [], key)}</del>
      case 'codespan':
        return <code key={key}>{token.text}</code>
      case 'br':
        return <br key={key} />
      case 'link': {
        const href = safeHref(token.href)
        const text = renderInline(token.tokens ?? [], key)
        if (!href) return <Fragment key={key}>{text}</Fragment>
        const external = /^https?:/i.test(href)
        return (
          <a
            key={key}
            href={href}
            target={external ? '_blank' : undefined}
            rel={external ? 'noopener noreferrer' : undefined}
          >
            {text}
          </a>
        )
      }
      case 'image': {
        const alt = token.text || 'image'
        return <span key={key}>{`[image: ${alt}]`}</span>
      }
      case 'html':
        return <Fragment key={key}>{token.raw}</Fragment>
      default:
        if ('tokens' in token && Array.isArray(token.tokens)) {
          return <Fragment key={key}>{renderInline(token.tokens, key)}</Fragment>
        }
        return <Fragment key={key}>{token.raw ?? ''}</Fragment>
    }
  })
}

function renderListItem(item: Tokens.ListItem, keyPrefix: string): ReactNode {
  if (item.tokens.length === 1) {
    const only = item.tokens[0]
    if (only.type === 'paragraph') return renderInline(only.tokens ?? [], `${keyPrefix}-p`)
    if (only.type === 'text') return renderInline(only.tokens ?? [only], `${keyPrefix}-t`)
  }
  return renderBlocks(item.tokens, `${keyPrefix}-blocks`)
}

function renderBlocks(tokens: readonly Token[], keyPrefix: string): ReactNode[] {
  return tokens.map((token, idx) => {
    const key = `${keyPrefix}-bl-${idx}`

    switch (token.type) {
      case 'space':
        return null
      case 'paragraph':
        return <p key={key}>{renderInline(token.tokens ?? [], key)}</p>
      case 'heading':
        if (token.depth <= 1) return <h1 key={key}>{renderInline(token.tokens ?? [], key)}</h1>
        if (token.depth === 2) return <h2 key={key}>{renderInline(token.tokens ?? [], key)}</h2>
        if (token.depth === 3) return <h3 key={key}>{renderInline(token.tokens ?? [], key)}</h3>
        return <h4 key={key}>{renderInline(token.tokens ?? [], key)}</h4>
      case 'blockquote':
        return <blockquote key={key}>{renderBlocks(token.tokens ?? [], key)}</blockquote>
      case 'list': {
        const listToken = token as Tokens.List
        if (listToken.ordered) {
          return (
            <ol key={key} start={typeof listToken.start === 'number' ? listToken.start : undefined}>
              {listToken.items.map((item: Tokens.ListItem, itemIdx: number) => (
                <li key={`${key}-item-${itemIdx}`}>
                  {renderListItem(item, `${key}-item-${itemIdx}`)}
                </li>
              ))}
            </ol>
          )
        }
        return (
          <ul key={key}>
            {listToken.items.map((item: Tokens.ListItem, itemIdx: number) => (
              <li key={`${key}-item-${itemIdx}`}>
                {renderListItem(item, `${key}-item-${itemIdx}`)}
              </li>
            ))}
          </ul>
        )
      }
      case 'code': {
        const codeToken = token as Tokens.Code
        return (
          <pre key={key}>
            <code>{codeToken.text}</code>
          </pre>
        )
      }
      case 'hr':
        return <hr key={key} />
      case 'table': {
        const tableToken = token as Tokens.Table
        return (
          <table key={key}>
            <thead>
              <tr>
                {tableToken.header.map((cell: Tokens.TableCell, cellIdx: number) => (
                  <th
                    key={`${key}-head-${cellIdx}`}
                    style={cell.align ? { textAlign: cell.align } : undefined}
                  >
                    {renderInline(cell.tokens, `${key}-head-${cellIdx}`)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {tableToken.rows.map((row: Tokens.TableCell[], rowIdx: number) => (
                <tr key={`${key}-row-${rowIdx}`}>
                  {row.map((cell: Tokens.TableCell, cellIdx: number) => (
                    <td
                      key={`${key}-row-${rowIdx}-cell-${cellIdx}`}
                      style={cell.align ? { textAlign: cell.align } : undefined}
                    >
                      {renderInline(cell.tokens, `${key}-row-${rowIdx}-cell-${cellIdx}`)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )
      }
      case 'html':
        return (
          <pre key={key}>
            <code>{token.raw}</code>
          </pre>
        )
      case 'text':
        if (token.tokens && token.tokens.length > 0) {
          return <p key={key}>{renderInline(token.tokens, key)}</p>
        }
        return <p key={key}>{token.text}</p>
      case 'def':
        return null
      default:
        if ('tokens' in token && Array.isArray(token.tokens)) {
          return <p key={key}>{renderInline(token.tokens, key)}</p>
        }
        return <p key={key}>{token.raw ?? ''}</p>
    }
  })
}

export function MarkdownText({ content, className }: Props) {
  const rootClass = className ? `chat-md ${className}` : 'chat-md'
  const rendered = useMemo(() => {
    try {
      const tokens = marked.lexer(content, MARKDOWN_OPTIONS)
      return renderBlocks(tokens, 'root')
    } catch {
      return null
    }
  }, [content])

  if (!rendered) {
    return (
      <div className={rootClass}>
        <span className="whitespace-pre-wrap break-words">{content}</span>
      </div>
    )
  }

  return <div className={rootClass}>{rendered}</div>
}
