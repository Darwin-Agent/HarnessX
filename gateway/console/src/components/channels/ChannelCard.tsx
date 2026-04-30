import type { ChannelInfo } from '@gw/api/types'

const STATE_COLORS: Record<string, { bg: string; text: string }> = {
  online:     { bg: 'rgba(52,199,89,0.12)',  text: '#34c759' },
  connecting: { bg: 'rgba(255,159,10,0.12)', text: '#ff9f0a' },
  offline:    { bg: 'rgba(142,142,147,0.15)',text: '#8e8e93' },
  error:      { bg: 'rgba(255,59,48,0.12)',  text: '#ff3b30' },
}

const PLATFORM_COLORS: Record<string, string> = {
  feishu:   '#00b96b',
  telegram: '#2aabee',
  slack:    '#4a154b',
  discord:  '#5865f2',
  dingtalk: '#1677ff',
}

interface Props {
  channel: ChannelInfo
  selected: boolean
  onClick: () => void
}

export function ChannelCard({ channel, selected, onClick }: Props) {
  const stateStyle = STATE_COLORS[channel.connection_state] ?? STATE_COLORS.offline
  const platformColor = PLATFORM_COLORS[channel.name] ?? 'var(--accent)'
  const initials = channel.display_name.slice(0, 2).toUpperCase()

  return (
    <div
      className="flex items-center gap-3 px-3 py-2.5 rounded-lg cursor-pointer transition-all duration-100"
      style={{
        background: selected ? 'var(--accent-bg)' : 'transparent',
        border: `1px solid ${selected ? 'var(--accent-ring)' : 'transparent'}`,
      }}
      onClick={onClick}
    >
      {/* Platform badge */}
      <div
        className="flex items-center justify-center rounded-lg shrink-0 font-bold"
        style={{
          width: 32,
          height: 32,
          fontSize: 11,
          background: `${platformColor}22`,
          color: platformColor,
          border: `1px solid ${platformColor}44`,
          letterSpacing: '0.02em',
        }}
      >
        {initials}
      </div>

      {/* Name */}
      <div className="flex flex-col min-w-0 flex-1">
        <span
          className="font-medium truncate"
          style={{ fontSize: 13, color: selected ? 'var(--accent)' : 'var(--text-1)' }}
        >
          {channel.display_name}
        </span>
        <span className="truncate" style={{ fontSize: 11, color: 'var(--text-4)', fontFamily: 'monospace' }}>
          {channel.name}
        </span>
      </div>

      {/* State badge */}
      <span
        className="shrink-0 px-1.5 py-0.5 rounded"
        style={{ fontSize: 10, fontWeight: 600, ...stateStyle }}
      >
        {channel.connection_state}
      </span>
    </div>
  )
}
