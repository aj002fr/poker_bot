const SUITS: Record<string, { symbol: string; name: string; red: boolean }> = {
  s: { symbol: '♠', name: 'spades', red: false },
  h: { symbol: '♥', name: 'hearts', red: true },
  d: { symbol: '♦', name: 'diamonds', red: true },
  c: { symbol: '♣', name: 'clubs', red: false },
}

const RANK_NAMES: Record<string, string> = {
  A: 'ace', K: 'king', Q: 'queen', J: 'jack', T: 'ten',
  '9': 'nine', '8': 'eight', '7': 'seven', '6': 'six', '5': 'five', '4': 'four', '3': 'three', '2': 'two',
}

export function rankLabel(rank: string) {
  return rank === 'T' ? '10' : rank
}

export function cardText(card: string) {
  return `${rankLabel(card[0])}${SUITS[card[1]]?.symbol ?? card[1]}`
}

export function Card({ card, size = 'md' }: { card: string; size?: 'sm' | 'md' | 'lg' }) {
  const suit = SUITS[card[1]]
  const rank = card[0]
  return (
    <span
      className={`card card-${size}${suit?.red ? ' card-red' : ''}`}
      data-card={card}
      role="img"
      aria-label={`${RANK_NAMES[rank] ?? rank} of ${suit?.name ?? card[1]}`}
    >
      <span className="card-rank">{rankLabel(rank)}</span>
      <span className="card-suit">{suit?.symbol}</span>
    </span>
  )
}

export function CardBack({ size = 'md' }: { size?: 'sm' | 'md' | 'lg' }) {
  return <span className={`card card-${size} card-back`} role="img" aria-label="face-down card" />
}

export function CardSlot({ size = 'md' }: { size?: 'sm' | 'md' | 'lg' }) {
  return <span className={`card card-${size} card-slot`} aria-hidden="true" />
}
