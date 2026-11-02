import { useMemo, type CSSProperties, type ReactNode } from 'react'
import { Card, CardBack, CardSlot } from '../cards'
import { chips, seatWho } from '../format'
import { POSITION_NAMES } from '../learn/positions'
import { badgeText, badgeTitle } from '../learn/styles'
import type { PlayerView, PositionLabel, SeatInfo, SeatStats, TableView } from '../types'

const STATUS_TEXT: Record<string, string> = { folded: 'Folded', all_in: 'All in', eliminated: 'Out' }

function seatCards(view: TableView, p: PlayerView, revealed: Record<string, string[]> | null | undefined) {
  if (p.seat === view.hero_seat && view.hole_cards) {
    return view.hole_cards.map((c) => <Card key={c} card={c} size="sm" />)
  }
  if (p.shown_cards) return p.shown_cards.map((c) => <Card key={c} card={c} size="sm" />)
  const reviewed = revealed?.[String(p.seat)]
  if (reviewed && p.status !== 'eliminated') {
    return reviewed.map((c) => <Card key={c} card={c} size="sm" muted={p.status === 'folded'} />)
  }
  if (!view.hand_id || p.status === 'folded' || p.status === 'eliminated') return null
  const dealt = view.action_history.filter((e) => e.type === 'HoleDealt' && e.data.seat === p.seat).length
  return Array.from({ length: dealt }, (_, i) => <CardBack key={i} size="sm" />)
}

function Markers({ view, seat, position }: { view: TableView; seat: number; position?: PositionLabel }) {
  return (
    <span className="markers">
      {view.button_seat === seat && (
        <span className="marker marker-d" title="Dealer button">
          D
        </span>
      )}
      {position && position !== 'BTN' && (
        <span className="marker marker-pos" title={POSITION_NAMES[position]} data-testid={`pos-badge-${seat}`}>
          {position}
        </span>
      )}
      {!position && view.small_blind_seat === seat && (
        <span className="marker" title="Small blind">
          SB
        </span>
      )}
      {!position && view.big_blind_seat === seat && (
        <span className="marker" title="Big blind">
          BB
        </span>
      )}
    </span>
  )
}

function Seat({
  view,
  player,
  info,
  active,
  pos,
  timer,
  revealed,
  won,
  position,
  stats,
  name,
}: {
  view: TableView
  player: PlayerView
  info?: SeatInfo
  active: boolean
  pos: { x: number; y: number }
  timer?: ReactNode
  revealed?: Record<string, string[]> | null
  won?: number
  position?: PositionLabel
  stats?: SeatStats
  name?: string
}) {
  const hero = player.seat === view.hero_seat
  const cards = seatCards(view, player, revealed)
  const count = Array.isArray(cards) ? cards.length : 0
  const classes = ['seat', `seat-st-${player.status}`, active && 'seat-active', hero && 'seat-hero', won && 'seat-won']
    .filter(Boolean)
    .join(' ')
  const who = hero ? 'You' : name ?? seatWho(info)
  return (
    <div
      className={classes}
      style={{ '--x': `${pos.x}%`, '--y': `${pos.y}%` } as CSSProperties}
      data-testid={`seat-${player.seat}`}
      aria-current={active ? 'true' : undefined}
      aria-label={`Seat ${player.seat}, ${who}, ${chips(player.stack)} chips${active ? ', to act' : ''}${
        STATUS_TEXT[player.status] ? `, ${STATUS_TEXT[player.status].toLowerCase()}` : ''
      }`}
    >
      <div className="seat-head">
        <span className="mono seat-id">Seat {player.seat}</span>
        <Markers view={view} seat={player.seat} position={position} />
      </div>
      <div className="seat-who" title={info?.artifact_hash ? `${who} · sha256 ${info.artifact_hash}` : who}>
        {who}
      </div>
      <div className={`seat-cards seat-cards-${count}`}>{cards}</div>
      <div className="seat-foot">
        <span className="mono seat-stack">{chips(player.stack)}</span>
        {active && <span className="seat-badge seat-badge-act">To act</span>}
        {!!won && <span className="seat-badge seat-badge-won">Wins {chips(won)}</span>}
        {STATUS_TEXT[player.status] && <span className="seat-status">{STATUS_TEXT[player.status]}</span>}
        {player.street_committed > 0 && <span className="mono seat-bet">bet {chips(player.street_committed)}</span>}
      </div>
      {stats && !hero && stats.hands > 0 && (
        <div className="seat-stats" data-testid={`stats-badge-${player.seat}`} title={badgeTitle(stats)}>
          {badgeText(stats)}
        </div>
      )}
      {active && timer && <div className="seat-timer">{timer}</div>}
    </div>
  )
}

/** Seat centres on an ellipse, the viewer's own seat at the bottom. CSS clamps keep every seat inside the felt. */
export function useSeatPositions(seats: number[], heroSeat: number | null) {
  return useMemo(() => {
    const n = seats.length
    const heroIndex = Math.max(0, heroSeat !== null ? seats.indexOf(heroSeat) : 0)
    return Object.fromEntries(
      seats.map((seat, i) => {
        const k = (i - heroIndex + n) % n
        const angle = Math.PI / 2 + (2 * Math.PI * k) / n
        const c = Math.cos(angle)
        const s = Math.sin(angle)
        return [seat, { x: 50 + 42 * c, y: 50 + 40 * s, bx: 50 + 27 * c, by: 50 + 22 * s }]
      }),
    ) as Record<number, { x: number; y: number; bx: number; by: number }>
  }, [seats, heroSeat])
}

export function Felt({
  view,
  seats,
  timer,
  revealed,
  winners,
  label = 'Table',
  seatLabels,
  stats,
  names,
}: {
  view: TableView
  seats: SeatInfo[]
  timer?: ReactNode
  revealed?: Record<string, string[]> | null
  winners?: Record<number, number>
  label?: string
  seatLabels?: Record<string, { label: PositionLabel }>
  stats?: Record<string, SeatStats>
  names?: Record<number, string>
}) {
  const positions = useSeatPositions(view.seats, view.hero_seat)
  const info = Object.fromEntries(seats.map((s) => [s.seat, s]))
  const bets = view.pot_total - view.settled_pot
  return (
    <section className={`felt felt-${view.seats.length}`} aria-label={label}>
      <div className="felt-oval" aria-hidden="true" />
      {view.players.map((p) => (
        <Seat
          key={p.seat}
          view={view}
          player={p}
          info={info[p.seat]}
          active={view.actor_seat === p.seat}
          pos={positions[p.seat]}
          timer={timer}
          revealed={revealed}
          won={winners?.[p.seat]}
          position={seatLabels?.[String(p.seat)]?.label}
          stats={stats?.[String(p.seat)]}
          name={names?.[p.seat]}
        />
      ))}
      {view.players
        .filter((p) => p.street_committed > 0)
        .map((p) => (
          <span
            key={p.seat}
            className="bet-chip mono"
            style={{ left: `${positions[p.seat].bx}%`, top: `${positions[p.seat].by}%` }}
          >
            {chips(p.street_committed)}
          </span>
        ))}
      <div className="center">
        <div className="board" aria-label="Board">
          {Array.from({ length: 5 }, (_, i) =>
            view.board[i] ? <Card key={view.board[i]} card={view.board[i]} size="md" /> : <CardSlot key={i} size="md" />,
          )}
        </div>
        <div className="pots" data-testid="pots">
          <span>
            <span className="label">Pot</span> <span className="mono pot-total">{chips(view.pot_total)}</span>
          </span>
          {view.pots.map((pot, i) => (
            <span key={i} className="pot" title={`Eligible: seats ${pot.eligible.join(', ')}`}>
              <span className="label">{i === 0 ? 'Main' : `Side ${i}`}</span> <span className="mono">{chips(pot.amount)}</span>
            </span>
          ))}
          {bets > 0 && (
            <span className="pot">
              <span className="label">Bets</span> <span className="mono">{chips(bets)}</span>
            </span>
          )}
        </div>
        {view.street && <div className="street mono">{view.street}</div>}
      </div>
    </section>
  )
}
