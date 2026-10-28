import { useEffect, useMemo, useState } from 'react'
import type { StreamStatus } from '../api'
import { Card, CardBack, CardSlot } from '../cards'
import { RULESETS, chips, describeEvents } from '../format'
import type { PlayerView, SeatInfo, Snapshot, TableView } from '../types'
import { ActionPanel } from './ActionPanel'

function Timer({ snapshot }: { snapshot: Snapshot }) {
  const turn = snapshot.turn
  const [deadline, setDeadline] = useState<number | null>(null)
  const [now, setNow] = useState(() => performance.now())

  useEffect(() => {
    setDeadline(turn?.remaining_ms != null ? performance.now() + turn.remaining_ms : null)
  }, [turn?.decision_id, snapshot.version, turn?.remaining_ms])

  useEffect(() => {
    if (deadline === null) return
    const id = window.setInterval(() => setNow(performance.now()), 200)
    return () => window.clearInterval(id)
  }, [deadline])

  if (!turn || turn.kind !== 'human' || deadline === null || !turn.turn_ms) return null
  const left = Math.max(0, deadline - now)
  const frac = left / turn.turn_ms
  return (
    <span className={`timer${frac < 0.25 ? ' timer-low' : ''}`} role="timer" aria-label={`${Math.ceil(left / 1000)} seconds left`}>
      <span className="timer-bar">
        <span className="timer-fill" style={{ width: `${frac * 100}%` }} />
      </span>
      <span className="mono timer-text">{Math.ceil(left / 1000)}s</span>
    </span>
  )
}

function seatCards(view: TableView, p: PlayerView, size: 'sm' | 'md') {
  if (p.seat === view.hero_seat && view.hole_cards) {
    return view.hole_cards.map((c) => <Card key={c} card={c} size={size} />)
  }
  if (p.shown_cards) return p.shown_cards.map((c) => <Card key={c} card={c} size={size} />)
  if (!view.hand_id || p.status === 'folded' || p.status === 'eliminated') return null
  const dealt = view.action_history.filter((e) => e.type === 'HoleDealt' && e.data.seat === p.seat).length
  return Array.from({ length: dealt }, (_, i) => <CardBack key={i} size={size} />)
}

function Markers({ view, seat }: { view: TableView; seat: number }) {
  return (
    <span className="markers">
      {view.button_seat === seat && (
        <span className="marker marker-d" title="Dealer button">
          D
        </span>
      )}
      {view.small_blind_seat === seat && (
        <span className="marker" title="Small blind">
          SB
        </span>
      )}
      {view.big_blind_seat === seat && (
        <span className="marker" title="Big blind">
          BB
        </span>
      )}
    </span>
  )
}

const STATUS_TEXT: Record<string, string> = { folded: 'Folded', all_in: 'All in', eliminated: 'Out' }

function Seat({
  view,
  player,
  info,
  active,
  pos,
  timer,
}: {
  view: TableView
  player: PlayerView
  info?: SeatInfo
  active: boolean
  pos: { x: number; y: number }
  timer: React.ReactNode
}) {
  const hero = player.seat === view.hero_seat
  const classes = ['seat', `seat-st-${player.status}`, active && 'seat-active', hero && 'seat-hero'].filter(Boolean).join(' ')
  return (
    <div
      className={classes}
      style={{ '--x': `${pos.x}%`, '--y': `${pos.y}%` } as React.CSSProperties}
      data-testid={`seat-${player.seat}`}
      aria-current={active ? 'true' : undefined}
    >
      <div className="seat-head">
        <span className="mono seat-id">Seat {player.seat}</span>
        <Markers view={view} seat={player.seat} />
      </div>
      <div className="seat-who">{hero ? 'You' : info?.kind === 'bot' ? `Bot · ${info.label}` : 'Human'}</div>
      <div className="seat-cards">{seatCards(view, player, 'sm')}</div>
      <div className="seat-foot">
        <span className="mono seat-stack">{chips(player.stack)}</span>
        {STATUS_TEXT[player.status] && <span className="seat-status">{STATUS_TEXT[player.status]}</span>}
        {player.street_committed > 0 && <span className="mono seat-bet">bet {chips(player.street_committed)}</span>}
      </div>
      {active && <div className="seat-timer">{timer}</div>}
    </div>
  )
}

function LastHand({ snapshot }: { snapshot: Snapshot }) {
  const last = snapshot.last_hand
  if (!last) return null
  const board = last.events.filter((e) => e.type === 'BoardDealt').flatMap((e) => e.data.cards as string[])
  const lines = describeEvents(
    last.events.filter((e) => ['CardsRevealed', 'PotAwarded', 'PlayerEliminated', 'ExcessReturned'].includes(e.type)),
    snapshot.setup.seats,
  )
  return (
    <div className="last-hand" data-testid="last-hand">
      <div className="label">Hand {last.hand_number} result</div>
      {board.length > 0 && (
        <div className="last-board">
          {board.map((c) => (
            <Card key={c} card={c} size="sm" />
          ))}
        </div>
      )}
      <ul className="log-lines">
        {lines.map((l) => (
          <li key={l.seq}>{l.text}</li>
        ))}
      </ul>
    </div>
  )
}

export function TableScreen({
  snapshot,
  token,
  stream,
  onSnapshot,
}: {
  snapshot: Snapshot
  token: string | null
  stream: StreamStatus
  onSnapshot: (s: Snapshot) => void
}) {
  const view = snapshot.view as TableView
  const seats = view.seats
  const heroIndex = Math.max(0, view.hero_seat !== null ? seats.indexOf(view.hero_seat) : 0)
  const positions = useMemo(() => {
    const n = seats.length
    return Object.fromEntries(
      seats.map((seat, i) => {
        const k = (i - heroIndex + n) % n
        const angle = Math.PI / 2 + (2 * Math.PI * k) / n
        return [seat, { x: 50 + 43 * Math.cos(angle), y: 50 + 39 * Math.sin(angle), bx: 50 + 27 * Math.cos(angle), by: 50 + 23 * Math.sin(angle) }]
      }),
    )
  }, [seats, heroIndex])
  const info = Object.fromEntries(snapshot.setup.seats.map((s) => [s.seat, s]))
  const bets = view.pot_total - view.settled_pot
  const log = describeEvents(view.action_history, snapshot.setup.seats).slice(-14)
  const timer = <Timer snapshot={snapshot} />
  const level = view.blinds ? Math.floor((view.hand_number - 1) / snapshot.setup.config.hands_per_level) + 1 : null

  return (
    <main className="play">
      <div className="hand-strip">
        <span>
          <span className="label">Hand</span> <span className="mono">{view.hand_number}</span>
          <span className="muted mono">/{view.hand_cap}</span>
        </span>
        {view.blinds && (
          <span>
            <span className="label">Level {level}</span>{' '}
            <span className="mono">
              {chips(view.blinds.small_blind)}/{chips(view.blinds.big_blind)}
            </span>
          </span>
        )}
        {view.next_level && (
          <span className="muted">
            next {chips(view.next_level.small_blind)}/{chips(view.next_level.big_blind)} at hand {view.next_level.starts_at_hand}
          </span>
        )}
        <span className="muted">
          {view.surviving_player_count}/{view.initial_player_count} left · {RULESETS[view.ruleset_id]?.short}
        </span>
        <span className={`stream stream-${stream}`}>{stream === 'live' ? 'live' : stream}</span>
      </div>

      <div className="play-grid">
        <section className={`felt felt-${seats.length}`} aria-label="Table">
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

        <aside className="log panel" aria-label="Action log">
          <LastHand snapshot={snapshot} />
          <div className="label">This hand</div>
          <ol className="log-lines">
            {log.map((l) => (
              <li key={l.seq} className={`log-${l.type}`}>
                {l.text}
              </li>
            ))}
          </ol>
        </aside>
      </div>

      <ActionPanel snapshot={snapshot} token={token} onSnapshot={onSnapshot} timer={timer} />
    </main>
  )
}
