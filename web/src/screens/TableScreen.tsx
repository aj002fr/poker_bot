import { useEffect, useState } from 'react'
import type { StreamStatus } from '../api'
import { Card } from '../cards'
import { RULESETS, chips, describeEvents } from '../format'
import type { Snapshot, TableView } from '../types'
import { ActionPanel } from './ActionPanel'
import { Felt } from './Felt'

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

function LastHand({ snapshot }: { snapshot: Snapshot }) {
  const last = snapshot.last_hand
  if (!last) return null
  const board = last.events.filter((e) => e.type === 'BoardDealt').flatMap((e) => e.data.cards as string[])
  const lines = describeEvents(
    last.events.filter((e) => ['CardsRevealed', 'PotAwarded', 'PlayerEliminated', 'ExcessReturned'].includes(e.type)),
    snapshot.setup.seats,
  )
  return (
    <div className="last-hand" data-testid="last-hand" data-hand-number={last.hand_number ?? undefined}>
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
  connections,
  onSnapshot,
}: {
  snapshot: Snapshot
  token: string | null
  stream: StreamStatus
  connections: number
  onSnapshot: (s: Snapshot) => void
}) {
  const view = snapshot.view as TableView
  const log = describeEvents(view.action_history, snapshot.setup.seats).slice(-14)
  const timer = <Timer snapshot={snapshot} />
  const level = view.blinds ? Math.floor((view.hand_number - 1) / snapshot.setup.config.hands_per_level) + 1 : null

  return (
    <main className="play">
      <div className="hand-strip">
        <span>
          <span className="label">Hand</span>{' '}
          <span className="mono" data-testid="hand-number">
            {view.hand_number}
          </span>
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
        <span className="muted" data-testid="survivors">
          {view.surviving_player_count}/{view.initial_player_count} left · {RULESETS[view.ruleset_id]?.short}
        </span>
        <a className="strip-link" href={`#/t/${snapshot.table_id}/replay`}>
          Replay so far
        </a>
        <span className={`stream stream-${stream}`} data-testid="stream-status" role="status">
          {stream === 'live' ? 'live' : stream}
        </span>
      </div>

      <div className="play-grid">
        <Felt view={view} seats={snapshot.setup.seats} timer={timer} />
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

      <ActionPanel snapshot={snapshot} token={token} onSnapshot={onSnapshot} timer={timer} connections={connections} />
    </main>
  )
}
