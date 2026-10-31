import { useState } from 'react'
import { ApiFailure, api, saveCredentials, type Credentials } from '../api'
import { RULESETS, chips, seatWho } from '../format'
import type { Snapshot } from '../types'
import { formatFailure } from './Setup'

export function Lobby({
  snapshot,
  creds,
  onCreds,
}: {
  snapshot: Snapshot
  creds: Credentials
  onCreds: (c: Credentials) => void
}) {
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const { setup, table_id } = snapshot
  const openSeats = setup.seats.filter((s) => s.kind === 'human' && !s.claimed)
  const link = `${location.origin}${location.pathname}#/t/${table_id}`

  const claim = async (seat: number) => {
    setBusy(true)
    setError(null)
    try {
      const joined = await api.join(table_id, seat)
      const next = { ...creds, seat: joined.seat, seatToken: joined.seat_token }
      saveCredentials(table_id, next)
      onCreds(next)
    } catch (err) {
      setError(err instanceof ApiFailure ? formatFailure(err) : 'Could not reach the server.')
    } finally {
      setBusy(false)
    }
  }

  const start = async () => {
    if (!creds.hostToken) return
    setBusy(true)
    setError(null)
    try {
      await api.start(table_id, creds.hostToken)
    } catch (err) {
      setError(err instanceof ApiFailure ? formatFailure(err) : 'Could not reach the server.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <main className="lobby panel">
      <h1 className="screen-title">Waiting for players</h1>
      <p className="lede">
        {RULESETS[setup.config.ruleset_id]?.name} · {setup.seats.length} seats · {chips(setup.config.starting_stack)}{' '}
        chips each · cap {setup.config.hand_cap} hands
      </p>
      <ul className="lobby-seats">
        {setup.seats.map((s) => (
          <li key={s.seat} className={s.you ? 'lobby-you' : ''}>
            <span className="mono seat-id">Seat {s.seat}</span>
            <span>{s.kind === 'bot' ? seatWho(s) : s.you ? 'You' : s.claimed ? 'Human · joined' : 'Human · open'}</span>
            {s.kind === 'human' && !s.claimed && creds.seat === undefined && (
              <button className="btn btn-small" disabled={busy} onClick={() => claim(s.seat)}>
                Take this seat
              </button>
            )}
            {s.kind === 'human' && !s.claimed && creds.seat !== undefined && (
              <a className="btn btn-small" href={`#/t/${table_id}/join/${s.seat}`} target="_blank" rel="noreferrer">
                Open in new tab
              </a>
            )}
          </li>
        ))}
      </ul>
      {openSeats.length > 0 && (
        <div className="invite">
          <span className="label">Table link</span>
          <code className="mono invite-link">{link}</code>
          <p className="hint">The link carries no credential. Each tab that claims a seat gets its own private seat key.</p>
        </div>
      )}
      {error && <p className="error">{error}</p>}
      {creds.hostToken ? (
        <button className="btn btn-primary" disabled={busy || openSeats.length > 0} onClick={start}>
          {openSeats.length > 0 ? `Waiting for ${openSeats.length} human seat${openSeats.length > 1 ? 's' : ''}` : 'Start table'}
        </button>
      ) : (
        <p className="hint">The host starts the table once every human seat is taken.</p>
      )}
    </main>
  )
}
