import { api, type Credentials } from '../api'
import { RULESETS, chips, placeLabel, points } from '../format'
import type { Snapshot } from '../types'

export function Results({
  snapshot,
  creds,
  navigate,
}: {
  snapshot: Snapshot
  creds: Credentials
  navigate: (path: string) => void
}) {
  const results = snapshot.results!
  const seats = snapshot.setup.seats
  const label = (seat: number) => {
    const s = seats.find((x) => x.seat === seat)
    if (!s) return ''
    if (s.you) return 'You'
    return s.kind === 'bot' ? `Bot · ${s.label}` : 'Human'
  }
  const ordered = [...results.placements].sort((a, b) => a.positions[0] - b.positions[0] || b.points - a.points)
  const totalFallbacks = (seat: number) =>
    Object.values(results.fallbacks[String(seat)] ?? {}).reduce((sum, n) => sum + n, 0)
  const fallbackDetail = (seat: number) =>
    Object.entries(results.fallbacks[String(seat)] ?? {})
      .map(([reason, n]) => `${n} ${reason.replace('_', ' ')}`)
      .join(', ')

  const download = async () => {
    const history = await api.history(snapshot.table_id, creds.seatToken ?? null)
    const blob = new Blob([JSON.stringify(history, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `${snapshot.table_id}-history.json`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  const endedBy =
    results.placements.some((p) => p.method === 'icm')
      ? `Hand cap reached after ${results.hands_played} hands; survivors scored by ICM.`
      : `One survivor after ${results.hands_played} hands.`

  return (
    <main className="results panel" data-testid="results">
      <h1 className="screen-title">Table complete</h1>
      <p className="lede">
        {RULESETS[snapshot.setup.config.ruleset_id]?.name} · {seats.length} seats · {endedBy}
      </p>
      <div className="table-scroll">
        <table className="results-table">
          <thead>
            <tr>
              <th>Place</th>
              <th>Seat</th>
              <th>Player</th>
              <th className="num">Points</th>
              <th>Scored by</th>
              <th className="num">Fallbacks</th>
            </tr>
          </thead>
          <tbody>
            {ordered.map((p) => (
              <tr key={p.seat} className={seats.find((s) => s.seat === p.seat)?.you ? 'row-you' : ''}>
                <td className="mono">{placeLabel(p.positions)}</td>
                <td className="mono">{p.seat}</td>
                <td>{label(p.seat)}</td>
                <td className="mono num">{points(p.points)}</td>
                <td>
                  {p.method === 'icm' ? 'ICM at cap' : p.method === 'winner' ? 'Last survivor' : `Out on hand ${p.hand_number}`}
                </td>
                <td className="mono num" title={fallbackDetail(p.seat)}>
                  {totalFallbacks(p.seat)}
                  {totalFallbacks(p.seat) > 0 && <span className="hint"> ({fallbackDetail(p.seat)})</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <dl className="kv results-kv">
        <dt>Initial payout vector</dt>
        <dd className="mono">[{results.payouts.join(', ')}]</dd>
        <dt>Eliminations</dt>
        <dd>
          {results.eliminations.length === 0
            ? 'None'
            : results.eliminations.map((e) => `Seat ${e.seat} (hand ${e.hand_number})`).join(' · ')}
        </dd>
        <dt>Starting stack</dt>
        <dd className="mono">{chips(snapshot.setup.config.starting_stack)}</dd>
        <dt>Config hash</dt>
        <dd className="mono hash">{snapshot.setup.config_hash.slice(0, 24)}…</dd>
      </dl>
      <div className="results-actions">
        <button className="btn btn-primary" onClick={() => navigate('/')}>
          New table
        </button>
        <a className="btn" href={`#/t/${snapshot.table_id}/replay`}>
          Replay this table
        </a>
        <button className="btn" onClick={download}>
          Download my history
        </button>
      </div>
    </main>
  )
}
