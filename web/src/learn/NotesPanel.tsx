import { cardText } from '../cards'
import { chips } from '../format'
import type { SeatInfo, SeatStats, Showdown, Snapshot, StylesCatalog } from '../types'
import { guessesKey, useStored } from './prefs'
import { isHiddenLive, range, rate, visibleProfile } from './styles'
import { Term } from './Term'

export type Guesses = Record<string, { style?: string; note?: string }>

const LINE_WORDS: Record<string, string> = { fold: 'fold', check: 'check', call: 'call', bet: 'bet', raise: 'raise' }

export function lineText(line: Showdown['line']) {
  return (['preflop', 'flop', 'turn', 'river'] as const)
    .filter((st) => line[st]?.length)
    .map((st) => `${st}: ${line[st]!.map((t) => LINE_WORDS[t] ?? t).join(', ')}`)
    .join(' · ')
}

export function ShowdownList({ items }: { items: Showdown[] }) {
  if (items.length === 0) return <p className="hint">No showdowns yet.</p>
  return (
    <ul className="showdowns">
      {items.map((s) => (
        <li key={s.hand_id}>
          <span className="mono">Hand {s.hand_number}</span> · <span className="mono">{s.cards.map(cardText).join(' ')}</span> ·{' '}
          {s.hand_in_words}
          {s.won > 0 ? `, won ${chips(s.won)}` : ''}
          <span className="hint showdown-line">{lineText(s.line)}</span>
        </li>
      ))}
    </ul>
  )
}

function StatRows({ s, expected }: { s: SeatStats; expected?: Record<string, [number, number] | null | undefined> }) {
  const exp = (k: string, ratio = false) => {
    const r = range(expected?.[k], ratio)
    return r ? <span className="stat-exp">usually {r}</span> : null
  }
  return (
    <dl className="stat-rows">
      <dt>
        <Term k="vpip">Hands played</Term>
      </dt>
      <dd>
        <span className="mono">
          {s.vpip.count} of {s.vpip.of}
        </span>{' '}
        ({rate(s.vpip)}) {exp('vpip')}
      </dd>
      <dt>
        <Term k="pfr">Raise first</Term>
      </dt>
      <dd>
        <span className="mono">
          {s.pfr.count} of {s.pfr.of}
        </span>{' '}
        ({rate(s.pfr)}) {exp('pfr')}
      </dd>
      <dt>
        <Term k="aggression">Callers or raisers</Term>
      </dt>
      <dd>
        <span className="mono">
          {s.aggression.bets_raises} bets/raises, {s.aggression.calls} calls
        </span>
        {s.aggression.factor !== null && ` (${s.aggression.factor.toFixed(1)}×)`} {exp('aggression', true)}
      </dd>
      <dt>Reraised</dt>
      <dd>
        {s.reraise.faced === 0 ? (
          'Not tested yet'
        ) : (
          <>
            <span className="mono">
              faced {s.reraise.faced}: folded {s.reraise.folded}, called {s.reraise.called}, raised again {s.reraise.reraised}
            </span>{' '}
            {exp('fold_to_reraise')}
          </>
        )}
      </dd>
    </dl>
  )
}

/** Stats for every opponent bot, and a style guess plus free notes for hidden ones. Guesses stay in this browser. */
export function NotesPanel({ snapshot, catalog }: { snapshot: Snapshot; catalog: StylesCatalog | null }) {
  const [guesses, setGuesses] = useStored<Guesses>(guessesKey(snapshot.table_id), {})
  const stats = snapshot.stats?.seats ?? {}
  const opponents = snapshot.setup.seats.filter((s) => s.kind === 'bot')
  if (opponents.length === 0) return null
  const update = (seat: number, patch: { style?: string; note?: string }) =>
    setGuesses({ ...guesses, [seat]: { ...guesses[seat], ...patch } })
  const hiddenCount = opponents.filter((o) => isHiddenLive(o, snapshot)).length

  return (
    <section className="notes-panel panel" aria-label="Notes and guesses" data-testid="notes-panel">
      <h2 className="label">Notes and guess</h2>
      <p className="hint">
        {hiddenCount > 0
          ? `Watch how each hidden player bets, then pick the style you think they play. You can change your guess any time; the answers appear when the table ends.`
          : 'Every style is labelled. Compare what each player does with what their style usually does.'}{' '}
        Small samples mislead: a few hands prove little (<Term k="sample_size">sample size</Term>).
      </p>
      <ul className="notes-seats">
        {opponents.map((o: SeatInfo) => {
          const s = stats[String(o.seat)]
          const hidden = isHiddenLive(o, snapshot)
          const profile = hidden ? null : visibleProfile(o, catalog)
          const g = guesses[o.seat] ?? {}
          return (
            <li key={o.seat} className="notes-seat" data-testid={`notes-seat-${o.seat}`}>
              <div className="notes-seat-head">
                <span className="mono seat-id">Seat {o.seat}</span>
                <strong>{o.label}</strong>
                {hidden ? <span className="seat-badge tag-hidden">Hidden</span> : profile && <span className="muted">{profile.family}</span>}
              </div>
              {profile && <p className="hint">{profile.description}</p>}
              {s && s.hands > 0 ? (
                <>
                  <p className="hint">
                    Based on <span className="mono">{s.hands}</span> hand{s.hands === 1 ? '' : 's'}
                    {s.hands < 20 ? ', still a small sample' : ''}.
                  </p>
                  <StatRows s={s} expected={profile?.expected} />
                  <details>
                    <summary className="label">Showdowns ({s.showdowns.count})</summary>
                    <ShowdownList items={s.showdowns.items} />
                  </details>
                </>
              ) : (
                <p className="hint">No hands yet.</p>
              )}
              {hidden && (
                <div className="guess">
                  <label className="field-inline">
                    <span className="label">Your guess</span>
                    <select
                      aria-label={`Style guess for seat ${o.seat}`}
                      value={g.style ?? ''}
                      onChange={(e) => update(o.seat, { style: e.target.value || undefined })}
                    >
                      <option value="">Not sure yet</option>
                      {catalog?.styles.map((st) => (
                        <option key={st.id} value={st.id}>
                          {st.name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="field-inline">
                    <span className="label">Notes</span>
                    <textarea
                      aria-label={`Notes on seat ${o.seat}`}
                      rows={2}
                      value={g.note ?? ''}
                      placeholder="e.g. raised twice from early seats, folded to a reraise"
                      onChange={(e) => update(o.seat, { note: e.target.value })}
                    />
                  </label>
                </div>
              )}
            </li>
          )
        })}
      </ul>
      {catalog && hiddenCount > 0 && (
        <details className="style-cheats">
          <summary className="label">What each style tends to do</summary>
          <ul>
            {catalog.styles.map((st) => (
              <li key={st.id}>
                <strong>{st.name}</strong>: {st.description}{' '}
                <span className="hint">
                  Plays {range(st.expected.vpip) ?? '?'} of hands, raises first {range(st.expected.pfr) ?? '?'}.
                </span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  )
}
