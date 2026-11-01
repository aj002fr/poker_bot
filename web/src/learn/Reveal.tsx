import { useEffect } from 'react'
import type { Snapshot, StylesCatalog } from '../types'
import type { Guesses } from './NotesPanel'
import { ShowdownList } from './NotesPanel'
import { guessesKey, useStored } from './prefs'
import { recordFinish } from './progress'
import { range } from './styles'

const value = (stat: string, v: number | null) => (v === null ? 'no sample' : stat === 'aggression' ? `${v.toFixed(1)}×` : `${Math.round(v * 100)}%`)

/** The end-of-table reveal: each seat's true style next to your guess, with the evidence that pointed to it. */
export function Reveal({ snapshot, catalog }: { snapshot: Snapshot; catalog: StylesCatalog | null }) {
  const reveal = snapshot.reveal
  const [guesses] = useStored<Guesses>(guessesKey(snapshot.table_id), {})
  const bots = reveal?.seats.filter((s) => s.kind === 'bot') ?? []
  const hidden = bots.filter((s) => s.hidden)
  const correct = hidden.filter((s) => s.style && guesses[s.seat]?.style === s.style).length
  const you = snapshot.setup.seats.find((s) => s.you)?.seat
  const placements = snapshot.results?.placements ?? []
  const place = (seat: number) => placements.find((p) => p.seat === seat)?.positions[0] ?? Infinity
  const beaten =
    you === undefined ? [] : bots.filter((b) => b.style && place(you) < place(b.seat)).map((b) => b.style as string)
  const score = hidden.length > 0 ? `${correct}/${hidden.length}` : undefined

  useEffect(() => {
    if (reveal) recordFinish(snapshot.table_id, beaten, score)
  }, [reveal?.lineup_hash, score])

  if (!reveal || bots.length === 0) return null
  const styleName = (id: string | undefined | null) => catalog?.styles.find((s) => s.id === id)?.name ?? id ?? 'No guess'

  return (
    <section className="reveal" data-testid="reveal" aria-labelledby="reveal-title">
      <h2 id="reveal-title" className="screen-title">
        Reveal
      </h2>
      {hidden.length > 0 ? (
        <p className="reveal-score" data-testid="reveal-score">
          You read <strong>{correct}</strong> of <strong>{hidden.length}</strong> hidden player{hidden.length === 1 ? '' : 's'} correctly.
          {correct === hidden.length ? ' Sharp reading.' : ' Check the evidence below to see what gave each one away.'}
        </p>
      ) : (
        <p className="lede">Every style was labelled. Here is how each one actually played against its usual pattern.</p>
      )}
      <ul className="reveal-seats">
        {bots.map((s) => {
          const guess = guesses[s.seat]?.style
          const right = s.hidden && guess === s.style
          return (
            <li key={s.seat} className={`reveal-seat${s.hidden ? (right ? ' reveal-right' : ' reveal-wrong') : ''}`} data-testid={`reveal-seat-${s.seat}`}>
              <div className="reveal-head">
                <span className="mono seat-id">Seat {s.seat}</span>
                <strong className="reveal-style">{s.label}</strong>
                {s.random_pick && <span className="muted">picked at random</span>}
                {s.hidden ? (
                  <span className={`seat-badge ${right ? 'seat-badge-won' : 'tag-miss'}`}>
                    {right ? 'Correct' : guess ? `You guessed ${styleName(guess)}` : 'No guess'}
                  </span>
                ) : (
                  <span className="muted">labelled</span>
                )}
              </div>
              {s.summary && <p>{s.summary}</p>}
              {s.book_ref && (
                <p className="hint">
                  Style: {s.book_ref.part}, “{s.book_ref.section}”, <cite>Harrington on Hold'em, Vol. I</cite>.
                </p>
              )}
              {s.evidence.length > 0 && (
                <div className="table-scroll">
                  <table className="evidence">
                    <thead>
                      <tr>
                        <th>Question</th>
                        <th>What they did</th>
                        <th>Usual for this style</th>
                        <th>Fits?</th>
                      </tr>
                    </thead>
                    <tbody>
                      {s.evidence.map((e) => (
                        <tr key={e.stat}>
                          <td>
                            {e.label}
                            {e.book_ref && <span className="hint evidence-ref">{e.book_ref.part}: {e.book_ref.section}</span>}
                          </td>
                          <td className="mono">
                            {value(e.stat, e.value)}
                            {e.of !== undefined && e.of > 0 && <span className="hint"> ({e.count} of {e.of})</span>}
                          </td>
                          <td className="mono">{range(e.expected, e.stat === 'aggression') ?? '–'}</td>
                          <td>{e.in_range === null ? '–' : e.in_range ? 'Yes' : 'No'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {s.showdowns.length > 0 && (
                <details>
                  <summary className="label">Showdowns ({s.showdowns.length})</summary>
                  <ShowdownList items={s.showdowns} />
                </details>
              )}
            </li>
          )
        })}
      </ul>
      <p className="hint mono">Lineup hash {reveal.lineup_hash.slice(0, 16)}… covers every seat's policy, so the table is reproducible.</p>
    </section>
  )
}
