import { useState } from 'react'
import { ApiFailure, api, createAndJoin, loadCredentials, useStyles, type CreateBody } from '../api'
import { GLOSSARY } from '../learn/glossary'
import { RANKINGS } from '../learn/HandRankings'
import { rememberLearnTable, useProgress } from '../learn/progress'
import { totals, useStudyScore } from '../learn/studyScore'
import { range } from '../learn/styles'
import type { StyleMode } from '../types'
import { formatFailure } from './Setup'

interface Step {
  id: string
  title: string
  goal: string
  watch: string
  book: string
  mode: StyleMode
  /** Bot seats after your own seat 0: a style policy id, optionally hidden in mixed mode. */
  bots: { policy: string; hide?: boolean }[]
}

export const STEPS: Step[] = [
  {
    id: 'value-betting',
    title: 'Value bet a calling station',
    goal: 'Heads-up against a player who calls almost everything. Bluffs do not work here, so bet your good hands and let them pay you.',
    watch: 'When you hold a strong hand, bet. When you have nothing, check and give up.',
    book: 'Part Two (playing styles) and the idea of betting for value',
    mode: 'labelled',
    bots: [{ policy: 'style_station' }],
  },
  {
    id: 'position-charts',
    title: 'Position and opening charts',
    goal: 'Three seats with a Rock and an Aggressive player. See how differently they open from the same seats.',
    watch: 'Open the starting-hand chart each hand. The Rock waits for big hands; the Aggressive player raises far more often.',
    book: 'Part Two: opening requirements by position',
    mode: 'labelled',
    bots: [{ policy: 'style_rock' }, { policy: 'style_aggressive' }],
  },
  {
    id: 'spot-styles',
    title: 'Spot the styles from their play',
    goal: 'Six seats, every style labelled. Match each player’s stats badge to what their style usually does.',
    watch: 'Compare “played” and “raised” on each seat with the usual ranges in the Notes panel.',
    book: 'Part Three: reading the table from betting patterns',
    mode: 'labelled',
    bots: [{ policy: 'style_rock' }, { policy: 'style_aggressive' }, { policy: 'style_super_aggressive' }, { policy: 'style_station' }, { policy: 'style_weak_tight' }],
  },
  {
    id: 'two-hidden',
    title: 'Two styles hidden',
    goal: 'Six seats, mixed: three styles labelled, two hidden. Work out the hidden two and enter your guesses.',
    watch: 'How often does each hidden player enter pots? Do they call or raise? What do they do when someone reraises?',
    book: 'Part Three: the four questions to ask about every opponent',
    mode: 'mixed',
    bots: [
      { policy: 'style_rock' },
      { policy: 'style_aggressive', hide: true },
      { policy: 'style_station' },
      { policy: 'style_weak_tight', hide: true },
      { policy: 'style_super_aggressive' },
    ],
  },
  {
    id: 'full-table',
    title: 'Full table, all hidden',
    goal: 'Eight seats, the competition size. Every style is secret and picked at random. Read the whole table.',
    watch: 'Keep notes as you go and do not trust a stat built on a handful of hands.',
    book: 'Part Three: table reading at a full table',
    mode: 'random',
    bots: Array.from({ length: 7 }, () => ({ policy: 'style_random' })),
  },
]

export function stepBody(step: Step): CreateBody {
  return {
    config: {
      ruleset_id: 'practice_nlhe',
      initial_player_count: step.bots.length + 1,
      starting_stack: 2000,
      hand_cap: 30,
      human_turn_ms: 60_000,
    },
    seats: [
      { kind: 'human' },
      ...step.bots.map((b) => ({ kind: 'bot' as const, policy: b.policy, ...(step.mode === 'mixed' ? { reveal: !b.hide } : {}) })),
    ],
    styles: { mode: step.mode, learning_aids: true },
  }
}

export function LearnScreen({ navigate }: { navigate: (p: string) => void }) {
  const catalog = useStyles()
  const [progress] = useProgress()
  const [score] = useStudyScore()
  const studied = totals(score)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const done = STEPS.filter((s) => progress.steps[s.id]?.done).length
  const next = STEPS.find((s) => !progress.steps[s.id]?.done)

  const start = async (step: Step) => {
    setBusy(step.id)
    setError(null)
    try {
      const tableId = await createAndJoin(stepBody(step))
      await api.start(tableId, loadCredentials(tableId).hostToken!)
      rememberLearnTable(tableId, step.id)
      navigate(`/t/${tableId}`)
    } catch (err) {
      setError(err instanceof ApiFailure ? formatFailure(err) : 'Could not reach the server. Is `make dev` running?')
      setBusy(null)
    }
  }

  return (
    <main className="learn" data-testid="learn">
      <section className="panel learn-intro">
        <h1 className="screen-title">Learn to read the table</h1>
        <p className="lede">
          Five short tables, each with one thing to practise. They follow two parts of <cite>Harrington on Hold'em, Vol. I</cite>: Part
          Two on playing styles and which hands to open, and Part Three on reading opponents from how they bet. All advice here is our
          own summary; read the book for the full picture.
        </p>
        <p className="learn-progress" data-testid="learn-progress">
          <strong>{done}</strong> of {STEPS.length} steps done
          {next ? ` · next up: ${next.title}` : ' · path complete'}
        </p>
        <div className="progress-bar" aria-hidden="true">
          <span style={{ width: `${(done / STEPS.length) * 100}%` }} />
        </div>
      </section>

      {error && <p className="error">{error}</p>}

      <section className="panel learn-step pstudy-card" data-testid="learn-study">
        <div className="step-head">
          <h2 className="step-title">Study mode: predict Pluribus</h2>
          {studied.decisions > 0 && (
            <span className="seat-badge seat-badge-won">
              {studied.action} of {studied.decisions} matched
            </span>
          )}
        </div>
        <p>
          Replay real six-player hands played by Pluribus, the AI that beat top professionals. The hand pauses at each of its decisions: you
          choose an action and a size, then see what Pluribus did, with the pot odds and equity for that spot.
        </p>
        <div className="step-actions">
          <a className="btn btn-primary" href="#/learn/pluribus">
            {studied.decisions > 0 ? 'Keep studying' : 'Start studying'}
          </a>
        </div>
      </section>

      <ol className="learn-steps">
        {STEPS.map((step, i) => {
          const p = progress.steps[step.id]
          return (
            <li key={step.id} className={`panel learn-step${p?.done ? ' step-done' : ''}${next?.id === step.id ? ' step-next' : ''}`} data-testid={`learn-step-${i + 1}`}>
              <div className="step-head">
                <span className="step-num mono">{i + 1}</span>
                <h2 className="step-title">{step.title}</h2>
                {p?.done ? (
                  <span className="seat-badge seat-badge-won">Done{p.score ? ` · read ${p.score}` : ''}</span>
                ) : p ? (
                  <span className="seat-badge tag-hidden">Started</span>
                ) : null}
              </div>
              <p>{step.goal}</p>
              <p className="hint">
                <strong>Watch for:</strong> {step.watch}
              </p>
              <p className="hint">
                <strong>Book:</strong> {step.book}.
              </p>
              <p className="hint mono">
                {step.bots.length + 1} seats · {step.mode === 'random' ? 'random hidden styles' : step.mode} · 30 hands · 60 s turns
              </p>
              <div className="step-actions">
                <button className="btn btn-primary" disabled={busy !== null} onClick={() => start(step)}>
                  {busy === step.id ? 'Setting up…' : p ? 'Play again' : 'Start this table'}
                </button>
                {p && !p.done && (
                  <a className="btn" href={`#/t/${p.tableId}`}>
                    Resume
                  </a>
                )}
              </div>
            </li>
          )
        })}
      </ol>

      <section className="panel learn-styles" aria-labelledby="styles-title">
        <h2 id="styles-title" className="screen-title">
          The opponent styles
        </h2>
        {!catalog ? (
          <p className="muted" aria-busy="true">
            Loading the style profiles…
          </p>
        ) : (
          <ul className="style-cards">
            {catalog.styles.map((st) => (
              <li key={st.id} className="style-card" data-testid={`style-card-${st.id}`}>
                <div className="step-head">
                  <h3 className="step-title">{st.name}</h3>
                  <span className="muted">{st.family}</span>
                  {st.advanced && <span className="seat-badge tag-hidden">Advanced</span>}
                </div>
                <p>{st.description}</p>
                <p className="hint">
                  Usually plays {range(st.expected.vpip) ?? '?'} of hands and raises first with {range(st.expected.pfr) ?? '?'}.
                </p>
                {st.tells.length > 0 && (
                  <ul className="tells">
                    {st.tells.map((t) => (
                      <li key={t}>{t}</li>
                    ))}
                  </ul>
                )}
                <p>
                  <strong>How to play against it:</strong> {st.counter_strategy}
                </p>
                <p className="hint">
                  {st.book_ref.part}, “{st.book_ref.section}”.
                </p>
              </li>
            ))}
          </ul>
        )}
      </section>

      <div className="learn-ref">
        <section className="panel" aria-labelledby="rankings-ref">
          <h2 id="rankings-ref" className="screen-title">
            Hand rankings
          </h2>
          <ol className="rankings-list">
            {RANKINGS.map((r) => (
              <li key={r.id}>
                <span className="ranking-name">{r.name}</span>
                <span className="mono ranking-example">{r.example}</span>
                <span className="ranking-text">{r.text}</span>
              </li>
            ))}
          </ol>
        </section>
        <section className="panel" aria-labelledby="glossary-title">
          <h2 id="glossary-title" className="screen-title">
            Glossary
          </h2>
          <dl className="glossary" data-testid="glossary">
            {Object.entries(GLOSSARY).map(([k, g]) => (
              <div key={k}>
                <dt>{g.term}</dt>
                <dd>{g.text}</dd>
              </div>
            ))}
          </dl>
        </section>
      </div>
    </main>
  )
}
