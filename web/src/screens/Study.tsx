import { useCallback, useEffect, useMemo, useState } from 'react'
import { ApiFailure, api } from '../api'
import { Card } from '../cards'
import { chips, describeEvents } from '../format'
import { POSITION_NAMES } from '../learn/positions'
import { judge, recordAnswer, resetStudyScore, tally, totals, useStudyScore, type StudyAnswer } from '../learn/studyScore'
import type {
  StudyDecision,
  StudyFilter,
  StudyFrame,
  StudyHand,
  StudyHandSummary,
  StudyOption,
  StudyPack,
  TableView,
} from '../types'
import { Felt } from './Felt'

const FILTERS: { id: StudyFilter; label: string }[] = [
  { id: 'all', label: 'All' },
  { id: 'postflop', label: 'Saw flop' },
  { id: 'showdown', label: 'Showdown' },
  { id: 'big_pot', label: 'Big pots' },
]
const PAGE = 40
const STEP_MS = 900

const pct = (x: number) => `${Math.round(100 * x)}%`
const bb = (n: number) => `${Number.isInteger(n) ? n : n.toFixed(1)} BB`

function failureText(err: unknown): string {
  if (!(err instanceof ApiFailure)) return 'Could not reach the server. Is `make dev` running?'
  if (err.code === 'pack_unreadable') return `${err.message}. Rebuild it with \`make study-sample\` or \`make study-pack\`.`
  if (err.status === 404) return 'That study hand is not in this pack any more. Pick another from the list.'
  return `${err.code}: ${err.message}`
}

function optionLabel(o: StudyOption) {
  if (o.kind === 'fold') return 'Fold'
  if (o.kind === 'check') return 'Check'
  if (o.kind === 'call') return `Call ${chips(o.amount ?? 0)}`
  if (o.size === 'all-in') return `All in ${chips(o.amount ?? 0)}`
  return `${o.bet ? 'Bet' : 'Raise to'} ${chips(o.amount ?? 0)}`
}

function chose(d: StudyDecision) {
  const p = d.pluribus
  const bet = d.options.some((o) => o.kind === 'raise' && o.bet)
  if (p.kind === 'fold') return 'folded'
  if (p.kind === 'check') return 'checked'
  if (p.kind === 'call') return `called ${chips(p.amount ?? 0)}`
  return `${bet ? 'bet' : 'raised to'} ${chips(p.amount ?? 0)}`
}

/** A TableView for the felt, seen from Pluribus's seat: its own cards, and others' only once shown. */
function viewOf(hand: StudyHand, f: StudyFrame): TableView {
  const seats = hand.seats.map((s) => s.seat)
  return {
    schema_version: 1,
    perspective: 'seat',
    compatibility_status: 'practice_assumptions',
    table_id: hand.id,
    ruleset_id: hand.ruleset_id,
    config_hash: '',
    state_version: f.seq,
    hero_seat: hand.hero_seat,
    hole_cards: hand.hole_cards,
    seats,
    initial_player_count: seats.length,
    surviving_player_count: seats.length,
    hand_id: hand.id,
    hand_number: 1,
    decision_id: null,
    button_seat: hand.button_seat,
    small_blind_seat: hand.small_blind_seat,
    big_blind_seat: hand.big_blind_seat,
    street: f.street,
    actor_seat: f.actor_seat,
    board: f.board,
    pot_total: f.pot_total,
    settled_pot: f.settled_pot,
    current_bet_to: f.current_bet_to,
    pots: f.pots,
    players: f.players.map((p) => ({ ...p, shown_cards: p.shown_cards ?? null })),
    blinds: hand.blinds,
    next_level: null,
    payouts: [],
    hands_played: 0,
    hands_remaining: 0,
    hand_cap: 1,
    decision_budget: { bot_budget_ns: 0, human_turn_ms: 0 },
    legal: null,
    action_history: seats.flatMap((seat, i) =>
      [0, 1].map((k) => ({ seq: -(2 * i + k + 1), hand_id: hand.id, type: 'HoleDealt', data: { seat } })),
    ),
    table_complete: false,
    points: null,
  }
}

function SpotPanel({ hand, decision }: { hand: StudyHand; decision: StudyDecision }) {
  const s = decision.spot
  const preflop = decision.street === 'preflop'
  const others = s.opponents === 1 ? '1 random hand' : `${s.opponents} random hands`
  const needs = s.to_call > 0 ? s.pot_odds : null
  return (
    <section className="panel pstudy-spot" aria-labelledby="pstudy-spot-title" data-testid="study-spot">
      <h2 id="pstudy-spot-title" className="pstudy-h2">
        The spot
      </h2>
      <div className="pstudy-hand">
        {hand.hole_cards.map((c) => (
          <Card key={c} card={c} size="sm" />
        ))}
        <span>
          <strong className="mono">{s.hand}</strong>
          {preflop && <span className="muted"> · top {pct(Math.max(0.01, 1 - s.preflop_percentile))} of starting hands</span>}
        </span>
      </div>
      <dl className="pstudy-facts">
        <div>
          <dt>Position</dt>
          <dd>
            {POSITION_NAMES[s.position]} ({s.position}) · {s.in_position ? 'in position: acts after the others' : 'out of position: someone acts after you'}
          </dd>
        </div>
        <div>
          <dt>Situation</dt>
          <dd>
            {s.facing[0].toUpperCase() + s.facing.slice(1)} · {s.opponents} opponent{s.opponents === 1 ? '' : 's'} in the hand
            {s.behind > 0 ? `, ${s.behind} still to act behind` : ''}
          </dd>
        </div>
        <div>
          <dt>Pot</dt>
          <dd className="mono">
            {chips(s.pot)} ({bb(s.pot_bb)}){s.to_call > 0 ? ` · ${chips(s.to_call)} to call` : ' · nothing to call'}
          </dd>
        </div>
        <div>
          <dt>Stacks</dt>
          <dd>
            <span className="mono">{bb(s.effective_bb)}</span> effective
            {!preflop && (
              <>
                {' '}
                · stack-to-pot ratio <span className="mono">{s.spr.toFixed(1)}</span>
              </>
            )}
          </dd>
        </div>
        {!preflop && (
          <div>
            <dt>Your hand</dt>
            <dd>
              {s.made ? s.made[0].toUpperCase() + s.made.slice(1) : 'No made hand'}
              {s.draws.length > 0 && ` · ${s.draws.join(', ')} (${s.outs} outs)`}
            </dd>
          </div>
        )}
        {!preflop && s.board.length > 0 && (
          <div>
            <dt>Board</dt>
            <dd>{s.board.join(', ')}</dd>
          </div>
        )}
      </dl>
      <div className="pstudy-odds" data-testid="study-odds">
        <div className="pstudy-meter" aria-hidden="true">
          <span className="pstudy-meter-fill" style={{ width: pct(s.equity) }} />
          {needs !== null && <span className="pstudy-meter-mark" style={{ left: pct(needs) }} />}
        </div>
        <p>
          <strong>Equity ≈ {pct(s.equity)}</strong> against {others}.
          {needs !== null ? (
            <>
              {' '}
              Pot odds: calling {chips(s.to_call)} to win {chips(s.pot)} needs <strong>{pct(needs)}</strong>.
            </>
          ) : (
            ' Checking is free here, so pot odds do not apply.'
          )}
        </p>
        <p className="hint">
          Real opponents only continue with decent hands, so true equity is usually lower than this. Equity is your share of the pot if all
          the cards were dealt out now; pot odds are the share you must win for a call to break even.
        </p>
      </div>
    </section>
  )
}

function Prompt({
  decision,
  onPick,
}: {
  decision: StudyDecision
  onPick: (o: StudyOption) => void
}) {
  const basic = decision.options.filter((o) => o.kind !== 'raise')
  const raises = decision.options.filter((o) => o.kind === 'raise')
  return (
    <section className="panel pstudy-prompt" aria-labelledby="pstudy-q" data-testid="study-prompt">
      <h2 id="pstudy-q" className="pstudy-h2">
        Pluribus to act. What would you do?
      </h2>
      <div className="pstudy-options">
        {basic.map((o) => (
          <button key={o.key} className="btn pstudy-opt" onClick={() => onPick(o)} data-testid={`study-opt-${o.key}`}>
            {optionLabel(o)}
          </button>
        ))}
      </div>
      {raises.length > 0 && (
        <>
          <p className="label pstudy-sub">{raises[0].bet ? 'Bet' : 'Raise'} size</p>
          <div className="pstudy-options pstudy-raises">
            {raises.map((o) => (
              <button key={o.key} className="btn pstudy-opt" onClick={() => onPick(o)} data-testid={`study-opt-${o.key}`}>
                <span>{optionLabel(o)}</span>
                <span className="pstudy-size">{o.size}</span>
              </button>
            ))}
          </div>
        </>
      )}
    </section>
  )
}

function Reveal({ decision, answer, onContinue, last }: { decision: StudyDecision; answer: StudyAnswer; onContinue: () => void; last: boolean }) {
  const pick = decision.options.find((o) => o.key === answer.pick)
  const p = decision.pluribus
  return (
    <section className={`panel pstudy-reveal${answer.action ? ' pstudy-match' : ''}`} aria-live="polite" data-testid="study-reveal">
      <p className="pstudy-verdict">
        <span className={`seat-badge ${answer.action ? 'seat-badge-won' : 'tag-hidden'}`}>
          {answer.action ? (answer.size === false ? 'Same action, other size' : 'Matched Pluribus') : 'Different from Pluribus'}
        </span>
      </p>
      <p>
        <strong>Pluribus {chose(decision)}</strong>
        {p.kind === 'raise' && p.size && (
          <span className="muted">
            {' '}
            · {p.exact ? p.size : `nearest menu size: ${p.size}`}
          </span>
        )}
        .{pick && <> You picked {optionLabel(pick).toLowerCase()}{pick.size ? ` (${pick.size})` : ''}.</>}
      </p>
      <p className="hint">
        Pluribus mixes its choices: in the same spot it sometimes takes another line. One hand is a single sample, not a verdict on your play.
      </p>
      <button className="btn btn-primary" onClick={onContinue} data-testid="study-continue" autoFocus>
        {last ? 'See how the hand ends' : 'Continue the hand'}
      </button>
    </section>
  )
}

function HandEnd({
  hand,
  answers,
  onNext,
  onRandom,
  onAgain,
  hasNext,
}: {
  hand: StudyHand
  answers: StudyAnswer[]
  onNext: () => void
  onRandom: () => void
  onAgain: () => void
  hasNext: boolean
}) {
  const net = hand.result.net[String(hand.hero_seat)] ?? 0
  const t = tally(answers)
  const hero = hand.hero_seat
  const rivals = hand.result.showdown.filter((s) => s !== hero).map((s) => hand.seats[s]?.name)
  const where = hand.result.showdown.includes(hero)
    ? ` at showdown against ${rivals.join(' and ')}`
    : rivals.length > 1
      ? `; ${rivals.join(' and ')} went on to a showdown`
      : ''
  return (
    <section className="panel pstudy-end" data-testid="study-end">
      <h2 className="pstudy-h2">Hand complete</h2>
      <p>
        Pluribus {net > 0 ? 'won ' : net < 0 ? 'lost ' : 'broke even'}
        {net !== 0 && <strong className="mono">{bb(Math.abs(net) / hand.blinds.big_blind)}</strong>}
        {where}. Every player’s cards are face up now.
      </p>
      {t.decisions > 0 && (
        <p>
          You matched <strong>{t.action}</strong> of {t.decisions} Pluribus decision{t.decisions === 1 ? '' : 's'}
          {t.sized > 0 && <>, and the size on {t.size} of {t.sized} raises you both made</>}.
        </p>
      )}
      {hand.result.notes.length > 0 && <p className="hint">{hand.result.notes.join(' · ')}</p>}
      <div className="step-actions">
        <button className="btn btn-primary" onClick={onNext} disabled={!hasNext} data-testid="study-next">
          Next hand
        </button>
        <button className="btn" onClick={onRandom}>
          Random hand
        </button>
        <button className="btn" onClick={onAgain}>
          Study this hand again
        </button>
      </div>
    </section>
  )
}

function Message({ title, children, busy }: { title: string; children?: React.ReactNode; busy?: boolean }) {
  return (
    <main className="panel narrow" aria-busy={busy || undefined} data-testid="study-message">
      <h1 className="screen-title">{title}</h1>
      {children}
    </main>
  )
}

export function StudyScreen({ route, navigate }: { route: string; navigate: (p: string) => void }) {
  const [, , , routePack, routeHand] = route.split('/')
  const [packs, setPacks] = useState<StudyPack[] | null>(null)
  const [packsError, setPacksError] = useState<string | null>(null)
  const [retry, setRetry] = useState(0)
  const [filter, setFilter] = useState<StudyFilter>('all')
  const [list, setList] = useState<{ key: string; total: number; hands: StudyHandSummary[] } | null>(null)
  const [listError, setListError] = useState<string | null>(null)
  const [listBusy, setListBusy] = useState(false)
  const [hand, setHand] = useState<StudyHand | null>(null)
  const [handError, setHandError] = useState<string | null>(null)
  const [frame, setFrame] = useState(0)
  const [picks, setPicks] = useState<Record<number, StudyAnswer>>({})
  const [playing, setPlaying] = useState(false)
  const [score] = useStudyScore()

  const pack = packs?.find((p) => p.id === routePack) ?? packs?.[0] ?? null
  const packId = pack?.id
  const listKey = packId ? `${packId}|${filter}` : ''
  const firstUnplayed = list?.key === listKey ? (list.hands.find((h) => !score.hands[h.id]) ?? list.hands[0])?.id : undefined
  // Answering records the hand in the score, which changes `firstUnplayed`; the hand on screen must not follow it.
  const [autoPick, setAutoPick] = useState<{ key: string; id: string } | null>(null)
  useEffect(() => {
    if (firstUnplayed && autoPick?.key !== listKey) setAutoPick({ key: listKey, id: firstUnplayed })
  }, [firstUnplayed, listKey, autoPick])
  const autoHand = autoPick?.key === listKey ? autoPick.id : undefined
  const handId = (routePack === pack?.id ? routeHand : undefined) ?? autoHand

  useEffect(() => {
    let live = true
    setPacksError(null)
    api.studyPacks().then(
      (r) => live && setPacks(r.packs),
      (err) => live && setPacksError(failureText(err)),
    )
    return () => {
      live = false
    }
  }, [retry])

  const loadPage = useCallback(
    (offset: number) => {
      if (!packId) return
      const key = `${packId}|${filter}`
      setListBusy(true)
      setListError(null)
      api.studyHands(packId, filter, offset, PAGE).then(
        (page) => {
          setList((prev) => ({
            key,
            total: page.total,
            hands: offset > 0 && prev?.key === key ? [...prev.hands, ...page.hands] : page.hands,
          }))
          setListBusy(false)
        },
        (err) => {
          setListError(failureText(err))
          setListBusy(false)
        },
      )
    },
    [packId, filter],
  )

  useEffect(() => loadPage(0), [loadPage])

  useEffect(() => {
    if (!packId || !handId) return
    let live = true
    setHand(null)
    setHandError(null)
    api.studyHand(packId, handId).then(
      (h) => {
        if (!live) return
        setHand(h)
        setFrame(0)
        setPicks({})
        setPlaying(true)
      },
      (err) => live && setHandError(failureText(err)),
    )
    return () => {
      live = false
    }
  }, [packId, handId, retry])

  const atDecision = useMemo(() => {
    const m = new Map<number, number>()
    hand?.decisions.forEach((d, i) => m.set(d.frame, i))
    return m
  }, [hand])

  const last = hand ? hand.frames.length - 1 : 0
  const shown = hand?.frames[Math.min(frame, last)]
  const di = atDecision.get(frame)
  const decision = di !== undefined ? hand?.decisions[di] : undefined
  const answer = di !== undefined ? picks[di] : undefined
  const blocked = decision !== undefined && answer === undefined
  const ended = shown?.kind === 'hand_end'

  useEffect(() => {
    if (!playing || !hand) return
    if (blocked || frame >= last) {
      setPlaying(false)
      return
    }
    const t = window.setTimeout(() => setFrame((f) => Math.min(f + 1, last)), STEP_MS)
    return () => window.clearTimeout(t)
  }, [playing, hand, frame, blocked, last])

  const select = (id: string) => pack && navigate(`/learn/pluribus/${pack.id}/${id}`)
  const listed = list?.key === listKey ? list : null
  const index = listed && hand ? listed.hands.findIndex((h) => h.id === hand.id) : -1
  const nextInList = listed && index >= 0 ? listed.hands[index + 1] : listed?.hands[0]
  const random = async () => {
    if (!pack || !listed?.total) return
    try {
      const page = await api.studyHands(pack.id, filter, Math.floor(Math.random() * listed.total), 1)
      if (page.hands[0]) select(page.hands[0].id)
    } catch (err) {
      setHandError(failureText(err))
    }
  }
  const pick = (o: StudyOption) => {
    if (!hand || !decision || di === undefined) return
    const a = judge(decision, o)
    setPicks((p) => ({ ...p, [di]: a }))
    recordAnswer(hand.id, di, a)
  }
  const nextDecisionFrame = () => {
    for (let f = frame + 1; f <= last; f++) {
      const i = atDecision.get(f)
      if (i !== undefined && picks[i] === undefined) return f
    }
    return last
  }
  const restart = () => {
    setFrame(0)
    setPicks({})
    setPlaying(true)
  }

  const names = useMemo(() => Object.fromEntries((hand?.seats ?? []).map((s) => [s.seat, s.name])), [hand])
  const seatLabels = useMemo(() => Object.fromEntries((hand?.seats ?? []).map((s) => [String(s.seat), { label: s.position }])), [hand])
  const lines = useMemo(() => {
    if (!hand || !shown) return []
    const who = (seat: number) => `${hand.seats[seat]?.name ?? `Seat ${seat}`} (${hand.seats[seat]?.position ?? '?'})`
    return describeEvents(
      hand.events.filter((e) => e.seq <= shown.seq),
      undefined,
      who,
    )
  }, [hand, shown])
  const winners = useMemo(() => {
    if (!hand || !ended) return undefined
    const out: Record<number, number> = {}
    for (const e of hand.events) {
      if (e.type !== 'PotAwarded') continue
      for (const [s, v] of Object.entries(e.data.awards as Record<string, number>)) out[Number(s)] = (out[Number(s)] ?? 0) + v
    }
    return out
  }, [hand, ended])

  const all = totals(score)

  if (packsError && !packs) {
    return (
      <Message title="Study mode is unavailable">
        <p className="error">{packsError}</p>
        <button className="btn btn-primary" onClick={() => setRetry((r) => r + 1)}>
          Try again
        </button>
      </Message>
    )
  }
  if (!packs) {
    return (
      <Message title="Study Pluribus" busy>
        <p className="muted">Loading the study hands…</p>
      </Message>
    )
  }
  if (!pack) {
    return (
      <Message title="No study hands yet">
        <p className="lede">
          Study mode replays hands played by Pluribus, the poker AI from Carnegie Mellon and Facebook AI, and pauses at each of its decisions
          so you can guess first. The hands are built from a public dataset and are not stored in this repository.
        </p>
        <ul className="pstudy-steps">
          <li>
            <code>make dev</code> builds 24 sample hands when it starts. To build them yourself, run <code>make study-sample</code>.
          </li>
          <li>
            For all 10,000 hands, run <code>make study-pack</code>. It downloads the dataset once into a cache outside the repository
            and builds the pack.
          </li>
        </ul>
        <button className="btn btn-primary" onClick={() => setRetry((r) => r + 1)}>
          Check again
        </button>
      </Message>
    )
  }

  const answersHere = Object.values(picks)
  const view = hand && shown ? viewOf(hand, shown) : null
  const lastDecision = hand ? di === hand.decisions.length - 1 : false

  return (
    <main className="pstudy" data-testid="study">
      <section className="pstudy-head">
        <div>
          <p className="label">
            <a href="#/learn">Learn</a> / Study Pluribus
          </p>
          <h1 className="screen-title">Predict what Pluribus does</h1>
          <p className="lede">
            Replays of real hands played by Pluribus, the six-player poker AI that beat top professionals in 2019. The hand stops each time
            Pluribus has to act: pick your move and size, then see what it chose. You sit in Pluribus’s seat, so you see its cards and nobody
            else’s until the end.
          </p>
        </div>
        <div className="pstudy-score" data-testid="study-score">
          <span className="label">Your agreement</span>
          <span className="pstudy-score-big mono">{all.decisions ? pct(all.action / all.decisions) : '–'}</span>
          <span className="muted">
            {all.action} of {all.decisions} actions
            {all.sized > 0 && ` · sizes ${all.size} of ${all.sized}`} · {all.hands} hand{all.hands === 1 ? '' : 's'}
          </span>
          {all.decisions > 0 && (
            <button
              className="btn btn-small"
              onClick={() => window.confirm('Clear your study score in this browser?') && resetStudyScore()}
            >
              Reset
            </button>
          )}
        </div>
      </section>

      <div className="pstudy-grid">
        <div className="pstudy-main">
          {handError ? (
            <section className="panel">
              <p className="error">{handError}</p>
              <button className="btn" onClick={() => setRetry((r) => r + 1)}>
                Try again
              </button>
            </section>
          ) : !hand || !view || !shown ? (
            <section className="panel pstudy-loading" aria-busy="true">
              <p className="muted">{listed && listed.total === 0 ? 'No hands match this filter.' : 'Loading the hand…'}</p>
            </section>
          ) : (
            <>
              <div className="pstudy-title">
                <h2 className="pstudy-h2" data-testid="study-hand-title">
                  {hand.title}
                </h2>
                <span className="muted">
                  Pluribus in the {POSITION_NAMES[hand.seats[hand.hero_seat].position].toLowerCase()} · blinds {chips(hand.blinds.small_blind)}/
                  {chips(hand.blinds.big_blind)} · {hand.decisions.length} decision{hand.decisions.length === 1 ? '' : 's'}
                </span>
              </div>
              <Felt
                view={view}
                seats={[]}
                names={names}
                seatLabels={seatLabels}
                revealed={ended ? hand.result.hole_cards : null}
                winners={winners}
                label="Study table"
              />
              {decision && !answer ? (
                <Prompt decision={decision} onPick={pick} />
              ) : decision && answer ? (
                <Reveal
                  decision={decision}
                  answer={answer}
                  last={lastDecision}
                  onContinue={() => {
                    setFrame((f) => Math.min(f + 1, last))
                    setPlaying(true)
                  }}
                />
              ) : ended ? (
                <HandEnd
                  hand={hand}
                  answers={answersHere}
                  hasNext={!!nextInList && nextInList.id !== hand.id}
                  onNext={() => nextInList && select(nextInList.id)}
                  onRandom={random}
                  onAgain={restart}
                />
              ) : null}
              <div className="pstudy-controls" role="group" aria-label="Hand controls">
                <button className="btn" onClick={() => setFrame((f) => Math.max(0, f - 1))} disabled={frame === 0}>
                  ◀ Back
                </button>
                <button className="btn" onClick={() => setPlaying((v) => !v)} disabled={blocked || ended} aria-pressed={playing}>
                  {playing ? '❚❚ Pause' : '▶ Play'}
                </button>
                <button className="btn" onClick={() => setFrame((f) => Math.min(f + 1, last))} disabled={blocked || ended}>
                  Step ▶
                </button>
                <button
                  className="btn"
                  onClick={() => {
                    setPlaying(false)
                    setFrame(nextDecisionFrame())
                  }}
                  disabled={blocked || ended}
                >
                  Skip to Pluribus ⏭
                </button>
                <span className="mono muted pstudy-step">
                  step {frame + 1}/{last + 1}
                </span>
              </div>
            </>
          )}
        </div>

        <aside className="pstudy-side">
          {hand && decision ? (
            <SpotPanel hand={hand} decision={decision} />
          ) : (
            hand && (
              <section className="panel pstudy-spot pstudy-wait">
                <h2 className="pstudy-h2">The spot</h2>
                <p className="muted">
                  {ended
                    ? 'The hand is over. Step back to any Pluribus decision to look at its spot again.'
                    : 'Pot odds, equity and the hand details appear when Pluribus is next to act.'}
                </p>
              </section>
            )
          )}
          {hand && (
            <section className="log panel pstudy-log" aria-label="Hand log">
              <ol className="log-lines" data-testid="study-log">
                {lines.map((l, i) => (
                  <li key={l.seq} className={`log-${l.type}${i === lines.length - 1 ? ' log-current' : ''}`}>
                    {l.text}
                  </li>
                ))}
              </ol>
            </section>
          )}
        </aside>

        <nav className="panel pstudy-picker" aria-label="Study hands">
          {packs.length > 1 && (
            <label className="field">
              <span className="label">Pack</span>
              <select value={pack.id} onChange={(e) => navigate(`/learn/pluribus/${e.target.value}`)} aria-label="Study pack">
                {packs.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.title} · {chips(p.hands)} hands
                  </option>
                ))}
              </select>
            </label>
          )}
          <div className="segmented pstudy-filters" role="radiogroup" aria-label="Filter hands">
            {FILTERS.map((f) => (
              <button
                key={f.id}
                type="button"
                role="radio"
                aria-checked={filter === f.id}
                className={filter === f.id ? 'seg-on' : ''}
                onClick={() => {
                  setFilter(f.id)
                  navigate(`/learn/pluribus/${pack.id}`)
                }}
                title={`${chips(pack.filters[f.id] ?? 0)} hands`}
              >
                {f.label}
              </button>
            ))}
          </div>
          {listError && (
            <p className="error">
              {listError}{' '}
              <button className="btn btn-small" onClick={() => loadPage(0)}>
                Retry
              </button>
            </p>
          )}
          {!listed && !listError ? (
            <p className="muted" aria-busy="true">
              Loading hands…
            </p>
          ) : listed && listed.total === 0 ? (
            <p className="muted">No hands in this pack match the filter.</p>
          ) : listed ? (
            <>
              <p className="hint">
                {chips(listed.total)} hands · {pack.title}
              </p>
              <ol className="pstudy-list" data-testid="study-list">
                {listed.hands.map((h) => {
                  const done = score.hands[h.id]
                  const t = done ? tally(Object.values(done)) : null
                  return (
                    <li key={h.id}>
                      <button
                        className={`pstudy-item${h.id === handId ? ' pstudy-item-on' : ''}`}
                        onClick={() => select(h.id)}
                        aria-current={h.id === handId ? 'true' : undefined}
                      >
                        <span className="pstudy-item-title">{h.title}</span>
                        <span className="mono pstudy-item-meta" title={`Pot reached ${h.pot_bb} BB`}>
                          {h.position} · {h.hand} · {h.decisions} decision{h.decisions === 1 ? '' : 's'}
                        </span>
                        {t && (
                          <span className="pstudy-item-done" title="Your last result on this hand">
                            ✓ {t.action}/{t.decisions}
                          </span>
                        )}
                      </button>
                    </li>
                  )
                })}
              </ol>
              {listed.hands.length < listed.total && (
                <button className="btn btn-small btn-block" onClick={() => loadPage(listed.hands.length)} disabled={listBusy}>
                  {listBusy ? 'Loading…' : 'Show more hands'}
                </button>
              )}
            </>
          ) : null}
          <p className="hint pstudy-credit">
            Hands: Brown and Sandholm, <cite>Science</cite> 2019, via the{' '}
            <a href={pack.dataset?.repo?.replace(/\.git$/, '')} target="_blank" rel="noreferrer">
              PHH dataset
            </a>{' '}
            (MIT). {pack.caveat}
          </p>
        </nav>
      </div>
    </main>
  )
}
