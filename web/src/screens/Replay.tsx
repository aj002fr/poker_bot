import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ApiFailure, api, loadCredentials } from '../api'
import { RULESETS, chips, describeEvents } from '../format'
import type { Perspective, Replay, SeatInfo, Snapshot } from '../types'
import { Felt } from './Felt'

const SPEEDS = [1, 4, 8, 16]
const BASE_STEP_MS = 1200

const KIND_TEXT: Record<string, string> = {
  deal: 'Cards dealt',
  action: 'Action',
  hand_end: 'Hand complete',
  complete: 'Table complete',
}

function perspectiveLabel(p: Perspective, seat: number | undefined) {
  if (p === 'seat') return `Seat ${seat} (you)`
  if (p === 'spectator') return 'Spectator'
  return 'Reviewer · all cards'
}

export function ReplayScreen({ tableId, navigate }: { tableId: string; navigate: (p: string) => void }) {
  const creds = useMemo(() => loadCredentials(tableId), [tableId])
  const [perspective, setPerspective] = useState<Perspective>(creds.seatToken ? 'seat' : 'spectator')
  const [hand, setHand] = useState<{ number?: number; at: 'first' | 'last' }>({ at: 'first' })
  const [data, setData] = useState<Replay | null>(null)
  const [frame, setFrame] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<ApiFailure | null>(null)
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const tokenFor = useCallback(
    (p: Perspective) => (p === 'seat' ? creds.seatToken ?? null : p === 'reviewer' ? creds.hostToken ?? null : null),
    [creds],
  )

  useEffect(() => {
    api.snapshot(tableId, creds.seatToken ?? creds.hostToken ?? null).then(setSnapshot, () => undefined)
  }, [tableId, creds])

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    api.replay(tableId, tokenFor(perspective), perspective, hand.number).then(
      (r) => {
        if (cancelled) return
        setData(r)
        setError(null)
        setFrame(hand.at === 'last' ? r.hand.frames.length - 1 : 0)
        setLoading(false)
      },
      (err) => {
        if (cancelled) return
        setError(err instanceof ApiFailure ? err : new ApiFailure(0, 'network', 'Could not reach the server.'))
        setPlaying(false)
        setLoading(false)
      },
    )
    return () => {
      cancelled = true
    }
  }, [tableId, perspective, hand, tokenFor])

  const hands = data?.hands ?? []
  const current = data?.hand
  const handIndex = current ? hands.findIndex((h) => h.hand_number === current.hand_number) : -1
  const frames = current?.frames ?? []
  const shown = frames[Math.min(frame, frames.length - 1)]

  const gotoHand = (index: number, at: 'first' | 'last' = 'first') => {
    if (index < 0 || index >= hands.length) return false
    setHand({ number: hands[index].hand_number, at })
    return true
  }
  const next = () => {
    if (loading) return
    if (frame < frames.length - 1) setFrame(frame + 1)
    else if (!gotoHand(handIndex + 1)) setPlaying(false)
  }
  const prev = () => {
    if (loading) return
    if (frame > 0) setFrame(frame - 1)
    else gotoHand(handIndex - 1, 'last')
  }

  const nextRef = useRef(next)
  nextRef.current = next
  useEffect(() => {
    if (!playing) return
    const id = window.setInterval(() => nextRef.current(), BASE_STEP_MS / speed)
    return () => window.clearInterval(id)
  }, [playing, speed])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement
      if (t instanceof HTMLSelectElement || (t instanceof HTMLInputElement && t.type !== 'radio')) return
      if (e.metaKey || e.ctrlKey || e.altKey) return
      if (e.key === 'ArrowRight' && e.shiftKey) gotoHand(handIndex + 1)
      else if (e.key === 'ArrowLeft' && e.shiftKey) gotoHand(handIndex - 1)
      else if (e.key === 'ArrowRight') next()
      else if (e.key === 'ArrowLeft') prev()
      else if (e.key === ' ' || e.key.toLowerCase() === 'p') {
        if (t instanceof HTMLButtonElement && e.key === ' ') return
        setPlaying((v) => !v)
      } else if (['1', '2', '3', '4'].includes(e.key)) setSpeed(SPEEDS[Number(e.key) - 1])
      else return
      e.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  const seats: SeatInfo[] = snapshot?.setup.seats ?? []
  const options: { p: Perspective; enabled: boolean; why?: string }[] = [
    ...(creds.seatToken ? [{ p: 'seat' as const, enabled: true }] : []),
    { p: 'spectator', enabled: true },
    ...(creds.hostToken
      ? [
          {
            p: 'reviewer' as const,
            enabled: !!data?.table_complete,
            why: data?.table_complete ? undefined : 'Opens after the table is complete',
          },
        ]
      : []),
  ]

  const upTo = current && shown ? current.events.filter((e) => e.seq <= shown.seq) : []
  const lines = describeEvents(upTo, seats.map((s) => ({ ...s, you: perspective === 'seat' && s.you })))
  const winners = useMemo(() => {
    if (!shown || (shown.kind !== 'hand_end' && shown.kind !== 'complete')) return undefined
    const out: Record<number, number> = {}
    for (const e of upTo) {
      if (e.type !== 'PotAwarded') continue
      for (const [s, v] of Object.entries(e.data.awards as Record<string, number>)) out[Number(s)] = (out[Number(s)] ?? 0) + v
    }
    return out
  }, [shown, upTo])

  const download = async () => {
    const history = await api.history(tableId, tokenFor(perspective), perspective)
    const blob = new Blob([JSON.stringify(history, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `${tableId}-${perspective}-history.json`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  if (error && !data) {
    return (
      <main className="panel narrow">
        <h1 className="screen-title">{error.status === 404 ? 'Table not found' : 'Replay unavailable'}</h1>
        <p className="lede">
          {error.status === 409
            ? 'The table has not started yet, so there is nothing to replay.'
            : error.status === 404
              ? 'This server keeps tables in memory, so a restart clears them.'
              : `${error.code}: ${error.message}`}
        </p>
        <button className="btn btn-primary" onClick={() => navigate(`/t/${tableId}`)}>
          Back to the table
        </button>
      </main>
    )
  }
  if (!data || !current || !shown) {
    return (
      <main className="panel narrow" aria-busy="true">
        <p className="muted">Rebuilding the table from its log…</p>
      </main>
    )
  }

  const view = shown.view
  return (
    <main className="replay" data-testid="replay">
      <div className="replay-head">
        <div>
          <h1 className="screen-title">Replay</h1>
          <p className="lede">
            {RULESETS[view.ruleset_id]?.name} · {view.seats.length} seats ·{' '}
            {data.table_complete ? 'table complete' : 'table in progress: hands so far'}
          </p>
        </div>
        <div className="replay-actions">
          <a className="btn btn-small" href={`#/t/${tableId}`}>
            {data.table_complete ? 'Results' : 'Back to the table'}
          </a>
          <button className="btn btn-small" onClick={download}>
            Download this history
          </button>
        </div>
      </div>

      <fieldset className="perspectives">
        <legend className="label">Perspective</legend>
        <div className="segmented" role="radiogroup" aria-label="Perspective">
          {options.map(({ p, enabled, why }) => (
            <button
              key={p}
              type="button"
              role="radio"
              aria-checked={perspective === p}
              className={perspective === p ? 'seg-on' : ''}
              disabled={!enabled}
              title={why}
              onClick={() => {
                setPerspective(p)
                setHand({ number: current.hand_number, at: 'first' })
              }}
            >
              {perspectiveLabel(p, creds.seat)}
            </button>
          ))}
        </div>
        <p className="hint" data-testid="perspective-note">
          {perspective === 'seat'
            ? 'What your seat saw: your own cards and anything shown at showdown.'
            : perspective === 'spectator'
              ? 'Public information only.'
              : 'Every hole card. Available to the host after the table is complete.'}
        </p>
      </fieldset>

      <div className="replay-grid">
        <div className="replay-main">
          <Felt view={view} seats={seats} revealed={shown.hole_cards} winners={winners} label="Replay table" />
          <div className="replay-controls" role="group" aria-label="Replay controls">
            <div className="replay-buttons">
              <button className="btn" onClick={() => gotoHand(handIndex - 1)} disabled={handIndex <= 0 || loading} aria-keyshortcuts="Shift+ArrowLeft">
                ⏮ Prev hand
              </button>
              <button className="btn" onClick={prev} disabled={(frame === 0 && handIndex <= 0) || loading} aria-keyshortcuts="ArrowLeft">
                ◀ Prev action
              </button>
              <button className="btn btn-primary btn-play" onClick={() => setPlaying((v) => !v)} aria-keyshortcuts="P" aria-pressed={playing}>
                {playing ? '❚❚ Pause' : '▶ Play'}
              </button>
              <button
                className="btn"
                onClick={next}
                disabled={(frame >= frames.length - 1 && handIndex >= hands.length - 1) || loading}
                aria-keyshortcuts="ArrowRight"
              >
                Next action ▶
              </button>
              <button
                className="btn"
                onClick={() => gotoHand(handIndex + 1)}
                disabled={handIndex >= hands.length - 1 || loading}
                aria-keyshortcuts="Shift+ArrowRight"
              >
                Next hand ⏭
              </button>
            </div>
            <div className="replay-row">
              <div className="segmented speed" role="radiogroup" aria-label="Playback speed">
                {SPEEDS.map((s, i) => (
                  <button
                    key={s}
                    type="button"
                    role="radio"
                    aria-checked={speed === s}
                    className={speed === s ? 'seg-on' : ''}
                    onClick={() => setSpeed(s)}
                    aria-keyshortcuts={String(i + 1)}
                  >
                    {s}×
                  </button>
                ))}
              </div>
              <label className="replay-hand">
                <span className="label">Hand</span>
                <select
                  value={current.hand_number}
                  onChange={(e) => setHand({ number: Number(e.target.value), at: 'first' })}
                  aria-label="Jump to hand"
                >
                  {hands.map((h) => (
                    <option key={h.hand_number} value={h.hand_number}>
                      Hand {h.hand_number} · {h.frames} steps
                    </option>
                  ))}
                </select>
              </label>
              <label className="replay-scrub">
                <span className="label">
                  Step <span className="mono" data-testid="replay-step">{frame + 1}</span>/{frames.length}
                </span>
                <input
                  type="range"
                  min={0}
                  max={frames.length - 1}
                  value={frame}
                  onChange={(e) => setFrame(Number(e.target.value))}
                  aria-label="Step within hand"
                />
              </label>
            </div>
            <p className="hint keys">Keys: ← → step · Shift+← → hand · P play/pause · 1–4 speed</p>
          </div>
        </div>
        <aside className="log panel" aria-label="Event log">
          <div className="replay-now">
            <span className="label">
              Hand <span className="mono" data-testid="replay-hand">{current.hand_number}</span> · {KIND_TEXT[shown.kind]}
            </span>
            {view.pot_total > 0 && <span className="mono muted">pot {chips(view.pot_total)}</span>}
          </div>
          <ol className="log-lines" data-testid="replay-log">
            {lines.map((l, i) => (
              <li key={l.seq} className={`log-${l.type}${i === lines.length - 1 ? ' log-current' : ''}`}>
                {l.text}
              </li>
            ))}
          </ol>
        </aside>
      </div>
    </main>
  )
}
