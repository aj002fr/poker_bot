import { useEffect, useMemo, useRef, useState } from 'react'
import { ApiFailure, newActionId, submitWithRetry } from '../api'
import { chips } from '../format'
import type { Action, Snapshot, TableView } from '../types'

interface Pending {
  decisionId: string
  label: string
}

export function ActionPanel({
  snapshot,
  token,
  onSnapshot,
  timer,
}: {
  snapshot: Snapshot
  token: string | null
  onSnapshot: (s: Snapshot) => void
  timer: React.ReactNode
}) {
  const view = snapshot.view as TableView
  const legal = view.legal
  const hero = view.players.find((p) => p.seat === view.hero_seat)
  const committed = hero?.street_committed ?? 0
  const [pending, setPending] = useState<Pending | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [raiseTo, setRaiseTo] = useState<number>(legal?.min_full_raise_to ?? legal?.short_all_in_raise_to ?? 0)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    setPending((p) => (p && p.decisionId !== view.decision_id ? null : p))
    setRaiseTo(legal?.min_full_raise_to ?? legal?.short_all_in_raise_to ?? 0)
    setError(null)
  }, [view.decision_id, legal?.min_full_raise_to, legal?.short_all_in_raise_to])

  const raiseValid = useMemo(() => {
    if (!legal?.can_raise) return false
    if (legal.short_all_in_raise_to !== null && raiseTo === legal.short_all_in_raise_to) return true
    return (
      legal.min_full_raise_to !== null &&
      legal.max_raise_to !== null &&
      Number.isInteger(raiseTo) &&
      raiseTo >= legal.min_full_raise_to &&
      raiseTo <= legal.max_raise_to
    )
  }, [legal, raiseTo])

  const verb = view.current_bet_to === 0 ? 'Bet' : 'Raise to'
  const disabled = !legal || pending !== null || !token

  const send = async (action: Action, label: string) => {
    if (!legal || !token || !view.hand_id || !view.decision_id || pending) return
    setPending({ decisionId: view.decision_id, label })
    setError(null)
    try {
      const res = await submitWithRetry(view.table_id, token, {
        action_id: newActionId(),
        table_id: view.table_id,
        hand_id: view.hand_id,
        decision_id: view.decision_id,
        expected_state_version: view.state_version,
        action,
      })
      onSnapshot(res.snapshot)
    } catch (err) {
      setPending(null)
      setError(err instanceof ApiFailure ? `${err.code}: ${err.message}` : 'Connection lost. Your action may not have been received.')
    }
  }

  const fold = () => legal?.fold && send({ type: 'fold' }, 'Fold')
  const checkOrCall = () => {
    if (legal?.check) send({ type: 'check' }, 'Check')
    else if (legal?.call_amount !== null && legal?.call_amount !== undefined) send({ type: 'call' }, 'Call')
  }
  const raise = () => raiseValid && send({ type: 'raise_to', amount: raiseTo }, `${verb} ${raiseTo}`)
  const allIn = () =>
    legal?.all_in_raise_to !== null &&
    legal?.all_in_raise_to !== undefined &&
    send({ type: 'raise_to', amount: legal.all_in_raise_to }, 'All in')

  useEffect(() => {
    if (disabled) return
    const onKey = (e: KeyboardEvent) => {
      const typing = e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement
      if (typing || e.metaKey || e.ctrlKey || e.altKey) return
      const k = e.key.toLowerCase()
      if (k === 'f') fold()
      else if (k === 'c' || k === 'k') checkOrCall()
      else if (k === 'r') inputRef.current?.focus()
      else if (k === 'a') allIn()
      else return
      e.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  if (!legal) {
    const turn = snapshot.turn
    return (
      <section className="action-panel action-idle" aria-label="Actions">
        <div className="action-status">
          {turn ? (
            <span>
              Waiting for <strong className="mono">Seat {turn.seat}</strong>
              {turn.kind === 'bot' ? ' (bot)' : ''}
            </span>
          ) : (
            <span>Dealing…</span>
          )}
          {timer}
        </div>
        {view.hero_seat !== null && hero?.status === 'eliminated' && <p className="hint">You are out of this table.</p>}
        {view.hero_seat === null && <p className="hint">Spectating: public information only.</p>}
      </section>
    )
  }

  const addFor = (target: number) => target - committed
  const presets: { label: string; value: number | null }[] = [
    { label: 'Min', value: legal.min_full_raise_to },
    { label: 'Pot', value: legal.pot_raise_to },
    { label: 'Max', value: legal.max_raise_to },
  ]
  const showRange = legal.can_raise && legal.min_full_raise_to !== null && legal.max_raise_to !== null

  return (
    <section className="action-panel action-live" aria-label="Your action">
      <div className="action-status">
        <span>
          <strong>Your turn</strong>
          {legal.call_amount !== null && <span className="mono"> · {chips(legal.call_amount)} to call</span>}
        </span>
        {timer}
      </div>

      <div className="action-buttons">
        <button className="btn btn-action" disabled={disabled || !legal.fold} onClick={fold} aria-keyshortcuts="F">
          Fold
        </button>
        {legal.check ? (
          <button className="btn btn-action" disabled={disabled} onClick={checkOrCall} aria-keyshortcuts="C">
            Check
          </button>
        ) : (
          <button
            className="btn btn-action"
            disabled={disabled || legal.call_amount === null}
            onClick={checkOrCall}
            aria-keyshortcuts="C"
          >
            Call {legal.call_amount !== null ? chips(legal.call_amount) : ''}
            {legal.call_amount !== null && legal.call_amount === hero?.stack ? ' · all in' : ''}
          </button>
        )}
        <button
          className="btn btn-action btn-allin"
          disabled={disabled || legal.all_in_raise_to === null}
          onClick={allIn}
          aria-keyshortcuts="A"
          title={legal.all_in_raise_to === null ? 'All in is not a legal raise here' : undefined}
        >
          All in{legal.all_in_raise_to !== null ? ` · ${chips(legal.all_in_raise_to)}` : ''}
        </button>
      </div>

      <div className={`raise-row${legal.can_raise ? '' : ' raise-off'}`}>
        {legal.can_raise ? (
          <>
            {showRange && (
              <input
                type="range"
                aria-label="Raise-to amount"
                min={legal.min_full_raise_to!}
                max={legal.max_raise_to!}
                value={Math.min(Math.max(raiseTo, legal.min_full_raise_to!), legal.max_raise_to!)}
                onChange={(e) => setRaiseTo(Number(e.target.value))}
                disabled={disabled}
              />
            )}
            <div className="raise-controls">
              <label className="raise-input">
                <span className="label">{verb === 'Bet' ? 'Bet' : 'Raise to'}</span>
                <input
                  ref={inputRef}
                  type="number"
                  inputMode="numeric"
                  value={Number.isFinite(raiseTo) ? raiseTo : ''}
                  min={legal.short_all_in_raise_to ?? legal.min_full_raise_to ?? undefined}
                  max={legal.max_raise_to ?? undefined}
                  onChange={(e) => setRaiseTo(Math.trunc(Number(e.target.value)))}
                  onKeyDown={(e) => e.key === 'Enter' && raise()}
                  disabled={disabled}
                  aria-invalid={!raiseValid}
                />
              </label>
              <div className="presets">
                {legal.short_all_in_raise_to !== null && (
                  <button className="btn btn-small" disabled={disabled} onClick={() => setRaiseTo(legal.short_all_in_raise_to!)}>
                    Short all-in
                  </button>
                )}
                {presets
                  .filter((p) => p.value !== null)
                  .map((p) => (
                    <button key={p.label} className="btn btn-small" disabled={disabled} onClick={() => setRaiseTo(p.value!)}>
                      {p.label}
                    </button>
                  ))}
              </div>
            </div>
            <button className="btn btn-primary btn-raise" disabled={disabled || !raiseValid} onClick={raise}>
              {raiseValid ? (
                <>
                  {verb} {chips(raiseTo)} <span className="add">· add {chips(addFor(raiseTo))}</span>
                </>
              ) : (
                <>
                  {verb} {chips(legal.min_full_raise_to ?? legal.short_all_in_raise_to ?? 0)}–{chips(legal.max_raise_to ?? 0)}
                </>
              )}
            </button>
            {legal.short_all_in_raise_to !== null && legal.min_full_raise_to === null && (
              <p className="hint">Only a short all-in is possible: your stack is below a full raise.</p>
            )}
          </>
        ) : (
          <p className="hint raise-blocked">{legal.raise_blocked_reason ?? 'Raising is not available on this decision.'}</p>
        )}
      </div>

      {pending && (
        <p className="pending" role="status">
          Sending {pending.label}… chips move when the server commits.
        </p>
      )}
      {error && <p className="error">{error}</p>}
      <p className="hint keys">Keys: F fold · C check/call · R raise amount · A all in</p>
    </section>
  )
}
