import { useEffect, useMemo, useRef, useState } from 'react'
import { ApiFailure, api, isTransient, loadPending, newActionId, savePending, submitWithRetry, type PendingSubmission } from '../api'
import { chips } from '../format'
import type { Action, Snapshot, TableView } from '../types'

const RESEND_MS = 1500

const BLOCKED_HELP: Record<string, string> = {
  'betting was not reopened: raises since your last action total less than a full raise':
    'Full raise unavailable: a short all-in since your last action raised by less than a full raise, so betting was not reopened. You can call or fold.',
  'stack covers only a call': 'Raise unavailable: your stack only covers a call.',
  'no opponent can respond to a raise': 'Raise unavailable: every opponent is already all in.',
}

export function blockedText(reason: string | null) {
  if (!reason) return 'Raising is not available on this decision.'
  return BLOCKED_HELP[reason] ?? `Raise unavailable: ${reason}.`
}

export function ActionPanel({
  snapshot,
  token,
  onSnapshot,
  timer,
  connections,
}: {
  snapshot: Snapshot
  token: string | null
  onSnapshot: (s: Snapshot) => void
  timer: React.ReactNode
  connections: number
}) {
  const view = snapshot.view as TableView
  const legal = view.legal
  const hero = view.players.find((p) => p.seat === view.hero_seat)
  const committed = hero?.street_committed ?? 0
  const tableId = view.table_id
  const [pending, setPending] = useState<PendingSubmission | null>(() => loadPending(tableId))
  const [notice, setNotice] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const inflight = useRef(false)
  const [raiseTo, setRaiseTo] = useState<number>(legal?.min_full_raise_to ?? legal?.short_all_in_raise_to ?? 0)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    setRaiseTo(legal?.min_full_raise_to ?? legal?.short_all_in_raise_to ?? 0)
    if (!pending) setError(null)
  }, [view.decision_id, legal?.min_full_raise_to, legal?.short_all_in_raise_to])

  const settle = (message: string | null) => {
    savePending(tableId, null)
    setPending(null)
    setNotice(message)
  }

  // An unanswered submission is resent with the same action_id: the server either
  // reports the stored receipt (it had committed) or rejects it (it had not).
  const resolvePending = async (p: PendingSubmission) => {
    if (!token || inflight.current) return
    inflight.current = true
    try {
      const res = await api.act(tableId, token, p.envelope)
      settle(
        res.replayed
          ? `${p.label} was received before the connection dropped (confirmed by its action id).`
          : `${p.label} sent after reconnecting.`,
      )
      setError(null)
      onSnapshot(res.snapshot)
    } catch (err) {
      if (err instanceof ApiFailure && !isTransient(err)) {
        settle(null)
        setError(`${p.label} was not applied: ${err.code === 'deadline_passed' ? 'the turn timed out first' : err.message}.`)
      }
    } finally {
      inflight.current = false
    }
  }

  useEffect(() => {
    const p = loadPending(tableId)
    if (!p || p.seat !== view.hero_seat) return
    resolvePending(p)
    const id = window.setInterval(() => resolvePending(p), RESEND_MS)
    return () => window.clearInterval(id)
  }, [connections, pending?.envelope.action_id, token])

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
    if (!legal || !token || !view.hand_id || !view.decision_id || pending || view.hero_seat === null) return
    const submission: PendingSubmission = {
      seat: view.hero_seat,
      label,
      envelope: {
        action_id: newActionId(),
        table_id: tableId,
        hand_id: view.hand_id,
        decision_id: view.decision_id,
        expected_state_version: view.state_version,
        action,
      },
    }
    savePending(tableId, submission)
    setPending(submission)
    setNotice(null)
    setError(null)
    inflight.current = true
    try {
      const res = await submitWithRetry(tableId, token, submission.envelope)
      settle(null)
      onSnapshot(res.snapshot)
    } catch (err) {
      if (isTransient(err)) {
        setError(`The table is paused while storage is unavailable. ${label} is saved and will be confirmed by its action id when it resumes.`)
      } else if (err instanceof ApiFailure) {
        settle(null)
        setError(`${err.code}: ${err.message}`)
      } else {
        setError(`Connection lost. ${label} is saved and will be confirmed by its action id when the server is reachable.`)
      }
    } finally {
      inflight.current = false
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
      const typing =
        e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement || e.target instanceof HTMLTextAreaElement
      if (typing && e.key === 'Escape') {
        ;(e.target as HTMLElement).blur()
        return
      }
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

  const messages = (
    <>
      {pending && (
        <p className="pending" role="status" data-testid="pending">
          Sending {pending.label}… chips move when the server commits.
        </p>
      )}
      {notice && (
        <p className="notice" role="status" data-testid="notice">
          {notice}
        </p>
      )}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
    </>
  )

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
        {messages}
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
          <p className="hint raise-blocked" data-testid="raise-blocked">
            {blockedText(legal.raise_blocked_reason)}
          </p>
        )}
      </div>

      {messages}
      <p className="hint keys">Keys: F fold · C check/call · R raise amount, Enter to send, Esc to leave · A all in</p>
    </section>
  )
}
