import { useEffect, useState } from 'react'
import { ApiFailure, api, useOperatorToken, type RunBody } from '../api'
import { RULESETS } from '../format'
import type { Policy, RunBrief } from '../types'
import { formatFailure } from './Setup'

const SEAT_COUNTS = [2, 3, 4, 5, 6, 7, 8]

export function OperatorRequired({ what }: { what: string }) {
  return (
    <main className="panel narrow">
      <h1 className="screen-title">Operator only</h1>
      <p className="lede">
        {what} needs the operator credential. Use <strong>Operator</strong> in the header and paste the token that{' '}
        <code>make dev</code> printed (also in <code>artifacts/operator-token</code>).
      </p>
    </main>
  )
}

export function StatusPill({ status }: { status: string }) {
  return (
    <span className={`pill pill-${status}`} data-testid="run-status">
      {status}
    </span>
  )
}

export function RunsScreen({ navigate }: { navigate: (p: string) => void }) {
  const operator = useOperatorToken()
  const [policies, setPolicies] = useState<Policy[]>([])
  const [runs, setRuns] = useState<RunBrief[] | null>(null)
  const [cells, setCells] = useState<Set<string>>(new Set(['practice_nlhe:2', 'practice_nlhe:6']))
  const [bots, setBots] = useState<string[]>(['calling_station', 'legal_random', 'check_fold'])
  const [blocks, setBlocks] = useState(4)
  const [isolation, setIsolation] = useState<'sandbox' | 'in_process'>('sandbox')
  const [handCap, setHandCap] = useState(50)
  const [stack, setStack] = useState(2000)
  const [seed, setSeed] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [botPath, setBotPath] = useState('')
  const [botMsg, setBotMsg] = useState<{ ok: boolean; text: string } | null>(null)

  const refresh = () => {
    if (!operator) return
    api.catalog(operator).then((c) => setPolicies(c.policies.filter((p) => !p.alias_of)), () => undefined)
    api.runs(operator).then(
      (r) => setRuns(r.runs),
      () => setRuns([]),
    )
  }
  useEffect(refresh, [operator])
  useEffect(() => {
    if (!operator || !runs?.some((r) => r.status === 'running' || r.status === 'queued')) return
    const id = window.setInterval(refresh, 1500)
    return () => window.clearInterval(id)
  }, [operator, runs])

  if (!operator) return <OperatorRequired what="Launching benchmark runs" />

  const chosen = policies.filter((p) => bots.includes(p.id))
  const registeredChosen = chosen.some((p) => p.source === 'registered')
  const cellList = [...cells].map((c) => {
    const [ruleset_id, players] = c.split(':')
    return { ruleset_id, players: Number(players) }
  })
  const unsupported = cellList.flatMap((c) =>
    chosen.filter((p) => !p.rulesets.includes(c.ruleset_id)).map((p) => `${p.label} cannot play ${RULESETS[c.ruleset_id].short}`),
  )
  const tables = cellList.reduce((n, c) => n + c.players * blocks, 0)

  const toggleCell = (key: string) =>
    setCells((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  const toggleBot = (id: string) => setBots((prev) => (prev.includes(id) ? prev.filter((b) => b !== id) : [...prev, id]))

  const launch = async () => {
    setBusy(true)
    setError(null)
    const body: RunBody = {
      cells: cellList,
      bots,
      blocks,
      isolation: registeredChosen ? 'sandbox' : isolation,
      jobs: 1,
      config: { hand_cap: handCap, starting_stack: stack },
      ...(seed ? { seed } : {}),
    }
    try {
      const run = await api.createRun(operator, body)
      navigate(`/runs/${run.run_id}`)
    } catch (err) {
      setError(err instanceof ApiFailure ? formatFailure(err) : 'Could not reach the server.')
      setBusy(false)
    }
  }

  const register = async () => {
    setBotMsg(null)
    try {
      const { bot } = await api.registerBot(operator, botPath.trim())
      setBotMsg({ ok: true, text: `Registered ${bot.label} (${bot.id}) · ${bot.rulesets.map((r) => RULESETS[r]?.short ?? r).join(', ')}` })
      setBotPath('')
      setBots((prev) => (prev.includes(bot.id) ? prev : [...prev, bot.id]))
      refresh()
    } catch (err) {
      setBotMsg({ ok: false, text: err instanceof ApiFailure ? formatFailure(err) : 'Could not reach the server.' })
    }
  }

  return (
    <main className="runs">
      <section className="panel">
        <h1 className="screen-title">Batch benchmark</h1>
        <p className="lede">
          Runs the milestone 4 benchmark in the background: paired seed blocks with full seat rotation, one result set per
          scenario and seat count. Bots are cycled to fill the seats.
        </p>

        <fieldset className="field">
          <legend className="label">Scenarios × seat counts</legend>
          <div className="table-scroll">
            <table className="cell-grid">
              <thead>
                <tr>
                  <th scope="col">Scenario</th>
                  {SEAT_COUNTS.map((n) => (
                    <th key={n} scope="col" className="num">
                      {n}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(RULESETS).map(([id, r]) => (
                  <tr key={id}>
                    <th scope="row">{r.name}</th>
                    {SEAT_COUNTS.map((n) => (
                      <td key={n} className="num">
                        <input
                          type="checkbox"
                          checked={cells.has(`${id}:${n}`)}
                          onChange={() => toggleCell(`${id}:${n}`)}
                          aria-label={`${r.name}, ${n} seats`}
                        />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </fieldset>

        <fieldset className="field">
          <legend className="label">Bots</legend>
          <ul className="bot-picks">
            {policies.map((p) => (
              <li key={p.id}>
                <label className="check">
                  <input type="checkbox" checked={bots.includes(p.id)} onChange={() => toggleBot(p.id)} />
                  <span>
                    <span className="option-name">{p.label}</span>{' '}
                    <span className="muted">{p.source === 'registered' ? 'registered file' : 'baseline'}</span>
                    <span className="option-detail">
                      {p.rulesets.map((r) => RULESETS[r]?.short ?? r).join(' · ')} · <span className="mono">{p.artifact_hash.slice(0, 12)}</span>
                    </span>
                  </span>
                </label>
              </li>
            ))}
          </ul>
        </fieldset>

        <div className="field-row field-row-4">
          <label className="field">
            <span className="label">Seed blocks</span>
            <input type="number" min={1} max={500} value={blocks} onChange={(e) => setBlocks(Math.max(1, Number(e.target.value)))} />
          </label>
          <label className="field">
            <span className="label">Hand cap</span>
            <input type="number" min={1} value={handCap} onChange={(e) => setHandCap(Number(e.target.value))} />
          </label>
          <label className="field">
            <span className="label">Starting stack</span>
            <input type="number" min={1} step={100} value={stack} onChange={(e) => setStack(Number(e.target.value))} />
          </label>
          <label className="field">
            <span className="label">Isolation</span>
            <select value={registeredChosen ? 'sandbox' : isolation} onChange={(e) => setIsolation(e.target.value as 'sandbox')} disabled={registeredChosen}>
              <option value="sandbox">Sandbox (competitive)</option>
              <option value="in_process">In-process (diagnostic)</option>
            </select>
          </label>
        </div>
        <label className="field">
          <span className="label">Base seed (optional; kept private)</span>
          <input type="text" className="text" value={seed} onChange={(e) => setSeed(e.target.value)} placeholder="random" />
        </label>

        {unsupported.length > 0 && <p className="error">{unsupported.join(' · ')}</p>}
        {error && <p className="error">{error}</p>}
        <div className="launch-row">
          <button
            className="btn btn-primary"
            disabled={busy || cellList.length === 0 || bots.length === 0 || unsupported.length > 0}
            onClick={launch}
          >
            {busy ? 'Launching…' : `Launch ${tables} tables`}
          </button>
          <span className="hint">
            {cellList.length} scenario{cellList.length === 1 ? '' : 's'} · {blocks} block{blocks === 1 ? '' : 's'} × N rotations each
          </span>
        </div>
      </section>

      <aside className="runs-side">
        <section className="panel">
          <h2 className="label">Recent runs</h2>
          {runs === null ? (
            <p className="muted">Loading…</p>
          ) : runs.length === 0 ? (
            <p className="muted">No runs yet on this server.</p>
          ) : (
            <ul className="run-list">
              {runs.map((r) => (
                <li key={r.run_id}>
                  <a href={`#/runs/${r.run_id}`} className="mono">
                    {r.run_id}
                  </a>
                  <StatusPill status={r.status} />
                  <span className="muted mono">
                    {r.progress.done}/{r.progress.total}
                  </span>
                  <span className="run-meta">{r.scenarios.join(', ')}</span>
                </li>
              ))}
            </ul>
          )}
        </section>
        <section className="panel">
          <h2 className="label">Register a local bot file</h2>
          <p className="hint">
            Absolute path to a bot file or a directory with <code>bot.py</code>. It is copied, hashed and loaded only inside a
            sandboxed worker.
          </p>
          <div className="inline-form">
            <input
              type="text"
              className="text mono"
              value={botPath}
              onChange={(e) => setBotPath(e.target.value)}
              placeholder="/home/you/my_bot.py"
              aria-label="Bot file path"
              onKeyDown={(e) => e.key === 'Enter' && botPath.trim() && register()}
            />
            <button className="btn" disabled={!botPath.trim()} onClick={register}>
              Register
            </button>
          </div>
          {botMsg && <p className={botMsg.ok ? 'notice' : 'error'}>{botMsg.text}</p>}
        </section>
      </aside>
    </main>
  )
}
