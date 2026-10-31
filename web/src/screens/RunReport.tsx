import { useEffect, useState } from 'react'
import { ApiFailure, api, useOperatorToken } from '../api'
import { RULESETS, points } from '../format'
import type { CellManifest, EntrantSummary, RunCellView, RunView } from '../types'
import { OperatorRequired, StatusPill } from './Runs'

function duration(s: number | null | undefined) {
  if (s === null || s === undefined) return '–'
  if (s < 1e-3) return `${(s * 1e6).toFixed(0)} µs`
  if (s < 1) return `${(s * 1e3).toFixed(2)} ms`
  return `${s.toFixed(2)} s`
}

const short = (h: string | undefined) => (h ? `${h.slice(0, 12)}…` : '–')
const baseName = (name: string) => name.replace(/#\d+$/, '')

function CiBar({ mean, ci, max }: { mean: number | null; ci: [number, number] | null; max: number }) {
  if (mean === null || max <= 0) return null
  const pct = (v: number) => `${Math.max(0, Math.min(100, (v / max) * 100))}%`
  return (
    <span className="ci-bar" aria-hidden="true">
      {ci && <span className="ci-range" style={{ left: pct(ci[0]), width: `calc(${pct(ci[1])} - ${pct(ci[0])})` }} />}
      <span className="ci-mean" style={{ left: pct(mean) }} />
    </span>
  )
}

function EntrantsTable({ m }: { m: CellManifest }) {
  const max = Math.max(1, ...m.config.payout_points)
  const hash = (e: EntrantSummary) => m.entrants.find((x) => x.entrant === e.entrant)?.artifact_sha256
  return (
    <div className="table-scroll">
      <table className="results-table report-table">
        <thead>
          <tr>
            <th>Entrant</th>
            <th>Artifact sha256</th>
            <th className="num">Tables</th>
            <th className="num">Mean points</th>
            <th>95% CI (block bootstrap)</th>
            <th className="num">Decisions</th>
            <th className="num">Fallbacks</th>
            <th className="num">Timeouts</th>
            <th className="num">p50</th>
            <th className="num">p95</th>
            <th className="num">p99</th>
          </tr>
        </thead>
        <tbody>
          {m.summary.entrants.map((e) => {
            const breakdown = Object.entries(e.fallbacks)
              .map(([k, v]) => `${v} ${k.replace(/_/g, ' ')}`)
              .join(', ')
            const t = e.timing.end_to_end_s
            return (
              <tr key={e.entrant}>
                <td>
                  <span className="mono">{e.name}</span>
                  <span className="muted"> · {e.builtin ? 'baseline' : 'registered'}</span>
                </td>
                <td className="mono hash" title={hash(e)}>
                  {short(hash(e))}
                </td>
                <td className="mono num">{e.tables}</td>
                <td className="mono num">{e.mean_points === null ? '–' : points(Number(e.mean_points.toFixed(3)))}</td>
                <td>
                  <span className="ci-cell">
                    <span className="mono">
                      {e.points_ci95 ? `${e.points_ci95[0].toFixed(2)} – ${e.points_ci95[1].toFixed(2)}` : 'needs ≥ 2 blocks'}
                    </span>
                    <CiBar mean={e.mean_points} ci={e.points_ci95} max={max} />
                  </span>
                </td>
                <td className="mono num">{e.decisions.toLocaleString('en-US')}</td>
                <td className="mono num" title={breakdown || 'none'}>
                  {e.fallback_total}
                  {breakdown && <span className="hint"> ({breakdown})</span>}
                </td>
                <td className="mono num">{e.timeout_fallbacks}</td>
                <td className="mono num">{duration(t.p50)}</td>
                <td className="mono num">{duration(t.p95)}</td>
                <td className="mono num">{duration(t.p99)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function Cell({ cell }: { cell: RunCellView }) {
  const m = cell.manifest
  const s = m?.summary
  return (
    <section className="panel report-cell" data-testid={`cell-${cell.cell_id}`}>
      <div className="report-cell-head">
        <h2 className="report-cell-title">
          {RULESETS[cell.ruleset_id]?.name ?? cell.ruleset_id} · {cell.players} seats
        </h2>
        <StatusPill status={cell.status} />
        {m && (
          <span className={`pill ${m.competitive ? 'pill-complete' : 'pill-diag'}`}>
            {m.competitive ? 'competitive · sandboxed' : 'diagnostic · not competitive'}
          </span>
        )}
      </div>
      {cell.status !== 'complete' && cell.status !== 'failed' && (
        <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={cell.total} aria-valuenow={cell.done}>
          <span style={{ width: `${cell.total ? (cell.done / cell.total) * 100 : 0}%` }} />
          <span className="progress-text mono">
            {cell.done}/{cell.total} tables
          </span>
        </div>
      )}
      {cell.error && <p className="error">{cell.error}</p>}
      <dl className="kv kv-wide">
        <dt>Config hash</dt>
        <dd className="mono hash" title={cell.config_hash}>
          {cell.config_hash}
        </dd>
        {m && (
          <>
            <dt>Manifest hash</dt>
            <dd className="mono hash" title={m.content_hash}>
              {m.content_hash}
            </dd>
            <dt>Benchmark id</dt>
            <dd className="mono">{m.run_id}</dd>
            <dt>Seed list sha256</dt>
            <dd className="mono hash">{short(m.seed_list_sha256)}</dd>
            <dt>Schedule</dt>
            <dd>
              {m.seed_blocks} seed blocks × {m.rotations_per_block} rotations = {m.scheduled_tables} tables · hand cap{' '}
              {m.config.hand_cap} · stack {m.config.starting_stack}
            </dd>
            <dt>Isolation</dt>
            <dd>
              {m.isolation === 'sandbox' ? `sandboxed workers, ${duration(m.budget_ns / 1e9)} budget` : 'in-process, deadline not enforced'}
            </dd>
          </>
        )}
      </dl>
      {s && (
        <div className="stat-row">
          <div className="stat">
            <span className="label">Completed</span>
            <span className="stat-value mono" data-testid="tables-completed">
              {s.tables_completed}/{s.tables_scheduled}
            </span>
          </div>
          <div className="stat">
            <span className="label">Invalid</span>
            <span className={`stat-value mono${s.tables_failed.length ? ' bad' : ''}`} data-testid="tables-invalid">
              {s.tables_failed.length + s.tables_incomplete.length}
            </span>
          </div>
          <div className="stat">
            <span className="label">Mean table wall</span>
            <span className="stat-value mono">{duration(s.table_wall_s_mean)}</span>
          </div>
          <div className="stat">
            <span className="label">Scenario wall</span>
            <span className="stat-value mono">{duration(cell.wall_s)}</span>
          </div>
        </div>
      )}
      {m && <EntrantsTable m={m} />}
      {s && s.tables_failed.length > 0 && (
        <details>
          <summary>Failed tables ({s.tables_failed.length})</summary>
          <ul className="log-lines">
            {s.tables_failed.map((f) => (
              <li key={f.table_id}>
                {f.table_id}: {f.error}
              </li>
            ))}
          </ul>
        </details>
      )}
      {m && <p className="hint">{m.report_note}</p>}
    </section>
  )
}

function Overview({ run }: { run: RunView }) {
  const done = run.cells.filter((c) => c.manifest)
  if (done.length === 0) return null
  const names = [...new Set(done.flatMap((c) => c.manifest!.summary.entrants.map((e) => baseName(e.name))))]
  const cellMean = (c: RunCellView, name: string) => {
    const es = c.manifest!.summary.entrants.filter((e) => baseName(e.name) === name && e.mean_points !== null)
    const tables = es.reduce((n, e) => n + e.tables, 0)
    return tables ? es.reduce((sum, e) => sum + (e.mean_points ?? 0) * e.tables, 0) / tables : null
  }
  return (
    <section className="panel">
      <h2 className="label">Mean points by scenario and seat count</h2>
      <div className="table-scroll">
        <table className="results-table" data-testid="overview">
          <thead>
            <tr>
              <th>Bot</th>
              {done.map((c) => (
                <th key={c.cell_id} className="num">
                  {RULESETS[c.ruleset_id]?.short} · {c.players}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {names.map((n) => (
              <tr key={n}>
                <td className="mono">{n}</td>
                {done.map((c) => {
                  const v = cellMean(c, n)
                  return (
                    <td key={c.cell_id} className="mono num">
                      {v === null ? '–' : v.toFixed(2)}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="hint">
        Points are per table; the payout vector differs by seat count, so compare within a column. Copies of one bot at a
        table are averaged.
      </p>
    </section>
  )
}

export function RunReport({ runId }: { runId: string }) {
  const operator = useOperatorToken()
  const [run, setRun] = useState<RunView | null>(null)
  const [error, setError] = useState<ApiFailure | null>(null)

  useEffect(() => {
    if (!operator) return
    let stop = false
    let timer: number | undefined
    const poll = () =>
      api.run(operator, runId).then(
        (r) => {
          if (stop) return
          setRun(r)
          setError(null)
          if (r.status === 'queued' || r.status === 'running') timer = window.setTimeout(poll, 1000)
        },
        (err) => !stop && setError(err instanceof ApiFailure ? err : new ApiFailure(0, 'network', 'Could not reach the server.')),
      )
    poll()
    return () => {
      stop = true
      window.clearTimeout(timer)
    }
  }, [operator, runId])

  if (!operator) return <OperatorRequired what="The batch report" />
  if (error && !run) {
    return (
      <main className="panel narrow">
        <h1 className="screen-title">{error.status === 404 ? 'Run not found' : 'Report unavailable'}</h1>
        <p className="lede">
          {error.status === 404 ? 'Runs are kept in memory; a server restart clears the list (manifests stay on disk).' : error.message}
        </p>
        <a className="btn" href="#/runs">
          Back to runs
        </a>
      </main>
    )
  }
  if (!run) {
    return (
      <main className="panel narrow" aria-busy="true">
        <p className="muted">Loading report…</p>
      </main>
    )
  }
  const pct = run.progress.total ? (run.progress.done / run.progress.total) * 100 : 0
  return (
    <main className="report" data-testid="run-report">
      <section className="panel">
        <div className="report-cell-head">
          <h1 className="screen-title">Batch report</h1>
          <span className="mono muted">{run.run_id}</span>
          <StatusPill status={run.status} />
        </div>
        <div className="progress" role="progressbar" aria-valuemin={0} aria-valuemax={run.progress.total} aria-valuenow={run.progress.done}>
          <span style={{ width: `${pct}%` }} />
          <span className="progress-text mono" data-testid="run-progress">
            {run.progress.done}/{run.progress.total} tables
          </span>
        </div>
        <dl className="kv kv-wide">
          <dt>Started</dt>
          <dd className="mono">{run.created_at}</dd>
          <dt>Schedule</dt>
          <dd>
            {run.request.blocks} paired seed blocks per scenario · {run.request.isolation === 'sandbox' ? 'sandboxed' : 'in-process diagnostic'}
          </dd>
          <dt>Config overrides</dt>
          <dd className="mono">{JSON.stringify(run.request.config)}</dd>
          <dt>Bots</dt>
          <dd>
            <ul className="hash-list">
              {run.bots.map((b) => (
                <li key={b.id}>
                  <span className="mono">{b.id}</span> <span className="muted">({b.label})</span>{' '}
                  <span className="mono hash" title={b.artifact_hash}>
                    sha256 {b.artifact_hash}
                  </span>
                </li>
              ))}
            </ul>
          </dd>
        </dl>
        {run.error && <p className="error">{run.error}</p>}
        <p className="practice-note">Practice assumptions; not official Sequoia rules. Results against baselines say little about strength.</p>
      </section>
      <Overview run={run} />
      {run.cells.map((c) => (
        <Cell key={c.cell_id} cell={c} />
      ))}
    </main>
  )
}
