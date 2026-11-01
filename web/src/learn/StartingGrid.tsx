import { useMemo } from 'react'
import type { ChartGroup, StylesCatalog } from '../types'
import { GROUP_TEXT } from './positions'

const nice = (code: string) => code.replace('T', '10')

/** Where every starting hand sits in the conservative and aggressive opening charts for one position group. */
export function StartingGrid({ catalog, group, hand }: { catalog: StylesCatalog; group: ChartGroup; hand: string | null }) {
  const cons = useMemo(() => new Set(catalog.charts.conservative[group]), [catalog, group])
  const aggr = useMemo(() => new Set(catalog.charts.aggressive[group]), [catalog, group])
  const verdict = !hand
    ? null
    : cons.has(hand)
      ? 'in the conservative chart: a solid open for any style'
      : aggr.has(hand)
        ? 'in the aggressive chart only: playable if you like to attack'
        : 'outside both charts: usually a fold when nobody has raised'
  return (
    <div className="grid-wrap" data-testid="starting-grid">
      {hand && (
        <p className="grid-verdict">
          <strong className="mono">{nice(hand)}</strong> from {GROUP_TEXT[group]} is {verdict}.
        </p>
      )}
      <div className="hand-grid" role="grid" aria-label={`Opening charts for ${GROUP_TEXT[group]}`}>
        {catalog.grid.map((row, r) => (
          <div role="row" key={r} className="hand-grid-row">
            {row.map((code) => {
              const tier = cons.has(code) ? 'cons' : aggr.has(code) ? 'aggr' : 'none'
              const name = `${nice(code)}: ${tier === 'cons' ? 'conservative chart' : tier === 'aggr' ? 'aggressive chart only' : 'in neither chart'}`
              return (
                <span
                  role="gridcell"
                  key={code}
                  className={`hg-cell hg-${tier}${code === hand ? ' hg-you' : ''}`}
                  aria-label={name}
                  title={name}
                  aria-current={code === hand ? 'true' : undefined}
                >
                  {code}
                </span>
              )
            })}
          </div>
        ))}
      </div>
      <ul className="grid-legend">
        <li>
          <span className="hg-swatch hg-cons" aria-hidden="true" /> Conservative
        </li>
        <li>
          <span className="hg-swatch hg-aggr" aria-hidden="true" /> Aggressive only
        </li>
        <li>
          <span className="hg-swatch hg-you-swatch" aria-hidden="true" /> Your hand
        </li>
      </ul>
      <p className="hint">
        Pairs run down the diagonal, suited hands (s) sit above it and offsuit hands (o) below. Charts follow the opening styles in
        Part Two of <cite>Harrington on Hold'em, Vol. I</cite>.
      </p>
    </div>
  )
}
