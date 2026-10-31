import type { Ratio, SeatInfo, SeatStats, Snapshot, StyleProfile, StylesCatalog } from '../types'

/** The profile shown for a seat: only labelled seats, or any seat once the table has revealed. */
export function visibleProfile(info: SeatInfo | undefined, catalog: StylesCatalog | null): StyleProfile | null {
  if (!info || !catalog || !info.style) return null
  return catalog.styles.find((s) => s.id === info.style) ?? null
}

export function isHiddenLive(info: SeatInfo | undefined, snapshot: Snapshot) {
  return !!info?.hidden && !snapshot.setup.styles?.revealed
}

export const rate = (r: Ratio) => (r.rate === null ? '–' : `${Math.round(r.rate * 100)}%`)
export const range = (r: [number, number] | null | undefined, ratio = false) =>
  !r ? null : ratio ? `${r[0].toFixed(1)}–${r[1].toFixed(1)}` : `${Math.round(r[0] * 100)}–${Math.round(r[1] * 100)}%`

/** Compact badge text for a seat on the felt, with sample sizes in plain words. */
export function badgeText(s: SeatStats) {
  return `Played ${s.vpip.count}/${s.vpip.of} · raised ${s.pfr.count}`
}

export function badgeTitle(s: SeatStats) {
  const af = s.aggression.factor
  return [
    `Hands played: ${s.vpip.count} of ${s.vpip.of}`,
    `Raised before the flop: ${s.pfr.count} of ${s.pfr.of}`,
    `Bets and raises vs calls: ${s.aggression.bets_raises} to ${s.aggression.calls}${af === null ? '' : ` (${af.toFixed(1)}×)`}`,
    `Showdowns: ${s.showdowns.count}`,
  ].join('. ')
}
