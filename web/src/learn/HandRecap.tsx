import { cardText } from '../cards'
import { chips } from '../format'
import type { PublicEvent, SeatInfo, StyleProfile, TableStats } from '../types'

const STREET_TEXT: Record<string, string> = {
  preflop: 'before the flop',
  flop: 'on the flop',
  turn: 'on the turn',
  river: 'on the river',
}

/**
 * A few plain lines on a finished hand, from its public events. Showdown hands are
 * named by the server's stats tracker. ``styleOf`` returns a profile only for labelled seats.
 */
export function recapLines(
  events: PublicEvent[],
  handId: string,
  seats: SeatInfo[],
  heroSeat: number | null,
  stats: TableStats | null | undefined,
  styleOf: (seat: number) => StyleProfile | null,
): string[] {
  const name = (s: number) => (s === heroSeat ? 'You' : seats.find((x) => x.seat === s)?.label ?? `Seat ${s}`)
  const tag = (s: number) => (s === heroSeat ? 'You' : `${name(s)} (seat ${s})`)
  const board = events.filter((e) => e.type === 'BoardDealt').flatMap((e) => e.data.cards as string[])
  const actions = events.filter((e) => e.type === 'ActionApplied' && !e.data.auto)
  const shown = events.filter((e) => e.type === 'CardsRevealed')
  const awards = new Map<number, number>()
  for (const e of events.filter((e) => e.type === 'PotAwarded'))
    for (const [s, v] of Object.entries(e.data.awards as Record<string, number>))
      if (v > 0) awards.set(Number(s), (awards.get(Number(s)) ?? 0) + v)
  const winners = [...awards.entries()].sort((a, b) => b[1] - a[1])
  const words = (seat: number) =>
    stats?.seats[String(seat)]?.showdowns.items.find((i) => i.hand_id === handId)?.hand_in_words ?? null
  const lines: string[] = []

  const opener = actions.find((e) => e.data.street === 'preflop' && e.data.action?.type === 'raise_to')
  lines.push(opener ? `${tag(opener.data.seat)} raised first before the flop, to ${chips(opener.data.action.amount)}.` : 'Nobody raised before the flop.')

  if (shown.length > 0) {
    for (const e of shown) {
      const w = words(e.data.seat)
      lines.push(`${tag(e.data.seat)} showed ${(e.data.cards as string[]).map(cardText).join(' ')}${w ? `: ${w.toLowerCase()}` : ''}.`)
    }
    if (winners.length > 0) {
      const [[seat, amount]] = winners
      const w = words(seat)
      lines.push(`${tag(seat)} won ${chips(amount)} at the showdown${w ? ` with ${w.toLowerCase()}` : ''}.`)
      for (const [s, v] of winners.slice(1)) lines.push(`${tag(s)} also collected ${chips(v)} from a split or side pot.`)
    }
  } else if (winners.length > 0) {
    const [[seat, amount]] = winners
    const street = board.length === 0 ? 'preflop' : board.length === 3 ? 'flop' : board.length === 4 ? 'turn' : 'river'
    lines.push(`${tag(seat)} won ${chips(amount)} without a showdown: everyone else folded ${STREET_TEXT[street]}.`)
  }

  const explained = new Set<number>()
  for (const s of [opener?.data.seat as number | undefined, winners[0]?.[0]]) {
    if (s === undefined || s === heroSeat || explained.has(s)) continue
    explained.add(s)
    const style = styleOf(s)
    if (style) lines.push(`${tag(s)} is a ${style.name}: ${style.tells[0] ?? style.description}`)
  }

  if (heroSeat !== null) {
    const before = (events.find((e) => e.type === 'HandStarted')?.data.start_stacks as Record<string, number> | undefined)?.[String(heroSeat)]
    const after = (events.find((e) => e.type === 'HandCompleted')?.data.final_stacks as Record<string, number> | undefined)?.[String(heroSeat)]
    const folded = actions.find((e) => e.data.seat === heroSeat && e.data.action?.type === 'fold')
    if (before !== undefined && after !== undefined) {
      const diff = after - before
      lines.push(
        `${folded ? `You folded ${STREET_TEXT[folded.data.street]}. ` : ''}Your stack: ${chips(before)} → ${chips(after)} (${diff >= 0 ? '+' : '−'}${chips(Math.abs(diff))}).`,
      )
    }
  }
  return lines
}
