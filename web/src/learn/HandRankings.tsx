import { useEffect, useRef } from 'react'

// Examples are plain text, not card components, so they never count as cards on the table.
export const RANKINGS: { id: string; name: string; example: string; text: string }[] = [
  { id: 'straight_flush', name: 'Straight flush', example: '9♥ 8♥ 7♥ 6♥ 5♥', text: 'Five in a row, all one suit. Ace-high is a royal flush.' },
  { id: 'four_of_a_kind', name: 'Four of a kind', example: 'Q♠ Q♥ Q♦ Q♣ 4♠', text: 'All four cards of one rank.' },
  { id: 'full_house', name: 'Full house', example: '8♠ 8♥ 8♦ K♣ K♠', text: 'Three of one rank plus a pair of another.' },
  { id: 'flush', name: 'Flush', example: 'A♦ J♦ 9♦ 6♦ 2♦', text: 'Any five cards of one suit.' },
  { id: 'straight', name: 'Straight', example: '10♣ 9♦ 8♠ 7♥ 6♣', text: 'Five ranks in a row, mixed suits. Ace can be high or low.' },
  { id: 'three_of_a_kind', name: 'Three of a kind', example: '7♠ 7♥ 7♣ K♦ 2♠', text: 'Three cards of one rank.' },
  { id: 'two_pair', name: 'Two pair', example: 'J♠ J♦ 4♥ 4♣ A♠', text: 'Two different pairs.' },
  { id: 'one_pair', name: 'One pair', example: 'K♥ K♣ 9♠ 6♦ 3♥', text: 'Two cards of one rank. The kicker breaks ties.' },
  { id: 'high_card', name: 'High card', example: 'A♣ Q♦ 8♥ 5♠ 3♣', text: 'Nothing matched: your highest card plays.' },
]

/** ``row`` is the server's ``rank_row`` for your current hand, highlighted in the sheet. */
export function HandRankings({ open, onClose, row }: { open: boolean; onClose: () => void; row: string | null }) {
  const ref = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    const d = ref.current
    if (!d) return
    if (open && !d.open) d.showModal()
    else if (!open && d.open) d.close()
  }, [open])
  const current = RANKINGS.findIndex((r) => r.id === row)
  return (
    <dialog ref={ref} className="dialog rankings" onCancel={onClose} onClose={() => !ref.current?.open && onClose()} aria-labelledby="rankings-title">
      <h2 id="rankings-title" className="screen-title">
        Hand rankings
      </h2>
      <p className="lede">Strongest at the top. You make your best five-card hand from your two cards and the board.</p>
      <ol className="rankings-list" data-testid="hand-rankings">
        {RANKINGS.map((r, i) => (
          <li key={r.name} className={i === current ? 'ranking-on' : ''} aria-current={i === current ? 'true' : undefined}>
            <span className="ranking-name">{r.name}</span>
            <span className="mono ranking-example">{r.example}</span>
            <span className="ranking-text">{r.text}</span>
            {i === current && <span className="seat-badge seat-badge-act">You</span>}
          </li>
        ))}
      </ol>
      <div className="dialog-actions">
        <button type="button" className="btn btn-primary" onClick={onClose}>
          Close
        </button>
      </div>
    </dialog>
  )
}
