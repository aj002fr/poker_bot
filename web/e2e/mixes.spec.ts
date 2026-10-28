import { expect, test, type Page } from '@playwright/test'
import {
  assertNoLeaks,
  bot,
  createTable,
  human,
  openSeat,
  playToResults,
  reviewerEvents,
  type SeatSpec,
} from './support'

const RULESET_NAMES: Record<string, string> = {
  practice_nlhe: "No-limit Hold'em",
  uncertainty_shortdeck: "Short-deck Hold'em",
  uncertainty_plo4: 'Pot-limit Omaha',
}

const MIXES: { n: number; ruleset: string; seats: SeatSpec[] }[] = [
  { n: 2, ruleset: 'practice_nlhe', seats: [human, bot('nlhe_heuristic')] },
  { n: 3, ruleset: 'uncertainty_shortdeck', seats: [human, human, bot('legal_random')] },
  { n: 4, ruleset: 'uncertainty_plo4', seats: [bot('calling_station'), human, bot('legal_random'), bot('check_fold')] },
  {
    n: 5,
    ruleset: 'practice_nlhe',
    seats: [human, bot('nlhe_heuristic'), human, bot('calling_station'), bot('legal_random')],
  },
  {
    n: 6,
    ruleset: 'uncertainty_shortdeck',
    seats: [human, bot('legal_random'), bot('calling_station'), bot('check_fold'), bot('legal_random'), bot('check_or_call')],
  },
  {
    n: 8,
    ruleset: 'uncertainty_plo4',
    seats: [
      human,
      bot('legal_random'),
      bot('calling_station'),
      human,
      bot('check_fold'),
      bot('legal_random'),
      bot('calling_station'),
      bot('legal_random'),
    ],
  },
]

for (const mix of MIXES) {
  const humans = mix.seats.flatMap((s, i) => (s.kind === 'human' ? [i] : []))
  test(`${mix.n} seats · ${humans.length} human(s) · ${mix.ruleset} plays to results with no hidden cards in the DOM`, async ({
    browser,
    request,
  }) => {
    const t = await createTable(request, mix.seats, { ruleset_id: mix.ruleset, hand_cap: 4, hands_per_level: 2 }, { seed: `mix-${mix.n}` })
    const players: [Page, number][] = []
    for (const [k, seat] of humans.entries()) {
      const ctx = await browser.newContext({ viewport: k === 0 ? { width: 1440, height: 1000 } : { width: 390, height: 844 } })
      const page = await ctx.newPage()
      await openSeat(page, t, seat, { host: k === 0 })
      await expect(page.locator('.felt')).toBeVisible()
      await expect(page.locator('.seat')).toHaveCount(mix.n)
      const cards = mix.ruleset === 'uncertainty_plo4' ? 4 : 2
      await expect(page.getByTestId(`seat-${seat}`).locator('[data-card]')).toHaveCount(cards)
      players.push([page, seat])
    }

    const { samples, acted } = await playToResults(players)
    expect(acted).toBeGreaterThan(0)

    const [first] = players[0]
    await expect(first.getByTestId('results')).toContainText(RULESET_NAMES[mix.ruleset])
    await expect(first.locator('.results-table tbody tr')).toHaveCount(mix.n)
    const pts = (await first.locator('.results-table tbody tr td:nth-child(4)').allTextContents()).map(Number)
    const payouts = (await request.get(`http://127.0.0.1:8417/tables/${t.tableId}`).then((r) => r.json())).results.payouts
    expect(pts.reduce((a, b) => a + b, 0)).toBeCloseTo(payouts.reduce((a: number, b: number) => a + b, 0), 1)

    assertNoLeaks(samples, await reviewerEvents(request, t))
  })
}

test('eight-seat table loses its first player: 8 → 7 left, the seat reads Out', async ({ page, request }) => {
  const seats: SeatSpec[] = [human, ...Array.from({ length: 7 }, () => bot('legal_random'))]
  const t = await createTable(request, seats, { starting_stack: 60, hand_cap: 60 }, { seed: 'elimination' })
  await page.setViewportSize({ width: 1440, height: 1000 })
  await openSeat(page, t, 0, { host: true })
  await expect(page.getByTestId('survivors')).toContainText('8/8 left')

  const stopAt = Date.now() + 60_000
  let left = 8
  while (Date.now() < stopAt) {
    const text = (await page.getByTestId('survivors').textContent().catch(() => '')) ?? ''
    left = Number(text.split('/')[0])
    if (left <= 7 || (await page.getByTestId('results').isVisible())) break
    const panel = page.locator('.action-live')
    if (await panel.isVisible()) {
      const fold = panel.getByRole('button', { name: 'Fold', exact: true })
      await (await fold.isEnabled().catch(() => false) ? fold : panel.getByRole('button', { name: /^(Check|Call)/ }))
        .click({ timeout: 1000 })
        .catch(() => undefined)
    }
    await page.waitForTimeout(50)
  }
  expect(left).toBe(7)
  const out = page.locator('.seat-st-eliminated')
  await expect(out).toHaveCount(1)
  await expect(out.locator('.seat-status')).toHaveText('Out')
  await expect(out).toHaveAttribute('aria-label', /, out$/)
  await expect(out.locator('[data-card]')).toHaveCount(0)
})
