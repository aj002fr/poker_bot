import { expect, test, type Page } from '@playwright/test'

interface DomSample {
  seat: number
  hand: number | null
  feltCards: string[]
  lastHand: number | null
  lastCards: string[]
  allCards: string[]
}

async function sample(page: Page, seat: number): Promise<DomSample | null> {
  return page.evaluate((seat) => {
    const handEl = document.querySelector('[data-testid="hand-number"]')
    if (!handEl) return null
    const cards = (root: Element | Document | null) =>
      root ? [...root.querySelectorAll('[data-card]')].map((e) => e.getAttribute('data-card')!) : []
    const last = document.querySelector('[data-testid="last-hand"]')
    return {
      seat,
      hand: Number(handEl.textContent),
      feltCards: cards(document.querySelector('.felt')),
      lastHand: last ? Number(last.getAttribute('data-hand-number')) : null,
      lastCards: cards(last),
      allCards: cards(document),
    }
  }, seat)
}

async function actIfMyTurn(page: Page) {
  const panel = page.locator('.action-live')
  if (!(await panel.isVisible())) return false
  const raise = panel.locator('.btn-raise:enabled')
  if ((await raise.count()) > 0) {
    expect(await raise.first().textContent({ timeout: 500 }).catch(() => 'Raise to 0 · add 0')).toMatch(
      /^(Raise to|Bet) [\d,]+ · add [\d,]+$/,
    )
  }
  const button = panel.getByRole('button', { name: /^(Check|Call)/ })
  try {
    if (await button.isEnabled({ timeout: 500 })) {
      await button.click({ timeout: 1000 })
      return true
    }
  } catch {
    // The decision moved on while we were looking; the next poll sees the new state.
  }
  return false
}

test('two humans and four bots finish a six-seat table', async ({ browser }) => {
  const host = await browser.newContext({ viewport: { width: 1440, height: 1000 } })
  const guest = await browser.newContext({ viewport: { width: 360, height: 780 }, isMobile: true, hasTouch: true })
  const a = await host.newPage()
  const b = await guest.newPage()

  await a.goto('/')
  await expect(a.getByTestId('practice-label')).toHaveText('Practice assumptions — official kit not yet validated')
  await expect(a.getByText('Resolved configuration')).toBeVisible()
  await expect(a.getByRole('radio', { name: '6', exact: true })).toHaveAttribute('aria-checked', 'true')
  await a.getByRole('combobox').filter({ hasText: '200 hands' }).selectOption('10')
  await expect(a.locator('.blinds tbody tr')).toHaveCount(1)
  await a.getByRole('button', { name: 'Create table' }).click()

  await expect(a.getByText('Waiting for players')).toBeVisible()
  const tableId = a.url().split('/t/')[1]
  await expect(a.getByRole('button', { name: /Waiting for 1 human seat/ })).toBeDisabled()

  await b.goto(`/#/t/${tableId}/join/1`)
  await expect(b.getByText('Waiting for players')).toBeVisible()
  await a.getByRole('button', { name: 'Start table' }).click()

  await expect(a.locator('.felt')).toBeVisible()
  await expect(b.locator('.felt')).toBeVisible()
  await expect(a.getByTestId('seat-0').locator('[data-card]')).toHaveCount(2)
  await expect(b.getByTestId('seat-1').locator('[data-card]')).toHaveCount(2)
  await expect(a.getByTestId('seat-1').locator('[data-card]')).toHaveCount(0)
  await expect(b.getByTestId('practice-label')).toBeVisible()

  const samples: DomSample[] = []
  let sawMyTurn = false
  const players: [Page, number][] = [
    [a, 0],
    [b, 1],
  ]
  const stopAt = Date.now() + 200_000
  while (Date.now() < stopAt) {
    let finished = 0
    for (const [page, seat] of players) {
      if (await page.getByTestId('results').isVisible()) {
        finished++
        continue
      }
      const s = await sample(page, seat)
      if (s) samples.push(s)
      if (await actIfMyTurn(page)) sawMyTurn = true
    }
    if (finished === players.length) break
    await a.waitForTimeout(120)
  }

  await expect(a.getByTestId('results')).toBeVisible()
  await expect(b.getByTestId('results')).toBeVisible()
  expect(sawMyTurn).toBe(true)
  const rows = a.locator('.results-table tbody tr')
  await expect(rows).toHaveCount(6)
  const pts = (await a.locator('.results-table tbody tr td:nth-child(4)').allTextContents()).map(Number)
  expect(pts.reduce((x, y) => x + y, 0)).toBeCloseTo(100, 6)

  const overflow = await b.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
  expect(overflow).toBeLessThanOrEqual(0)

  // DOM leak check: every rendered card must be one this seat was entitled to see in that hand.
  const hostToken = await a.evaluate((id) => JSON.parse(sessionStorage.getItem(`arena:${id}`)!).hostToken, tableId)
  const res = await a.request.get(`/api/tables/${tableId}/history?perspective=reviewer`, {
    headers: { authorization: `Bearer ${hostToken}` },
  })
  expect(res.status()).toBe(200)
  const events: { hand_id: string | null; type: string; data: any }[] = (await res.json()).events
  const allowed = (hand: number | null, seat: number) => {
    if (hand === null) return new Set<string>()
    const id = `h_${String(hand).padStart(3, '0')}`
    const out = new Set<string>()
    for (const e of events.filter((e) => e.hand_id === id)) {
      if (e.type === 'HoleDealt' && e.data.seat === seat) out.add(e.data.card)
      if (e.type === 'BoardDealt' || e.type === 'CardsRevealed') for (const c of e.data.cards) out.add(c)
    }
    return out
  }
  expect(samples.length).toBeGreaterThan(4)
  for (const s of samples) {
    const now = allowed(s.hand, s.seat)
    const prev = allowed(s.lastHand, s.seat)
    for (const c of s.feltCards) expect(now.has(c), `seat ${s.seat} hand ${s.hand} shows ${c}`).toBe(true)
    for (const c of s.lastCards) expect(prev.has(c), `seat ${s.seat} last hand ${s.lastHand} shows ${c}`).toBe(true)
    for (const c of s.allCards) expect(now.has(c) || prev.has(c), `seat ${s.seat} page shows ${c}`).toBe(true)
  }
})
