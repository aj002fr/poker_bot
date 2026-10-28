import { expect, test, type Page } from '@playwright/test'
import { bot, createTable, human, noHorizontalOverflow, openSeat } from './support'

const WIDTHS = [
  { width: 360, height: 780 },
  { width: 768, height: 1024 },
  { width: 1440, height: 1000 },
]

type Box = { id: string; x: number; y: number; w: number; h: number }

async function feltBoxes(page: Page) {
  return page.evaluate(() => {
    const box = (el: Element, id: string) => {
      const r = el.getBoundingClientRect()
      return { id, x: r.x, y: r.y, w: r.width, h: r.height }
    }
    const felt = document.querySelector('.felt')!
    return {
      felt: box(felt, 'felt'),
      seats: [...felt.querySelectorAll('[data-testid^="seat-"]')].map((e) => box(e, e.getAttribute('data-testid')!)),
      board: box(felt.querySelector('.board')!, 'board'),
    }
  })
}

const overlap = (a: Box, b: Box) =>
  Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x)) * Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y))

async function expectCleanFelt(page: Page, seats: number) {
  const { felt, seats: boxes, board } = await feltBoxes(page)
  expect(boxes).toHaveLength(seats)
  for (const [i, a] of boxes.entries()) {
    expect(a.x, `${a.id} left edge`).toBeGreaterThanOrEqual(felt.x - 0.5)
    expect(a.y, `${a.id} top edge`).toBeGreaterThanOrEqual(felt.y - 0.5)
    expect(a.x + a.w, `${a.id} right edge`).toBeLessThanOrEqual(felt.x + felt.w + 0.5)
    expect(a.y + a.h, `${a.id} bottom edge`).toBeLessThanOrEqual(felt.y + felt.h + 0.5)
    expect(overlap(a, board), `${a.id} overlaps the board`).toBeLessThanOrEqual(1)
    for (const b of boxes.slice(i + 1)) expect(overlap(a, b), `${a.id} overlaps ${b.id}`).toBeLessThanOrEqual(1)
  }
}

async function expectTextLabelledCards(page: Page) {
  const cards = await page.locator('[data-card]').evaluateAll((els) =>
    els.map((e) => ({ card: e.getAttribute('data-card')!, name: e.getAttribute('aria-label'), text: e.textContent ?? '' })),
  )
  expect(cards.length).toBeGreaterThan(0)
  const suits: Record<string, [string, string]> = { s: ['♠', 'spades'], h: ['♥', 'hearts'], d: ['♦', 'diamonds'], c: ['♣', 'clubs'] }
  for (const { card, name, text } of cards) {
    const [glyph, word] = suits[card.at(-1)!]
    expect(text, `${card} prints its suit`).toContain(glyph)
    expect(text.replace(glyph, '').trim().length, `${card} prints its rank`).toBeGreaterThan(0)
    expect(name, `${card} names its suit`).toContain(word)
  }
}

for (const ruleset of ['practice_nlhe', 'uncertainty_plo4', 'uncertainty_shortdeck']) {
  test(`8-seat ${ruleset} table fits at 360, 768 and 1440 px without overlaps`, async ({ page, request }) => {
    const seats = [human, ...Array.from({ length: 7 }, () => bot('calling_station'))]
    const t = await createTable(request, seats, { ruleset_id: ruleset, hand_cap: 10 }, { seed: `layout-${ruleset}` })
    await openSeat(page, t, 0)
    await expect(page.locator('.action-live')).toBeVisible()
    for (const size of WIDTHS) {
      await page.setViewportSize(size)
      await expect(page.locator('.felt')).toBeVisible()
      await expectCleanFelt(page, 8)
      await noHorizontalOverflow(page)
      await expectTextLabelledCards(page)
      for (const name of ['Fold', 'All in']) {
        const box = (await page.locator('.action-live').getByRole('button', { name: new RegExp(`^${name}`) }).boundingBox())!
        expect(box.x).toBeGreaterThanOrEqual(0)
        expect(box.x + box.width).toBeLessThanOrEqual(size.width)
      }
    }
    // Seat labels carry status in words, not colour alone.
    await expect(page.getByTestId('seat-0')).toHaveAttribute('aria-label', /Seat 0, You, [\d,]+ chips, to act/)
    await expect(page.getByTestId('seat-0').getByText('To act')).toBeVisible()
  })
}

test('setup, replay and spectator views have no horizontal overflow at each width', async ({ page, request }) => {
  const t = await createTable(request, Array.from({ length: 8 }, () => bot('calling_station')), { hand_cap: 2 }, { seed: 'layout-views' })
  for (const size of WIDTHS) {
    await page.setViewportSize(size)
    await page.goto('/')
    await page.getByRole('radio', { name: '8', exact: true }).click()
    await expect(page.getByRole('combobox', { name: 'Seat 7 player' })).toBeVisible()
    await noHorizontalOverflow(page)

    await page.goto(`/#/t/${t.tableId}`)
    await expect(page.locator('.felt, [data-testid="results"]').first()).toBeVisible()
    await noHorizontalOverflow(page)

    await page.goto(`/#/t/${t.tableId}/replay`)
    await expect(page.getByTestId('replay')).toBeVisible()
    await expectCleanFelt(page, 8)
    await noHorizontalOverflow(page)
  }
})
