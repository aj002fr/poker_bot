import { expect, test, type Page } from '@playwright/test'
import { API, WIDTHS, noHorizontalOverflow } from './support'

const PACK = 'pluribus-sample'
const HAND = 'pluribus-100-11'

async function handJson(request: import('@playwright/test').APIRequestContext, id = HAND) {
  const r = await request.get(`${API}/study/packs/${PACK}/hands/${id}`)
  expect(r.ok()).toBeTruthy()
  return r.json()
}

const cardsOnPage = (page: Page) =>
  page.evaluate(() => [...document.querySelectorAll('[data-card]')].map((e) => e.getAttribute('data-card')!))

test('pauses at each Pluribus decision, reveals its choice and keeps score', async ({ page, request }) => {
  const hand = await handJson(request)
  const hero: number = hand.hero_seat
  const visible = new Set<string>([...hand.hole_cards, ...hand.result.board])
  await page.goto(`/#/learn/pluribus/${PACK}/${HAND}`)
  await expect(page.getByTestId('study-hand-title')).toHaveText('Session 100, hand 11')

  for (const [i, d] of (hand.decisions as any[]).entries()) {
    await expect(page.getByTestId('study-prompt')).toBeVisible({ timeout: 20_000 })
    for (const c of await cardsOnPage(page)) expect(visible.has(c), `card ${c} visible before the end`).toBeTruthy()
    await expect(page.getByTestId(`seat-${hero}`)).toHaveAttribute('aria-current', 'true')
    await expect(page.getByTestId('study-odds')).toContainText('Equity')
    await page.getByTestId(`study-opt-${d.pluribus.option}`).click()
    await expect(page.getByTestId('study-reveal')).toContainText('Matched Pluribus')
    await expect(page.getByTestId('study-score')).toContainText(`${i + 1} of ${i + 1} actions`)
    await page.getByTestId('study-continue').click()
  }

  await expect(page.getByTestId('study-end')).toContainText(`You matched ${hand.decisions.length} of ${hand.decisions.length}`, {
    timeout: 20_000,
  })
  const shown = new Set(await cardsOnPage(page))
  for (const cards of Object.values(hand.result.hole_cards) as string[][]) for (const c of cards) expect(shown.has(c)).toBeTruthy()
  await expect(page.getByTestId('study-score')).toContainText('100%')

  await page.getByTestId('study-next').click()
  await expect(page.getByTestId('study-hand-title')).toHaveText('Session 100, hand 12')
})

test('a different pick says so and names what Pluribus did', async ({ page, request }) => {
  const hand = await handJson(request)
  const d = hand.decisions[0]
  const other = d.options.find((o: any) => o.kind !== d.pluribus.kind)
  await page.goto(`/#/learn/pluribus/${PACK}/${HAND}`)
  await page.getByTestId(`study-opt-${other.key}`).click()
  await expect(page.getByTestId('study-reveal')).toContainText('Different from Pluribus')
  const verb = { fold: 'folded', check: 'checked', call: 'called', raise: '' }[d.pluribus.kind as string]
  await expect(page.getByTestId('study-reveal')).toContainText(`Pluribus ${verb}`)
  await expect(page.getByTestId('study-score')).toContainText('0 of 1 actions')
})

test('from the bare study link, answering reveals on the same hand', async ({ page }) => {
  await page.goto('/#/learn/pluribus')
  await expect(page.getByTestId('study-prompt')).toBeVisible({ timeout: 20_000 })
  const title = await page.getByTestId('study-hand-title').textContent()
  await page.locator('[data-testid^="study-opt-"]').first().click()
  await expect(page.getByTestId('study-reveal')).toBeVisible()
  await expect(page.getByTestId('study-hand-title')).toHaveText(title!)
})

test('study mode fits 360, 768 and 1440 px', async ({ page }) => {
  for (const size of WIDTHS) {
    await page.setViewportSize(size)
    await page.goto(`/#/learn/pluribus/${PACK}/${HAND}`)
    await expect(page.getByTestId('study-prompt')).toBeVisible({ timeout: 20_000 })
    await expect(page.getByTestId('study-list')).toBeVisible()
    await noHorizontalOverflow(page)
  }
})

test('filters, empty and error states', async ({ page }) => {
  await page.goto('/#/learn')
  await page.getByTestId('learn-study').getByRole('link').click()
  await expect(page.getByTestId('study')).toBeVisible()
  await page.getByRole('radio', { name: 'Showdown' }).click()
  const items = page.getByTestId('study-list').locator('li')
  await expect(items.first()).toBeVisible()
  expect(await items.count()).toBeLessThan(24)

  await page.route('**/api/study/packs', (r) => r.fulfill({ json: { packs: [], compatibility_status: 'practice_assumptions' } }))
  await page.reload()
  await expect(page.getByTestId('study-message')).toContainText('No study hands yet')
  await expect(page.getByTestId('study-message')).toContainText('make study-sample')

  await page.unroute('**/api/study/packs')
  await page.route('**/api/study/packs', (r) => r.fulfill({ status: 502, body: 'bad gateway' }))
  await page.reload()
  await expect(page.getByTestId('study-message')).toContainText('Study mode is unavailable')
  await page.unroute('**/api/study/packs')
  await page.getByRole('button', { name: 'Try again' }).click()
  await expect(page.getByTestId('study')).toBeVisible()
})
