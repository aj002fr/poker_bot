import { expect, test, type Page } from '@playwright/test'
import {
  API,
  WIDTHS,
  actIfMyTurn,
  assertNoLeaks,
  bot,
  createTable,
  expectCleanFelt,
  human,
  noHorizontalOverflow,
  openSeat,
  playToResults,
  reviewerEvents,
  sampleDom,
  type DomSample,
} from './support'

const STYLE_NAMES = ['Rock', 'Aggressive', 'Super-aggressive', 'Calling station', 'Weak-tight', 'Style switcher']

/** Text of an element with every <select> removed: guess dropdowns list all style names by design. */
async function textWithoutSelects(page: Page, selector: string) {
  return page.evaluate((selector) => {
    return [...document.querySelectorAll(selector)].map((el) => {
      const copy = el.cloneNode(true) as Element
      copy.querySelectorAll('select').forEach((s) => s.remove())
      return { id: el.getAttribute('data-testid'), text: copy.textContent ?? '', title: el.getAttribute('title') ?? '' }
    })
  }, selector)
}

async function styleSecrets(request: import('@playwright/test').APIRequestContext) {
  const catalog = await (await request.get(`${API}/catalog`)).json()
  const styles = catalog.policies.filter((p: any) => p.source === 'style')
  return { ids: styles.map((p: any) => p.id as string), hashes: styles.map((p: any) => p.artifact_hash as string) }
}

test('labelled table shows learning aids from the seat’s own view at 360, 768 and 1440 px', async ({ page, request }) => {
  const t = await createTable(
    request,
    [human, bot('style_rock'), bot('style_aggressive'), bot('style_station'), bot('style_weak_tight'), bot('style_super_aggressive')],
    { hand_cap: 6 },
    { seed: 'learn-labelled', styles: { mode: 'labelled', learning_aids: true } },
  )
  await openSeat(page, t, 0)
  await expect(page.locator('.action-live')).toBeVisible()

  const aids = page.getByTestId('learning-aids')
  await expect(aids.getByTestId('hand-words').locator('.learn-big')).not.toBeEmpty()
  await expect(aids.getByTestId('hero-position')).toContainText(/Under the gun|Middle position|Cutoff|Button|Small blind|Big blind/)
  await expect(aids.getByTestId('equity')).toContainText(/about \d+%/)
  await expect(aids.getByTestId('stack-ratio')).toContainText('rounds of blinds')
  await expect(aids.getByTestId('starting-grid').locator('.hg-you')).toHaveCount(1)
  await expect(page.locator('[data-testid^="pos-badge-"]').first()).toBeVisible()

  // Labelled seats carry their style name on the felt and in the notes panel.
  await expect(page.getByTestId('seat-1')).toContainText('Rock')
  await expect(page.getByTestId('notes-seat-2')).toContainText('Aggressive')

  // The coach is optional and labelled as a heuristic.
  await aids.getByLabel('Coach hint').check()
  await expect(aids.getByTestId('coach-hint')).toContainText(/not optimal play/i)

  // Hand rankings sheet highlights the current row.
  await aids.getByRole('button', { name: 'Hand rankings' }).click()
  await expect(page.getByTestId('hand-rankings').locator('[aria-current="true"]')).toHaveCount(1)
  await page.keyboard.press('Escape')

  // Glossary terms open on focus with our own definition.
  await aids.locator('[data-term="equity"]').focus()
  await expect(page.getByRole('tooltip')).toContainText('Your share of the pot')

  // The aids endpoint only ever returns this seat's own hole cards.
  const res = await request.get(`${API}/tables/${t.tableId}/aids`, { headers: { authorization: `Bearer ${t.seatTokens[0]}` } })
  const body = await res.json()
  const snap = await (await request.get(`${API}/tables/${t.tableId}`, { headers: { authorization: `Bearer ${t.seatTokens[0]}` } })).json()
  expect(body.hero.hole_cards.sort()).toEqual([...snap.view.hole_cards].sort())

  for (const size of WIDTHS) {
    await page.setViewportSize(size)
    await expect(aids).toBeVisible()
    await expectCleanFelt(page, 6)
    await noHorizontalOverflow(page)
  }

  // Play the table out; the page never shows a card this seat may not see, and every hand gets a recap.
  const samples: DomSample[] = []
  const { samples: more } = await playToResults([[page, 0]], { policy: 'call' })
  samples.push(...more)
  assertNoLeaks(samples, await reviewerEvents(request, t))
  await expect(page.getByTestId('reveal')).toContainText('Every style was labelled')
})

test('hand recap explains each finished hand', async ({ page, request }) => {
  const t = await createTable(request, [human, bot('style_station'), bot('style_rock')], { hand_cap: 3 }, { seed: 'learn-recap' })
  await openSeat(page, t, 0)
  const stopAt = Date.now() + 60_000
  while (Date.now() < stopAt && !(await page.getByTestId('hand-recap').isVisible())) {
    await actIfMyTurn(page, 'call')
    await page.waitForTimeout(60)
  }
  const recap = page.getByTestId('hand-recap')
  await expect(recap).toContainText(/raised first|Nobody raised/)
  await expect(recap).toContainText(/won [\d,]+/)
  await expect(recap).toContainText('Your stack:')
})

test('hidden styles never reach the DOM or live frames before the reveal; guesses are scored after it', async ({ page, request }) => {
  const { ids, hashes } = await styleSecrets(request)
  const seats = [human, bot('style_rock'), bot('style_aggressive'), bot('style_station'), bot('style_weak_tight')]
  const t = await createTable(request, seats, { hand_cap: 5 }, { seed: 'learn-hidden', styles: { mode: 'hidden', learning_aids: true } })

  const frames: any[] = []
  page.on('websocket', (ws) => ws.on('framereceived', (f) => typeof f.payload === 'string' && frames.push(JSON.parse(f.payload))))
  const bodies: string[] = []
  page.on('response', async (r) => {
    if (r.url().includes(`/tables/${t.tableId}`)) bodies.push(await r.text().catch(() => ''))
  })

  await openSeat(page, t, 0)
  await expect(page.getByTestId('notes-panel')).toBeVisible()
  for (const seat of [1, 2, 3, 4]) {
    await expect(page.getByTestId(`seat-${seat}`)).toContainText(`Player ${seat}`)
    await expect(page.getByTestId(`notes-seat-${seat}`)).toContainText('Hidden')
  }
  await page.getByLabel('Style guess for seat 1').selectOption('rock')
  await page.getByLabel('Style guess for seat 2').selectOption('station')
  await page.getByLabel('Notes on seat 3').fill('calls everything so far')

  const checkDom = async () => {
    const html = await page.content()
    for (const id of ids) expect(html, `DOM names ${id}`).not.toContain(id)
    for (const h of hashes) expect(html, 'DOM shows a style artifact hash').not.toContain(h)
    for (const el of await textWithoutSelects(page, '[data-testid^="seat-"], [data-testid^="notes-seat-"], [data-testid="hand-recap"]'))
      for (const name of STYLE_NAMES) expect(`${el.text} ${el.title}`, `${el.id} names ${name}`).not.toContain(name)
  }

  let results = false
  let checks = 0
  const stopAt = Date.now() + 150_000
  while (Date.now() < stopAt) {
    if (await page.getByTestId('results').isVisible()) {
      results = true
      break
    }
    if (checks++ % 5 === 0) await checkDom()
    await sampleDom(page, 0)
    await actIfMyTurn(page, 'call')
    await page.waitForTimeout(60)
  }
  expect(results).toBe(true)

  const live = frames.filter((f) => f.type === 'snapshot' && f.status !== 'complete')
  expect(live.length).toBeGreaterThan(0)
  for (const f of live) {
    for (const s of f.setup.seats.filter((s: any) => s.kind === 'bot')) {
      expect(s.policy).toBeNull()
      expect(s.style ?? null).toBeNull()
      expect(s.label).toBe(`Player ${s.seat}`)
    }
    const text = JSON.stringify(f)
    for (const id of ids) expect(text).not.toContain(id)
  }
  for (const b of bodies.filter((b) => !b.includes('"status":"complete"'))) for (const id of ids) expect(b).not.toContain(id)

  const reveal = page.getByTestId('reveal')
  await expect(reveal).toBeVisible()
  await expect(page.getByTestId('reveal-score')).toContainText('You read 1 of 4 hidden players')
  await expect(page.getByTestId('reveal-seat-1')).toContainText('Rock')
  await expect(page.getByTestId('reveal-seat-1')).toContainText('Correct')
  await expect(page.getByTestId('reveal-seat-2')).toContainText('You guessed Calling station')
  await expect(page.getByTestId('reveal-seat-3')).toContainText('Calling station')
  await expect(reveal.locator('table.evidence').first()).toContainText('Hands played')
  for (const size of WIDTHS) {
    await page.setViewportSize(size)
    await noHorizontalOverflow(page)
  }
})

test('setup offers modes and styles; a mixed table hides only the chosen seat', async ({ page, request }) => {
  await page.goto('/')
  await page.getByRole('radio', { name: '3', exact: true }).click()
  await page.getByLabel('Seat 1 player').selectOption('style_rock')
  await page.getByLabel('Seat 2 player').selectOption('style_aggressive')
  await expect(page.getByTestId('seat-style-1')).not.toBeEmpty()
  await page.getByText('Mixed', { exact: true }).click()
  await page.getByLabel('Hide seat 2 style').check()
  await expect(page.getByTestId('seat-style-2')).toContainText('Player 2')
  await expect(page.getByRole('button', { name: 'Create table' })).toBeEnabled()
  await page.getByRole('button', { name: 'Create table' }).click()
  await expect(page.getByText('Waiting for players')).toBeVisible()
  const tableId = page.url().split('/t/')[1]
  const snap = await (await request.get(`${API}/tables/${tableId}`)).json()
  expect(snap.setup.styles.mode).toBe('mixed')
  expect(snap.setup.styles.hidden_seats).toEqual([2])
  expect(snap.setup.seats[1].label).toBe('Rock')
  expect(snap.setup.seats[2].label).toBe('Player 2')
  await expect(page.locator('.lobby-seats')).toContainText('Player 2')
  await expect(page.locator('.lobby-seats')).not.toContainText('Aggressive')

  await page.goto('/')
  await page.getByText('Random styles', { exact: true }).click()
  await expect(page.getByTestId('seat-style-2')).toContainText('picked in secret')
})

test('Learn page lists the path, starts a preset table and fits every width', async ({ page }) => {
  await page.goto('/#/learn')
  await expect(page.getByTestId('learn')).toBeVisible()
  for (let i = 1; i <= 5; i++) await expect(page.getByTestId(`learn-step-${i}`)).toBeVisible()
  await expect(page.getByTestId('style-card-rock')).toBeVisible()
  await expect(page.getByTestId('glossary')).toContainText('Side pot')
  await expect(page.getByTestId('learn-progress')).toContainText('0 of 5 steps done')
  for (const size of WIDTHS) {
    await page.setViewportSize(size)
    await noHorizontalOverflow(page)
  }

  await page.getByTestId('learn-step-1').getByRole('button', { name: 'Start this table' }).click()
  await expect(page.locator('.felt')).toBeVisible()
  await expect(page.locator('.felt')).toContainText('Calling station')
  await expect(page.getByTestId('learning-aids')).toBeVisible()
  await page.goto('/#/learn')
  await expect(page.getByTestId('learn-step-1')).toContainText('Started')
  await expect(page.getByTestId('learn-step-1').getByRole('link', { name: 'Resume' })).toBeVisible()
})

test('reviewer replay shows each bot’s reasoning note after the table', async ({ page, request }) => {
  const t = await createTable(request, [bot('style_rock'), bot('style_aggressive'), bot('style_station')], { hand_cap: 2 }, { seed: 'learn-notes' })
  const stopAt = Date.now() + 60_000
  while (Date.now() < stopAt) {
    const s = await (await request.get(`${API}/tables/${t.tableId}`)).json()
    if (s.status === 'complete') break
    await page.waitForTimeout(100)
  }
  await openSeat(page, t, null, { host: true, path: '/replay' })
  await page.getByRole('radio', { name: /Reviewer/ }).click()
  await expect(page.getByTestId('perspective-note')).toContainText('Every hole card')
  for (let i = 0; i < 60 && !(await page.getByTestId('bot-note').isVisible()); i++) {
    await page.getByRole('button', { name: /Next action/ }).click()
  }
  await expect(page.getByTestId('bot-note')).toContainText(/reasoning/)
  await page.getByRole('radio', { name: 'Spectator' }).click()
  await expect(page.getByTestId('bot-note')).toHaveCount(0)
})
