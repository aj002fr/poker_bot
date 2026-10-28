import { expect, test, type Page } from '@playwright/test'
import { API, OPERATOR, tabTo } from './support'

const focused = (page: Page) =>
  page.evaluate(() => {
    const el = document.activeElement as HTMLElement | null
    return { tag: el?.tagName ?? null, label: el?.getAttribute('aria-label') ?? el?.textContent?.trim() ?? null }
  })

async function seatHistory(page: Page, tableId: string) {
  const token = await page.evaluate((id) => JSON.parse(sessionStorage.getItem(`arena:${id}`)!).seatToken, tableId)
  const res = await page.request.get(`${API}/tables/${tableId}/history`, { headers: { authorization: `Bearer ${token}` } })
  expect(res.status()).toBe(200)
  return ((await res.json()).events as { type: string; data: any }[]).filter((e) => e.type === 'ActionApplied' && e.data.seat === 0)
}

test('setup, play, results and replay are reachable with the keyboard alone', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 })
  await page.goto('/')
  await expect(page.getByText('Resolved configuration')).toBeVisible()

  // Setup: two seats, a bot in seat 1, a 10-hand cap.
  const two = page.getByRole('radiogroup', { name: 'Seat count' }).getByRole('radio', { name: '2', exact: true })
  await tabTo(page, two)
  await page.keyboard.press('Space')
  await expect(two).toHaveAttribute('aria-checked', 'true')
  const seat1 = page.getByRole('combobox', { name: 'Seat 1 player' })
  await tabTo(page, seat1)
  await page.keyboard.press('ArrowDown')
  await expect(seat1).not.toHaveValue('human')
  const cap = page.getByRole('combobox', { name: 'Hand cap' })
  await tabTo(page, cap)
  await page.keyboard.press('Home')
  await expect(cap).toHaveValue('10')
  const create = page.getByRole('button', { name: 'Create table' })
  await expect(create).toBeEnabled()
  await tabTo(page, create)
  await page.keyboard.press('Enter')

  // Lobby.
  const start = page.getByRole('button', { name: 'Start table' })
  await expect(start).toBeEnabled()
  await tabTo(page, start)
  await page.keyboard.press('Enter')
  const tableId = page.url().split('/t/')[1]

  // Table: R focuses the raise amount, Esc leaves it, R + Enter raises at the minimum.
  const panel = page.locator('.action-live')
  await expect(panel).toBeVisible()
  await expect(panel.getByRole('button', { name: /^(Check|Call)/ })).toBeEnabled()
  await page.keyboard.press('r')
  expect((await focused(page)).tag).toBe('INPUT')
  await page.keyboard.press('Escape')
  expect((await focused(page)).tag).not.toBe('INPUT')
  await page.keyboard.press('r')
  await page.keyboard.press('Enter')
  await expect.poll(async () => (await seatHistory(page, tableId)).map((e) => e.data.action.type)).toEqual(['raise_to'])

  // C checks or calls for the rest of the table.
  const stopAt = Date.now() + 120_000
  while (!(await page.getByTestId('results').isVisible()) && Date.now() < stopAt) {
    if ((await panel.isVisible()) && (await panel.getByRole('button', { name: /^(Check|Call)/ }).isEnabled().catch(() => false))) {
      await page.keyboard.press('Escape')
      await page.keyboard.press('c')
    }
    await page.waitForTimeout(60)
  }
  await expect(page.getByTestId('results')).toBeVisible()
  const types = new Set((await seatHistory(page, tableId)).map((e) => e.data.action.type))
  expect(types.has('check') || types.has('call')).toBe(true)

  // Results → replay.
  const replayLink = page.getByRole('link', { name: 'Replay this table' })
  await tabTo(page, replayLink)
  await page.keyboard.press('Enter')
  await expect(page.getByTestId('replay')).toBeVisible()
  await expect(page.getByTestId('replay-step')).toHaveText('1')
  await page.keyboard.press('ArrowRight')
  await expect(page.getByTestId('replay-step')).toHaveText('2')
  await page.keyboard.press('ArrowLeft')
  await expect(page.getByTestId('replay-step')).toHaveText('1')
  await page.keyboard.press('Shift+ArrowRight')
  await expect(page.getByTestId('replay-hand')).toHaveText('2')
  await page.keyboard.press('Shift+ArrowLeft')
  await expect(page.getByTestId('replay-hand')).toHaveText('1')
  const speeds = page.getByRole('radiogroup', { name: 'Playback speed' })
  await page.keyboard.press('3')
  await expect(speeds.getByRole('radio', { name: '8×' })).toHaveAttribute('aria-checked', 'true')
  const play = page.getByRole('button', { name: /Play|Pause/ })
  await page.keyboard.press('p')
  await expect(play).toHaveAttribute('aria-pressed', 'true')
  await page.keyboard.press('p')
  await expect(play).toHaveAttribute('aria-pressed', 'false')

  const reviewer = page.getByRole('radiogroup', { name: 'Perspective' }).getByRole('radio', { name: /Reviewer/ })
  await tabTo(page, reviewer)
  await page.keyboard.press('Enter')
  await expect(reviewer).toHaveAttribute('aria-checked', 'true')
  await expect(page.getByTestId('seat-1').locator('[data-card]')).toHaveCount(2)
})

test('the operator dialog traps focus, closes on Escape and saves on Enter', async ({ page }) => {
  await page.goto('/')
  const open = page.getByRole('button', { name: 'Operator', exact: true })
  await tabTo(page, open)
  await page.keyboard.press('Enter')
  const dialog = page.getByRole('dialog', { name: 'Operator credential' })
  await expect(dialog).toBeVisible()
  await expect(page.getByLabel('Operator token')).toBeFocused()

  for (let i = 0; i < 8; i++) {
    await page.keyboard.press('Tab')
    const inside = await page.evaluate(() => !!document.activeElement?.closest('dialog'))
    // Native modal dialogs may hand focus to the browser chrome between cycles; it never lands on the page behind.
    const onPage = await page.evaluate(() => document.activeElement !== document.body && !document.activeElement?.closest('dialog'))
    expect(inside || !onPage).toBe(true)
  }

  await page.keyboard.press('Escape')
  await expect(dialog).toBeHidden()
  await expect(open).toBeFocused()

  await page.keyboard.press('Enter')
  await expect(page.getByLabel('Operator token')).toBeFocused()
  await page.keyboard.type('not-the-token')
  await page.keyboard.press('Enter')
  await expect(dialog.getByText('That is not this server’s operator token.')).toBeVisible()
  await page.keyboard.press('Control+A')
  await page.keyboard.type(OPERATOR)
  await page.keyboard.press('Enter')
  await expect(dialog).toBeHidden()
  await expect(page.getByRole('button', { name: 'Operator ✓' })).toBeVisible()
})
