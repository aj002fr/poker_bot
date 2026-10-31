import { expect, test } from '@playwright/test'
import { API, allowedCards, bot, createTable, human, openSeat, playToResults, reviewerEvents } from './support'

test('replay: perspectives follow permissions, controls step through actions and hands', async ({ page, request }) => {
  await page.setViewportSize({ width: 1440, height: 1000 })
  const t = await createTable(request, [human, bot('calling_station')], { hand_cap: 3, hands_per_level: 2 }, { seed: 'replay' })
  await openSeat(page, t, 0, { host: true })
  await expect(page.locator('.felt')).toBeVisible()

  // During the table: own seat and spectator only; the reviewer is closed until completion.
  await page.getByRole('link', { name: 'Replay so far' }).click()
  await expect(page.getByTestId('replay')).toBeVisible()
  const perspective = page.getByRole('radiogroup', { name: 'Perspective' })
  await expect(perspective.getByRole('radio', { name: 'Seat 0 (you)' })).toHaveAttribute('aria-checked', 'true')
  await expect(perspective.getByRole('radio', { name: 'Spectator' })).toBeEnabled()
  await expect(perspective.getByRole('radio', { name: /Reviewer/ })).toBeDisabled()
  const early = await request.get(`${API}/tables/${t.tableId}/replay?perspective=reviewer`, {
    headers: { authorization: `Bearer ${t.hostToken}` },
  })
  expect(early.status()).toBe(403)
  await page.getByRole('link', { name: 'Back to the table' }).click()

  await playToResults([[page, 0]])
  await page.getByRole('link', { name: 'Replay this table' }).click()
  await expect(page.getByTestId('replay')).toBeVisible()
  await expect(page.getByTestId('replay-hand')).toHaveText('1')
  await expect(page.getByTestId('replay-step')).toHaveText('1')

  const events = await reviewerEvents(request, t)
  const holes = (hand: number, seat: number) =>
    events.filter((e) => e.hand_id === `h_00${hand}` && e.type === 'HoleDealt' && e.data.seat === seat).map((e) => e.data.card)

  // Seat perspective: the opponent's cards stay hidden unless shown in that hand.
  const shownByOpponent = allowedCards(events, 1, 0)
  for (const card of await page.getByTestId('seat-1').locator('[data-card]').evaluateAll((els) => els.map((e) => e.getAttribute('data-card')))) {
    expect(shownByOpponent.has(card!)).toBe(true)
  }
  await expect(page.getByTestId('seat-0').locator('[data-card]')).toHaveCount(2)

  // Stepping.
  await page.getByRole('button', { name: 'Next action ▶' }).click()
  await expect(page.getByTestId('replay-step')).toHaveText('2')
  const logLines = await page.getByTestId('replay-log').locator('li').count()
  await page.getByRole('button', { name: '◀ Prev action' }).click()
  await expect(page.getByTestId('replay-step')).toHaveText('1')
  expect(await page.getByTestId('replay-log').locator('li').count()).toBeLessThan(logLines)
  await page.getByRole('button', { name: 'Next hand ⏭' }).click()
  await expect(page.getByTestId('replay-hand')).toHaveText('2')
  await page.getByRole('button', { name: '⏮ Prev hand' }).click()
  await expect(page.getByTestId('replay-hand')).toHaveText('1')
  await expect(page.getByRole('button', { name: '◀ Prev action' })).toBeDisabled()
  await expect(page.getByRole('button', { name: '⏮ Prev hand' })).toBeDisabled()

  // Play at 16×: steps advance by themselves until paused, crossing into later hands.
  const speeds = page.getByRole('radiogroup', { name: 'Playback speed' })
  await speeds.getByRole('radio', { name: '16×' }).click()
  await expect(speeds.getByRole('radio', { name: '16×' })).toHaveAttribute('aria-checked', 'true')
  await page.getByRole('button', { name: '▶ Play' }).click()
  await expect(page.getByTestId('replay-hand')).not.toHaveText('1', { timeout: 10_000 })
  await page.getByRole('button', { name: '❚❚ Pause' }).click()
  const pausedAt = `${await page.getByTestId('replay-hand').textContent()}:${await page.getByTestId('replay-step').textContent()}`
  await page.waitForTimeout(400)
  expect(`${await page.getByTestId('replay-hand').textContent()}:${await page.getByTestId('replay-step').textContent()}`).toBe(pausedAt)

  // Reviewer after completion sees every hole card; spectator sees none of the private ones.
  await page.getByRole('combobox', { name: 'Jump to hand' }).selectOption('1')
  await expect(page.getByTestId('replay-hand')).toHaveText('1')
  await perspective.getByRole('radio', { name: /Reviewer/ }).click()
  await expect(perspective.getByRole('radio', { name: /Reviewer/ })).toHaveAttribute('aria-checked', 'true')
  await expect(page.getByTestId('seat-1').locator('[data-card]')).toHaveCount(2)
  const reviewed = await page.getByTestId('seat-1').locator('[data-card]').evaluateAll((els) => els.map((e) => e.getAttribute('data-card')))
  expect(reviewed.sort()).toEqual(holes(1, 1).sort())

  await perspective.getByRole('radio', { name: 'Spectator' }).click()
  await expect(perspective.getByRole('radio', { name: 'Spectator' })).toHaveAttribute('aria-checked', 'true')
  await expect(page.getByTestId('seat-0').locator('[data-card]')).toHaveCount(0)
  await expect(page.getByTestId('seat-1').locator('[data-card]')).toHaveCount(0)

  // A call reads as the street total it reaches, like a raise, with the chips added alongside.
  const call = events.find((e) => e.type === 'ActionApplied' && e.data.action.type === 'call' && e.data.added !== e.data.street_committed)!
  expect(call).toBeTruthy()
  const fmt = (n: number) => n.toLocaleString('en-US')
  const callLine = `Seat ${call.data.seat} calls ${fmt(call.data.street_committed)} · adds ${fmt(call.data.added)}`
  await page.getByRole('combobox', { name: 'Jump to hand' }).selectOption(String(Number(call.hand_id.slice(2))))
  const log = page.getByTestId('replay-log')
  for (let i = 0; i < 60 && !(await log.getByText(callLine, { exact: true }).count()); i++) {
    await page.getByRole('button', { name: 'Next action ▶' }).click()
  }
  await expect(log.getByText(callLine, { exact: true })).toBeVisible()
})

test('replay without credentials offers the spectator view only', async ({ page, request }) => {
  const t = await createTable(request, [bot('calling_station'), bot('legal_random')], { hand_cap: 2 }, { seed: 'replay-anon' })
  await page.goto(`/#/t/${t.tableId}/replay`)
  const perspective = page.getByRole('radiogroup', { name: 'Perspective' })
  await expect(perspective.getByRole('radio')).toHaveCount(1)
  await expect(perspective.getByRole('radio', { name: 'Spectator' })).toHaveAttribute('aria-checked', 'true')
  await expect(page.getByTestId('perspective-note')).toHaveText('Public information only.')
})
