import { expect, test } from '@playwright/test'
import { actIfMyTurn, bot, createTable, human, openSeat } from './support'

// Opt-in: ARENA_SCREENSHOTS=/path/to/dir npx playwright test screenshots
const DIR = process.env.ARENA_SCREENSHOTS
test.skip(!DIR, 'set ARENA_SCREENSHOTS to a directory to capture walkthrough screenshots')

const shot = (name: string) => `${DIR}/${name}.png`

test('learning table screenshots', async ({ page, request }) => {
  await page.setViewportSize({ width: 1440, height: 1000 })

  const labelled = await createTable(
    request,
    [human, bot('style_rock'), bot('style_aggressive'), bot('style_station'), bot('style_weak_tight'), bot('style_super_aggressive')],
    { hand_cap: 30 },
    { seed: 'shots-labelled-3', styles: { mode: 'labelled', learning_aids: true } },
  )
  await openSeat(page, labelled, 0)
  await page.evaluate(() => localStorage.setItem('arena:learn:coach', 'true'))
  // Play a few hands so the stats badges and the recap have something to show.
  for (let hand = 0; hand < 60; hand++) {
    const n = Number(await page.getByTestId('hand-number').textContent())
    if (n >= 4 && (await page.locator('.action-live').isVisible())) break
    await actIfMyTurn(page, 'call')
    await page.waitForTimeout(80)
  }
  await page.reload()
  await expect(page.locator('.action-live')).toBeVisible()
  await expect(page.getByTestId('equity')).toBeVisible()
  await page.waitForTimeout(300)
  await page.screenshot({ path: shot('learning-table-labelled'), fullPage: true })

  const hidden = await createTable(
    request,
    [human, ...Array.from({ length: 5 }, () => bot('style_random'))],
    { hand_cap: 12 },
    { seed: 'shots-hidden-2', styles: { mode: 'random', learning_aids: true } },
  )
  await openSeat(page, hidden, 0)
  for (let i = 0; i < 400; i++) {
    const n = Number(await page.getByTestId('hand-number').textContent().catch(() => '0'))
    if (n >= 6 && (await page.locator('.action-live').isVisible())) break
    await actIfMyTurn(page, 'call')
    await page.waitForTimeout(60)
  }
  await page.getByLabel('Style guess for seat 1').selectOption({ index: 1 })
  await page.getByLabel('Style guess for seat 2').selectOption({ index: 4 })
  await page.getByLabel('Notes on seat 2').fill('Called every bet so far, never raised.')
  await page.getByTestId('notes-panel').scrollIntoViewIfNeeded()
  await page.screenshot({ path: shot('hidden-table-notes-and-guess'), fullPage: true })

  for (let i = 0; i < 3000 && !(await page.getByTestId('results').isVisible()); i++) {
    await actIfMyTurn(page, 'call')
    await page.waitForTimeout(40)
  }
  await expect(page.getByTestId('reveal')).toBeVisible()
  await page.getByTestId('reveal').scrollIntoViewIfNeeded()
  await page.screenshot({ path: shot('reveal-and-scoring'), fullPage: true })

  await page.goto('/#/learn')
  await expect(page.getByTestId('style-card-rock')).toBeVisible()
  await page.screenshot({ path: shot('learn-page'), fullPage: true })

  await page.setViewportSize({ width: 360, height: 780 })
  await openSeat(page, labelled, 0)
  await expect(page.getByTestId('learning-aids')).toBeVisible()
  await page.screenshot({ path: shot('learning-table-mobile'), fullPage: true })
})
