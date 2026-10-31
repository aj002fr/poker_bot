import { fileURLToPath } from 'node:url'
import { expect, test, type Page } from '@playwright/test'
import { API, OPERATOR } from './support'

const TEMPLATE_BOT = fileURLToPath(new URL('../../examples/template_bot.py', import.meta.url))

const pickRuleset = (page: Page, name: string) => page.locator('label.option').filter({ hasText: name }).click()

async function signIn(page: Page) {
  await page.goto('/')
  await page.evaluate((token) => localStorage.setItem('arena:operator', token), OPERATOR)
}

test('runs, reports and registered files are operator-only', async ({ page, request }) => {
  expect((await request.get(`${API}/runs`)).status()).toBe(401)
  expect((await request.post(`${API}/runs`, { data: { cells: [], bots: [] } })).status()).toBe(401)
  expect((await request.post(`${API}/bots`, { data: { path: TEMPLATE_BOT } })).status()).toBe(401)

  await page.goto('/#/runs')
  await expect(page.getByRole('heading', { name: 'Operator only' })).toBeVisible()
  await page.goto('/#/runs/r_anything')
  await expect(page.getByRole('heading', { name: 'Operator only' })).toBeVisible()
  await page.goto('/')
  const seat = page.getByRole('combobox', { name: 'Seat 2 player' })
  await expect(seat.locator('optgroup[label="Baseline bots"]')).toHaveCount(1)
  await expect(seat.locator('optgroup[label="Registered files (operator)"]')).toHaveCount(0)
  await expect(page.getByText('Operators can also seat registered bot files.')).toBeVisible()
})

test('setup filters bots by ruleset and offers registered files to the operator', async ({ page }) => {
  await signIn(page)
  await page.goto('/#/runs')
  await page.getByRole('textbox', { name: 'Bot file path' }).fill(TEMPLATE_BOT)
  await page.getByRole('button', { name: 'Register' }).click()
  await expect(page.getByText(/^Registered template_bot \(file_[0-9a-f]{12}\) · NLHE, Short deck, PLO4$/)).toBeVisible()

  await page.goto('/')
  const seat = page.getByRole('combobox', { name: 'Seat 2 player' })
  const registered = seat.locator('optgroup[label="Registered files (operator)"] option')
  await expect(registered).toHaveCount(1)
  await expect(registered).toHaveText('Bot · template_bot')
  const heuristic = seat.locator('option[value="nlhe_heuristic"]')
  await expect(heuristic).toBeEnabled()

  await pickRuleset(page, 'Pot-limit Omaha')
  await expect(heuristic).toBeDisabled()
  await expect(heuristic).toHaveText('Bot · NLHE heuristic (NLHE only)')
  await expect(registered).toBeEnabled()
  await seat.selectOption({ label: 'Bot · template_bot' })
  await expect(page.locator('.resolved').getByText('template_bot · file')).toBeVisible()

  // A seat already holding an NLHE-only bot falls back when the ruleset changes.
  await pickRuleset(page, "No-limit Hold'em")
  await page.getByRole('combobox', { name: 'Seat 3 player' }).selectOption('nlhe_heuristic')
  await pickRuleset(page, 'Short-deck')
  await expect(page.getByRole('combobox', { name: 'Seat 3 player' })).toHaveValue('legal_random')
})

test('an operator launches a batch run and reads the report', async ({ page }) => {
  test.setTimeout(240_000)
  await page.setViewportSize({ width: 1440, height: 1000 })
  await signIn(page)
  await page.goto('/#/runs')
  await expect(page.getByRole('heading', { name: 'Batch benchmark' })).toBeVisible()

  await page.getByRole('checkbox', { name: "No-limit Hold'em, 6 seats" }).uncheck()
  await page.getByRole('checkbox', { name: 'Pot-limit Omaha, 3 seats' }).check()
  await page.getByLabel('NLHE heuristic').check()
  await expect(page.getByText('NLHE heuristic cannot play PLO4')).toBeVisible()
  await expect(page.getByRole('button', { name: /^Launch/ })).toBeDisabled()
  await page.getByLabel('NLHE heuristic').uncheck()

  await page.getByRole('spinbutton', { name: 'Seed blocks' }).fill('2')
  await page.getByRole('spinbutton', { name: 'Hand cap' }).fill('4')
  await page.getByRole('textbox', { name: /Base seed/ }).fill('e2e-batch')
  const launch = page.getByRole('button', { name: 'Launch 10 tables' })
  await expect(launch).toBeEnabled()
  await launch.click()

  await expect(page.getByTestId('run-report')).toBeVisible()
  const runId = page.url().split('/runs/')[1]
  await expect(page.getByTestId('run-status').first()).toHaveText('complete', { timeout: 200_000 })
  await expect(page.getByTestId('run-progress')).toHaveText('10/10 tables')

  const overview = page.getByTestId('overview')
  await expect(overview.locator('thead th')).toHaveText(['Bot', 'NLHE · 2', 'PLO4 · 3'])
  await expect(overview.locator('tbody tr')).toHaveCount(3)

  for (const [id, tables] of [
    ['practice_nlhe-2p', '4/4'],
    ['uncertainty_plo4-3p', '6/6'],
  ]) {
    const cell = page.getByTestId(`cell-${id}`)
    await expect(cell.getByTestId('tables-completed')).toHaveText(tables)
    await expect(cell.getByTestId('tables-invalid')).toHaveText('0')
    await expect(cell.getByText('competitive · sandboxed')).toBeVisible()
    await expect(cell.locator('dd').first()).toHaveText(/^[0-9a-f]{64}$/)
    await expect(cell.locator('dd').nth(1)).toHaveText(/^[0-9a-f]{64}$/)
    const rows = cell.locator('.report-table tbody tr')
    expect(await rows.count()).toBeGreaterThan(1)
    for (const row of await rows.all()) {
      await expect(row.locator('td').nth(4)).toHaveText(/^-?\d+\.\d\d – -?\d+\.\d\d/)
      await expect(row.locator('td').nth(6)).toHaveText(/^\d+/)
    }
  }

  await page.goto('/#/runs')
  const listed = page.locator('.run-list li').filter({ hasText: runId })
  await expect(listed.getByTestId('run-status')).toHaveText('complete')
  await expect(listed).toContainText('10/10')
})
