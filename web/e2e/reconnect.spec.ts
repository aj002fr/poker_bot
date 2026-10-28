import { expect, test, type APIRequestContext, type Page, type WebSocketRoute } from '@playwright/test'
import { API, advanceClock, bot, createTable, human, openSeat, type TableHandle } from './support'

const ACTIONS = '**/tables/*/actions'

/** Seat 0's committed actions, read from its own history (the reviewer log opens only after completion). */
async function heroEvents(request: APIRequestContext, t: TableHandle, type: 'ActionApplied' | 'ActionFallback') {
  const res = await request.get(`${API}/tables/${t.tableId}/history`, { headers: { authorization: `Bearer ${t.seatTokens[0]}` } })
  expect(res.status()).toBe(200)
  const events: { hand_id: string; type: string; data: any }[] = (await res.json()).events
  return events.filter((e) => e.type === type && e.data.seat === 0)
}
const heroActions = (request: APIRequestContext, t: TableHandle) => heroEvents(request, t, 'ActionApplied')

/** Clicks Check or Call, whichever this decision offers, and returns the label the panel uses for it. */
async function checkOrCall(page: Page) {
  const button = page.locator('.action-live').getByRole('button', { name: /^(Check|Call)/ })
  const label = (await button.textContent())!.startsWith('Check') ? 'Check' : 'Call'
  await button.click()
  return label
}

test('a submission whose response was lost is confirmed by its action id after a reload, not applied twice', async ({ page, request }) => {
  const t = await createTable(request, [human, bot('calling_station')], { hand_cap: 3 }, { seed: 'reconnect-committed' })
  let reachedServer = 0
  await page.route(ACTIONS, async (route) => {
    if (reachedServer++ === 0) await route.fetch()
    await route.abort('connectionreset')
  })
  await openSeat(page, t, 0)
  const label = await checkOrCall(page)
  await expect(page.getByTestId('pending')).toContainText(`Sending ${label}`)
  await expect(page.getByRole('alert')).toContainText(`Connection lost. ${label} is saved`)
  expect((await heroActions(request, t)).length).toBe(1)

  // The pending submission survives a reload of the tab.
  await page.reload()
  await expect(page.getByTestId('pending')).toContainText(`Sending ${label}`)
  await expect(page.locator('.action-live button').first()).toBeDisabled()

  await page.unroute(ACTIONS)
  await expect(page.getByTestId('notice')).toHaveText(`${label} was received before the connection dropped (confirmed by its action id).`)
  await expect(page.getByTestId('pending')).toHaveCount(0)
  const applied = await heroActions(request, t)
  expect(applied.filter((e) => e.hand_id === 'h_001' && e.data.street === 'preflop')).toHaveLength(1)
  expect(applied[0].data.action.type).toBe(label.toLowerCase())
})

test('a submission that never arrived is sent on reconnect when the stream comes back', async ({ page, request }) => {
  const t = await createTable(request, [human, bot('calling_station')], { hand_cap: 3 }, { seed: 'reconnect-lost' })
  const sockets: WebSocketRoute[] = []
  await page.routeWebSocket(/\/stream$/, (ws) => {
    ws.connectToServer()
    sockets.push(ws)
  })
  await page.route(ACTIONS, (route) => route.abort('internetdisconnected'))
  await openSeat(page, t, 0)
  await expect(page.getByTestId('stream-status')).toHaveText('live')

  const label = await checkOrCall(page)
  await expect(page.getByRole('alert')).toContainText(`Connection lost. ${label} is saved`)
  expect(await heroActions(request, t)).toHaveLength(0)

  await sockets.at(-1)!.close()
  await expect(page.getByTestId('stream-status')).toHaveText('reconnecting')
  await page.unroute(ACTIONS)
  await expect(page.getByTestId('stream-status')).toHaveText('live')
  expect(sockets.length).toBeGreaterThan(1)
  await expect(page.getByTestId('notice')).toHaveText(`${label} sent after reconnecting.`)
  const applied = await heroActions(request, t)
  expect(applied).toHaveLength(1)
  expect(await heroEvents(request, t, 'ActionFallback')).toHaveLength(0)
})

test('a submission held past the deadline is rejected and the timeout fallback stands', async ({ page, request }) => {
  const t = await createTable(request, [human, bot('calling_station')], { hand_cap: 3 }, { seed: 'reconnect-timeout' })
  await page.route(ACTIONS, (route) => route.abort('internetdisconnected'))
  await openSeat(page, t, 0)
  const label = await checkOrCall(page)
  await expect(page.getByRole('alert')).toContainText(`Connection lost. ${label} is saved`)

  await advanceClock(request, 20_001)
  await expect
    .poll(async () => (await heroEvents(request, t, 'ActionFallback')).map((e) => e.data.reason))
    .toEqual(['timeout'])

  await page.unroute(ACTIONS)
  await expect(page.getByRole('alert')).toHaveText(`${label} was not applied: the turn timed out first.`)
  await expect(page.getByTestId('pending')).toHaveCount(0)
  expect(await heroActions(request, t)).toHaveLength(1)
})
