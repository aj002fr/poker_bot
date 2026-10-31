import { expect, type APIRequestContext, type Page } from '@playwright/test'

export const API = 'http://127.0.0.1:8417'
export const OPERATOR = 'e2e-operator-token'

export type SeatSpec = { kind: 'human' } | { kind: 'bot'; policy: string }
export const human: SeatSpec = { kind: 'human' }
export const bot = (policy: string): SeatSpec => ({ kind: 'bot', policy })

export interface TableHandle {
  tableId: string
  hostToken: string
  seatTokens: Record<number, string>
}

const auth = (token?: string | null) => (token ? { authorization: `Bearer ${token}` } : {})

/** Creates, fills and starts a seeded table through the API (the browser part of each test starts after this). */
export async function createTable(
  request: APIRequestContext,
  seats: SeatSpec[],
  config: Record<string, unknown> = {},
  { seed = 'e2e', start = true, operator = false }: { seed?: string; start?: boolean; operator?: boolean } = {},
): Promise<TableHandle> {
  const res = await request.post(`${API}/tables`, {
    data: { config, seats, seed },
    headers: operator ? auth(OPERATOR) : {},
  })
  expect(res.status(), await res.text()).toBe(201)
  const { table_id: tableId, host_token: hostToken } = await res.json()
  const seatTokens: Record<number, string> = {}
  for (const [i, s] of seats.entries()) {
    if (s.kind !== 'human') continue
    const j = await request.post(`${API}/tables/${tableId}/join`, { data: { seat: i } })
    expect(j.status()).toBe(200)
    seatTokens[i] = (await j.json()).seat_token
  }
  if (start) {
    const st = await request.post(`${API}/tables/${tableId}/start`, { headers: auth(hostToken) })
    expect(st.status(), await st.text()).toBe(200)
  }
  return { tableId, hostToken, seatTokens }
}

/** Opens the table in ``page`` holding seat ``seat``'s credential (and the host's, when ``host``). */
export async function openSeat(page: Page, t: TableHandle, seat: number | null, { host = false, path = '' } = {}) {
  await page.goto('/')
  const creds: Record<string, unknown> = {}
  if (seat !== null) Object.assign(creds, { seat, seatToken: t.seatTokens[seat] })
  if (host) creds.hostToken = t.hostToken
  await page.evaluate(([id, c]) => sessionStorage.setItem(`arena:${id}`, JSON.stringify(c)), [t.tableId, creds] as const)
  await page.goto(`/#/t/${t.tableId}${path}`)
}

export async function advanceClock(request: APIRequestContext, ms: number) {
  const res = await request.post(`${API}/__test__/clock/advance`, { data: { ms } })
  expect(res.status()).toBe(200)
}

export async function snapshot(request: APIRequestContext, t: TableHandle, token?: string | null) {
  const res = await request.get(`${API}/tables/${t.tableId}`, { headers: auth(token) })
  expect(res.status()).toBe(200)
  return res.json()
}

export interface DomSample {
  seat: number | null
  hand: number | null
  feltCards: string[]
  lastHand: number | null
  lastCards: string[]
  allCards: string[]
}

export async function sampleDom(page: Page, seat: number | null): Promise<DomSample | null> {
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

type Ev = { hand_id: string | null; type: string; data: any }

export async function reviewerEvents(request: APIRequestContext, t: TableHandle): Promise<Ev[]> {
  const res = await request.get(`${API}/tables/${t.tableId}/history?perspective=reviewer`, { headers: auth(t.hostToken) })
  expect(res.status()).toBe(200)
  return (await res.json()).events
}

/** Cards ``seat`` may see in hand ``hand``: its own hole cards, the board, and anything revealed. */
export function allowedCards(events: Ev[], hand: number | null, seat: number | null) {
  const out = new Set<string>()
  if (hand === null) return out
  const id = `h_${String(hand).padStart(3, '0')}`
  for (const e of events.filter((e) => e.hand_id === id)) {
    if (e.type === 'HoleDealt' && e.data.seat === seat) out.add(e.data.card)
    if (e.type === 'BoardDealt' || e.type === 'CardsRevealed') for (const c of e.data.cards) out.add(c)
  }
  return out
}

export function assertNoLeaks(samples: DomSample[], events: Ev[]) {
  expect(samples.length).toBeGreaterThan(0)
  for (const s of samples) {
    const now = allowedCards(events, s.hand, s.seat)
    const prev = allowedCards(events, s.lastHand, s.seat)
    for (const c of s.feltCards) expect(now.has(c), `seat ${s.seat} hand ${s.hand} shows ${c}`).toBe(true)
    for (const c of s.lastCards) expect(prev.has(c), `seat ${s.seat} last hand ${s.lastHand} shows ${c}`).toBe(true)
    for (const c of s.allCards) expect(now.has(c) || prev.has(c), `seat ${s.seat} page shows ${c}`).toBe(true)
  }
}

export type Policy = 'call' | 'fold'

/** Acts once if this page's seat is to act. ``call`` checks or calls; ``fold`` folds when it can, else checks. */
export async function actIfMyTurn(page: Page, policy: Policy = 'call'): Promise<boolean> {
  const panel = page.locator('.action-live')
  if (!(await panel.isVisible())) return false
  const fold = panel.getByRole('button', { name: 'Fold', exact: true })
  const button =
    policy === 'fold' && (await fold.isEnabled().catch(() => false)) ? fold : panel.getByRole('button', { name: /^(Check|Call)/ })
  try {
    if (await button.isEnabled({ timeout: 300 })) {
      await button.click({ timeout: 1000 })
      return true
    }
  } catch {
    // The decision moved on while we looked; the next poll sees the new state.
  }
  return false
}

/** Plays every page's human seat until each shows the results screen, sampling the DOM as it goes. */
export async function playToResults(players: [Page, number][], { policy = 'call' as Policy, limitMs = 150_000 } = {}) {
  const samples: DomSample[] = []
  let acted = 0
  const stopAt = Date.now() + limitMs
  while (Date.now() < stopAt) {
    let finished = 0
    for (const [page, seat] of players) {
      if (await page.getByTestId('results').isVisible()) {
        finished++
        continue
      }
      const s = await sampleDom(page, seat)
      if (s) samples.push(s)
      if (await actIfMyTurn(page, policy)) acted++
    }
    if (finished === players.length) break
    await players[0][0].waitForTimeout(60)
  }
  for (const [page] of players) await expect(page.getByTestId('results')).toBeVisible()
  return { samples, acted }
}

export async function noHorizontalOverflow(page: Page) {
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)
  expect(overflow).toBeLessThanOrEqual(0)
}

export async function tabTo(page: Page, target: ReturnType<Page['locator']>, max = 80) {
  for (let i = 0; i < max; i++) {
    if (await target.evaluate((el) => el === document.activeElement).catch(() => false)) return
    await page.keyboard.press('Tab')
  }
  throw new Error(`could not reach ${target} with Tab`)
}
