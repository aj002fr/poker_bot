import { useEffect, useRef, useState } from 'react'
import type { Action, Snapshot } from './types'

const BASE = '/api'

export class ApiFailure extends Error {
  status: number
  code: string
  details: unknown
  constructor(status: number, code: string, message: string, details?: unknown) {
    super(message)
    this.status = status
    this.code = code
    this.details = details
  }
}

async function request<T>(path: string, init: RequestInit & { token?: string | null } = {}): Promise<T> {
  const headers: Record<string, string> = { 'content-type': 'application/json' }
  if (init.token) headers.authorization = `Bearer ${init.token}`
  const res = await fetch(BASE + path, { ...init, headers })
  const body = await res.json().catch(() => ({}))
  if (!res.ok) {
    const err = body?.error ?? {}
    throw new ApiFailure(res.status, err.code ?? 'http_error', err.message ?? res.statusText, err.details)
  }
  return body as T
}

export interface Credentials {
  seat?: number
  seatToken?: string
  hostToken?: string
}

// Session storage is per tab, so two tabs can hold two different seats.
export function loadCredentials(tableId: string): Credentials {
  try {
    return JSON.parse(sessionStorage.getItem(`arena:${tableId}`) ?? '{}')
  } catch {
    return {}
  }
}

export function saveCredentials(tableId: string, creds: Credentials) {
  sessionStorage.setItem(`arena:${tableId}`, JSON.stringify(creds))
}

export interface CreateBody {
  config: Record<string, unknown>
  seats: { kind: 'human' | 'bot'; policy?: string | null }[]
}

export const api = {
  catalog: () => request<{ rulesets: string[]; policies: import('./types').Policy[] }>('/catalog'),
  resolve: (body: CreateBody) =>
    request<{ config: import('./types').ResolvedConfig; config_hash: string }>('/configs/resolve', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  create: (body: CreateBody) =>
    request<{ table_id: string; host_token: string; snapshot: Snapshot }>('/tables', {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  join: (tableId: string, seat?: number) =>
    request<{ seat: number; seat_token: string; snapshot: Snapshot }>(`/tables/${tableId}/join`, {
      method: 'POST',
      body: JSON.stringify(seat === undefined ? {} : { seat }),
    }),
  start: (tableId: string, hostToken: string) =>
    request<Snapshot>(`/tables/${tableId}/start`, { method: 'POST', token: hostToken }),
  snapshot: (tableId: string, token?: string | null) => request<Snapshot>(`/tables/${tableId}`, { token }),
  history: (tableId: string, token?: string | null) => request<unknown>(`/tables/${tableId}/history`, { token }),
  act: (tableId: string, token: string, envelope: ActionEnvelope) =>
    request<{ receipt: unknown; replayed: boolean; snapshot: Snapshot }>(`/tables/${tableId}/actions`, {
      method: 'POST',
      token,
      body: JSON.stringify(envelope),
    }),
}

export interface ActionEnvelope {
  action_id: string
  table_id: string
  hand_id: string
  decision_id: string
  expected_state_version: number
  action: Action
}

export function newActionId(): string {
  return `a_${crypto.randomUUID().replace(/-/g, '')}`
}

// Retries reuse the same action_id, so an ambiguous submission is resolved by
// the server's idempotency registry rather than applied twice.
/** Server-side pauses worth retrying with the same action_id (the server deduplicates it). */
const TRANSIENT = new Set(['storage_unavailable'])

export async function submitWithRetry(tableId: string, token: string, envelope: ActionEnvelope) {
  let lastError: unknown
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      return await api.act(tableId, token, envelope)
    } catch (err) {
      if (err instanceof ApiFailure && !TRANSIENT.has(err.code)) throw err
      lastError = err
      await new Promise((r) => setTimeout(r, 300 * (attempt + 1)))
    }
  }
  throw lastError
}

export type StreamStatus = 'connecting' | 'live' | 'reconnecting' | 'denied'

/** Keeps the newest committed snapshot for this tab's credential, over WebSocket with HTTP fallback. */
export function useTableStream(tableId: string, token: string | null) {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [status, setStatus] = useState<StreamStatus>('connecting')
  const [error, setError] = useState<ApiFailure | null>(null)
  const versionRef = useRef(-1)

  const accept = (snap: Snapshot) => {
    if (snap.version > versionRef.current) {
      versionRef.current = snap.version
      setSnapshot(snap)
    }
  }

  useEffect(() => {
    versionRef.current = -1
    setSnapshot(null)
    let closed = false
    let ws: WebSocket | null = null
    let retry = 0
    let timer: number | undefined

    api.snapshot(tableId, token).then(accept, (err) => err instanceof ApiFailure && setError(err))

    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      ws = new WebSocket(`${proto}://${location.host}${BASE}/tables/${tableId}/stream`)
      ws.onopen = () => {
        ws?.send(JSON.stringify({ type: 'auth', token }))
      }
      ws.onmessage = (ev) => {
        const frame = JSON.parse(ev.data)
        if (frame.type === 'snapshot') {
          retry = 0
          setStatus('live')
          setError(null)
          accept(frame as Snapshot)
        } else if (frame.type === 'error') {
          const code = frame.error?.code ?? 'denied'
          const status = TRANSIENT.has(code) || code === 'table_halted' ? 503 : 403
          setError(new ApiFailure(status, code, frame.error?.message ?? 'denied'))
          if (TRANSIENT.has(code)) {
            setStatus('reconnecting')
          } else {
            setStatus('denied')
            closed = true
          }
        }
      }
      ws.onclose = () => {
        if (closed) return
        setStatus('reconnecting')
        retry = Math.min(retry + 1, 5)
        timer = window.setTimeout(connect, 250 * 2 ** retry)
      }
    }
    connect()
    return () => {
      closed = true
      window.clearTimeout(timer)
      ws?.close()
    }
  }, [tableId, token])

  return { snapshot, status, error, accept }
}
