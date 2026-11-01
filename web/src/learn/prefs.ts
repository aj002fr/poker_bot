import { useEffect, useState } from 'react'

// Learning preferences live in this browser only; the server never sees them.
const EVENT = 'arena:learn-changed'

function read<T>(key: string, fallback: T): T {
  try {
    const raw = localStorage.getItem(key)
    return raw === null ? fallback : (JSON.parse(raw) as T)
  } catch {
    return fallback
  }
}

function write(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value))
  } catch {
    // Private mode or a full quota: the preference just does not persist.
  }
  window.dispatchEvent(new CustomEvent(EVENT, { detail: key }))
}

export function useStored<T>(key: string, fallback: T): [T, (v: T) => void] {
  const [value, setValue] = useState<T>(() => read(key, fallback))
  useEffect(() => {
    setValue(read(key, fallback))
    const update = (e: Event) => {
      if (e instanceof StorageEvent ? e.key === key : (e as CustomEvent).detail === key) setValue(read(key, fallback))
    }
    window.addEventListener(EVENT, update)
    window.addEventListener('storage', update)
    return () => {
      window.removeEventListener(EVENT, update)
      window.removeEventListener('storage', update)
    }
  }, [key])
  return [value, (v: T) => write(key, v)]
}

export const aidsKey = (tableId: string) => `arena:learn:aids:${tableId}`
export const guessesKey = (tableId: string) => `arena:learn:guesses:${tableId}`
export const COACH_KEY = 'arena:learn:coach'
export const SEEN_POSITIONS_KEY = 'arena:learn:seen-positions'
export const PROGRESS_KEY = 'arena:learn:progress'
export const LEARN_TABLES_KEY = 'arena:learn:tables'

export const readStored = read
export const writeStored = write
