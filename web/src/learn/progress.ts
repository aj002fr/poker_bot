import { LEARN_TABLES_KEY, PROGRESS_KEY, readStored, useStored, writeStored } from './prefs'

export interface Progress {
  /** Learn step id → the table started for it and whether it was finished. */
  steps: Record<string, { tableId: string; done: boolean; score?: string }>
  /** Style ids you have finished ahead of at a completed table. */
  beaten: string[]
}

const EMPTY: Progress = { steps: {}, beaten: [] }

export const useProgress = () => useStored<Progress>(PROGRESS_KEY, EMPTY)

export function rememberLearnTable(tableId: string, step: string) {
  const p = readStored<Progress>(PROGRESS_KEY, EMPTY)
  writeStored(PROGRESS_KEY, { ...p, steps: { ...p.steps, [step]: { tableId, done: false } } })
  writeStored(LEARN_TABLES_KEY, { ...readStored<Record<string, string>>(LEARN_TABLES_KEY, {}), [tableId]: step })
}

/** Called once a table completes: marks its Learn step done and records the styles you beat. */
export function recordFinish(tableId: string, beaten: string[], score?: string) {
  const p = readStored<Progress>(PROGRESS_KEY, EMPTY)
  const step = readStored<Record<string, string>>(LEARN_TABLES_KEY, {})[tableId]
  const steps = step ? { ...p.steps, [step]: { tableId, done: true, score } } : p.steps
  const merged = [...new Set([...p.beaten, ...beaten])].sort()
  if (merged.length === p.beaten.length && steps === p.steps) return
  writeStored(PROGRESS_KEY, { steps, beaten: merged })
}
