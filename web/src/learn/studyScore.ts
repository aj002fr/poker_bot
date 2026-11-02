import type { StudyDecision, StudyOption } from '../types'
import { STUDY_SCORE_KEY, readStored, useStored, writeStored } from './prefs'

export interface StudyAnswer {
  pick: string
  action: boolean
  /** Null unless both you and Pluribus raised. */
  size: boolean | null
}

export interface StudyScore {
  /** Hand id → decision index → your latest answer. Re-studying a hand replaces its answers. */
  hands: Record<string, Record<string, StudyAnswer>>
}

const EMPTY: StudyScore = { hands: {} }

export const useStudyScore = () => useStored<StudyScore>(STUDY_SCORE_KEY, EMPTY)

export function recordAnswer(handId: string, decision: number, answer: StudyAnswer) {
  const s = readStored<StudyScore>(STUDY_SCORE_KEY, EMPTY)
  writeStored(STUDY_SCORE_KEY, { hands: { ...s.hands, [handId]: { ...s.hands[handId], [decision]: answer } } })
}

/** Action agreement is the kind (fold, check, call, raise); size agreement needs the same menu size. */
export function judge(d: StudyDecision, pick: StudyOption): StudyAnswer {
  const action = pick.kind === d.pluribus.kind
  return { pick: pick.key, action, size: action && pick.kind === 'raise' ? pick.key === d.pluribus.option : null }
}

export const resetStudyScore = () => writeStored(STUDY_SCORE_KEY, EMPTY)

export function tally(answers: StudyAnswer[]) {
  const sized = answers.filter((a) => a.size !== null)
  return {
    decisions: answers.length,
    action: answers.filter((a) => a.action).length,
    sized: sized.length,
    size: sized.filter((a) => a.size).length,
  }
}

export function totals(score: StudyScore) {
  const hands = Object.values(score.hands)
  return { ...tally(hands.flatMap((h) => Object.values(h))), hands: hands.length }
}
