// Wire types mirroring arena/domain/observations.py and arena/api/manager.py.
// The browser renders these; it never derives game state on its own.

export type Street = 'preflop' | 'flop' | 'turn' | 'river'
export type PlayerStatus = 'active' | 'folded' | 'all_in' | 'eliminated'

export interface LegalActions {
  fold: boolean
  check: boolean
  call_amount: number | null
  can_raise: boolean
  min_full_raise_to: number | null
  max_raise_to: number | null
  short_all_in_raise_to: number | null
  all_in_raise_to: number | null
  pot_raise_to: number | null
  raise_blocked_reason: string | null
}

export interface PlayerView {
  seat: number
  stack: number
  street_committed: number
  hand_committed: number
  status: PlayerStatus
  shown_cards: string[] | null
}

export interface PotView {
  amount: number
  eligible: number[]
}

export interface PublicEvent {
  seq: number
  hand_id: string | null
  type: string
  data: Record<string, any>
}

export interface TableView {
  schema_version: number
  perspective: 'seat' | 'spectator'
  compatibility_status: 'practice_assumptions'
  table_id: string
  ruleset_id: string
  config_hash: string
  state_version: number
  hero_seat: number | null
  hole_cards: string[] | null
  seats: number[]
  initial_player_count: number
  surviving_player_count: number
  hand_id: string | null
  hand_number: number
  decision_id: string | null
  button_seat: number | null
  small_blind_seat: number | null
  big_blind_seat: number | null
  street: Street | null
  actor_seat: number | null
  board: string[]
  pot_total: number
  settled_pot: number
  current_bet_to: number
  pots: PotView[]
  players: PlayerView[]
  blinds: { small_blind: number; big_blind: number } | null
  next_level: { small_blind: number; big_blind: number; starts_at_hand: number } | null
  payouts: number[]
  hands_played: number
  hands_remaining: number
  hand_cap: number
  decision_budget: { bot_budget_ns: number; human_turn_ms: number }
  legal: LegalActions | null
  action_history: PublicEvent[]
  table_complete: boolean
  points: Record<string, number> | null
}

export interface SeatInfo {
  seat: number
  kind: 'human' | 'bot'
  policy: string | null
  label: string
  artifact_hash: string | null
  claimed: boolean
  you: boolean
}

export interface ResolvedConfig {
  ruleset_id: string
  initial_player_count: number
  starting_stack: number
  hand_cap: number
  hands_per_level: number
  blind_levels: [number, number][]
  payout_points: number[]
  human_turn_ms: number
  bot_budget_ns: number
  [key: string]: unknown
}

export interface Placement {
  seat: number
  positions: number[]
  points: number
  method: 'elimination' | 'winner' | 'icm'
  hand_number: number
}

export interface Results {
  placements: Placement[]
  points: Record<string, number>
  payouts: number[]
  hands_played: number
  hand_cap: number
  eliminations: { seat: number; hand_number: number; positions: number[]; points: number }[]
  fallbacks: Record<string, Record<string, number>>
}

export interface Snapshot {
  table_id: string
  status: 'waiting' | 'running' | 'complete'
  version: number
  compatibility_status: 'practice_assumptions'
  viewer: { role: 'seat' | 'host' | 'anonymous'; seat: number | null }
  setup: { config: ResolvedConfig; config_hash: string; seats: SeatInfo[] }
  view: TableView | null
  turn: {
    seat: number
    kind: 'human' | 'bot'
    decision_id: string
    turn_ms: number | null
    remaining_ms: number | null
  } | null
  last_hand: { hand_id: string; hand_number: number | null; events: PublicEvent[] } | null
  results: Results | null
}

export interface Policy {
  id: string
  label: string
  description: string
  artifact_hash: string
}

export type Action =
  | { type: 'fold' }
  | { type: 'check' }
  | { type: 'call' }
  | { type: 'raise_to'; amount: number }
