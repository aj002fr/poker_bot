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
  source: 'baseline' | 'registered' | null
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
  source: 'baseline' | 'registered'
  rulesets: string[]
  state_mode: string
  bot_name: string
  alias_of: string | null
  original_path?: string
  registered_at?: string
}

export type Perspective = 'seat' | 'spectator' | 'reviewer'

export interface ReplayFrame {
  index: number
  kind: 'deal' | 'action' | 'hand_end' | 'complete'
  seq: number
  hand_number: number
  view: TableView
  hole_cards: Record<string, string[]> | null
}

export interface ReplayHand {
  hand_number: number
  hand_id: string | null
  frames: ReplayFrame[]
  events: PublicEvent[]
}

export interface Replay {
  table_id: string
  perspective: Perspective
  viewer_seat: number | null
  table_complete: boolean
  available_perspectives: Perspective[]
  hands: { hand_number: number; hand_id: string | null; frames: number }[]
  hand: ReplayHand
}

export interface Quantiles {
  p50: number | null
  p95: number | null
  p99: number | null
  max: number | null
}

export interface EntrantSummary {
  entrant: number
  name: string
  bot_name: string
  state_mode: string | null
  builtin: boolean
  tables: number
  mean_points: number | null
  points_ci95: [number, number] | null
  decisions: number
  fallbacks: Record<string, number>
  fallback_total: number
  timeout_fallbacks: number
  restarts: number
  timing: { samples: number; end_to_end_s: Quantiles; over_budget_samples: number; worker_wall_s: Quantiles; worker_cpu_s: Quantiles }
}

export interface CellManifest {
  run_id: string
  config: ResolvedConfig
  config_hash: string
  content_hash: string
  competitive: boolean
  isolation: 'sandbox' | 'in_process'
  budget_ns: number
  seed_blocks: number
  rotations_per_block: number
  scheduled_tables: number
  seed_list_sha256: string
  report_note: string
  budget_semantics: string
  code: Record<string, unknown>
  entrants: { entrant: number; name: string; artifact_sha256: string; builtin: boolean; rulesets?: string[]; state_mode?: string }[]
  mean_points_by_seat: Record<string, number>
  summary: {
    tables_scheduled: number
    tables_completed: number
    tables_failed: { table_id: string; error: string }[]
    tables_incomplete: string[]
    blocks: number
    entrants: EntrantSummary[]
    table_wall_s_mean: number | null
  }
}

export type RunStatus = 'queued' | 'running' | 'complete' | 'failed'

export interface RunCellView {
  cell_id: string
  ruleset_id: string
  players: number
  config_hash: string
  entrants: { name: string; builtin: boolean }[]
  status: RunStatus
  done: number
  total: number
  wall_s: number | null
  error: string | null
  manifest: CellManifest | null
}

export interface RunView {
  run_id: string
  status: RunStatus
  created_at: string
  request: { cells: { ruleset_id: string; players: number }[]; bots: string[]; blocks: number; isolation: string; jobs: number; config: Record<string, unknown> }
  bots: Policy[]
  progress: { done: number; total: number }
  cells: RunCellView[]
  error: string | null
}

export interface RunBrief {
  run_id: string
  status: RunStatus
  created_at: string
  progress: { done: number; total: number }
  error: string | null
  scenarios: string[]
  bots: string[]
}

export type Action =
  | { type: 'fold' }
  | { type: 'check' }
  | { type: 'call' }
  | { type: 'raise_to'; amount: number }
