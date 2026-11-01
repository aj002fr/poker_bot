import type { PositionGroup, PositionLabel } from '../types'

// Position labels and groups come from the server (GET /tables/{id}/aids); this is only their copy text.

export const POSITION_NAMES: Record<PositionLabel, string> = {
  UTG: 'Under the gun',
  MP: 'Middle position',
  CO: 'Cutoff',
  BTN: 'Button',
  SB: 'Small blind',
  BB: 'Big blind',
}

export const POSITION_TIPS: Record<PositionLabel, string> = {
  UTG: 'You act first before the flop, with everyone still to come behind you. Anyone could wake up with a big hand, so stick to your strongest starting hands here.',
  MP: 'Middle position. A few players act before you, so you can open a slightly wider set of hands than from the first seat, but several players still act after you.',
  CO: 'The cutoff, one seat right of the button. Only the button and the blinds are left behind you, so this is a good seat to open more hands and try to win the blinds.',
  BTN: 'The button acts last on every street after the flop. Seeing what everyone else does before you decide makes it the best seat at the table, so you can play your widest range here.',
  SB: 'The small blind has half a bet in already, but you act first after the flop, which is a real handicap. Do not call just because it looks cheap.',
  BB: 'The big blind has a full bet in. If nobody raises you can check and see the flop for free. You act early after the flop, so continue with care.',
}

export const GROUP_TEXT: Record<PositionGroup, string> = {
  early: 'early position',
  middle: 'middle position',
  late: 'late position',
  blinds: 'the blinds',
}
