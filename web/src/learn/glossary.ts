// One-line definitions in our own words.
export const GLOSSARY: Record<string, { term: string; text: string }> = {
  button: { term: 'Button', text: 'The dealer seat, marked D. It acts last after the flop, which makes it the best seat.' },
  blinds: { term: 'Blinds', text: 'Forced bets posted by the two seats left of the button, so every pot has something to win.' },
  small_blind: { term: 'Small blind', text: 'The seat left of the button. Posts half a bet before the cards are dealt.' },
  big_blind: { term: 'Big blind', text: 'Two seats left of the button. Posts a full bet before the cards are dealt.' },
  position: { term: 'Position', text: 'Where you sit relative to the button. Acting later means you see more before you decide.' },
  utg: { term: 'Under the gun', text: 'The first seat to act before the flop, just left of the big blind.' },
  cutoff: { term: 'Cutoff', text: 'The seat just right of the button. The second-best seat for opening hands.' },
  pot_odds: { term: 'Pot odds', text: 'The price of a call compared with the pot you could win. Cheaper calls need less often to win.' },
  equity: { term: 'Equity', text: 'Your share of the pot if all the cards were dealt out right now, averaged over every way they could fall.' },
  outs: { term: 'Outs', text: 'Unseen cards that would improve your hand to one that is likely to win.' },
  kicker: { term: 'Kicker', text: 'Your highest unpaired card. It breaks ties between players with the same pair.' },
  side_pot: { term: 'Side pot', text: 'An extra pot for chips bet after a player went all in. The all-in player cannot win it.' },
  domination: { term: 'Domination', text: 'When your hand shares a card with a better hand, such as AJ against AK. You win far less often.' },
  stack_ratio: {
    term: 'Stack ratio',
    text: 'Your stack divided by the blinds: how many rounds you could sit out before the blinds use up your chips. The lower it is, the more you need to take chances.',
  },
  vpip: { term: 'Hands played', text: 'How often a player chooses to put chips in before the flop. High means loose, low means tight.' },
  pfr: { term: 'Raise first', text: 'How often a player raises before the flop. Players who raise a lot are aggressive.' },
  aggression: { term: 'Aggression', text: 'How often a player bets or raises compared with calling. Callers are passive, raisers are aggressive.' },
  cbet: { term: 'Continuation bet', text: 'A bet on the flop by the player who raised before the flop, whether or not the flop helped.' },
  check_raise: { term: 'Check-raise', text: 'Checking, then raising after someone bets. Often a sign of strength, sometimes a trap or a bluff.' },
  showdown: { term: 'Showdown', text: 'When the betting ends and the remaining players show their cards to decide who wins.' },
  gap_concept: {
    term: 'Gap concept',
    text: 'You need a stronger hand to call a raise than to open the pot yourself, because the raiser has already shown strength.',
  },
  hammer: { term: 'The Hammer', text: 'Moving all in over an aggressive raiser to stop them pushing you around.' },
  rope_a_dope: { term: 'Rope-a-dope', text: 'Checking and calling with a strong hand to let an aggressive player keep bluffing into you.' },
  value_bet: { term: 'Value bet', text: 'A bet made because worse hands will call it. The opposite of a bluff.' },
  bluff: { term: 'Bluff', text: 'A bet with a weak hand, hoping a better hand folds.' },
  sample_size: { term: 'Sample size', text: 'How many hands a stat is based on. A few hands can mislead; trust a stat more as the count grows.' },
  icm: { term: 'ICM', text: 'A way to turn chip stacks into a fair share of the prize when the table stops before one player has everything.' },
}

export type GlossaryKey = keyof typeof GLOSSARY
