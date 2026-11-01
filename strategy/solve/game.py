"""Generic extensive-form game interface and its dense compiled form.

A game is described once through the small `ExtensiveGame` protocol and then
compiled into a `GameTree`: flat numpy arrays in breadth-first order. Every
solver and evaluator in `strategy.solve` operates only on `GameTree`, so a
later stage can plug in a push/fold or abstracted hold'em subgame either by
implementing the protocol or by emitting the arrays directly.

Conventions
-----------
* Nodes are numbered breadth-first from the root (node 0). A node's children
  are contiguous: ``first_child[i] .. first_child[i] + num_children[i] - 1``,
  and every child has a larger index than its parent.
* Each decision node belongs to an information set. Information-set actions
  occupy a contiguous range of *slots* ``infoset_offset[I] ..
  infoset_offset[I + 1] - 1``; all per-action solver state (regrets, strategy
  sums, policies) is a flat float64 array indexed by slot.
* ``node_slot[c]`` is the slot of the action leading into child ``c`` (or -1
  when its parent is a chance node); ``node_chance_prob[c]`` is the chance
  probability of that edge (1 for decision edges).
* Utilities are stored once per terminal in ``terminal_utility[T, P]``.
* Perfect recall is checked during compilation: every node of an information
  set must share the acting player's previous (infoset, action) slot.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any, Generic, Protocol, TypeVar

import numpy as np

CHANCE = -1
TERMINAL = -2

S = TypeVar("S")


@dataclass(frozen=True, slots=True)
class Terminal:
    utility: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Chance(Generic[S]):
    """Chance outcomes as ``(label, probability, child_state)``."""

    outcomes: tuple[tuple[Hashable, float, S], ...]


@dataclass(frozen=True, slots=True)
class Decision(Generic[S]):
    """A player decision; ``infoset`` is any hashable information-set key."""

    player: int
    infoset: Hashable
    actions: tuple[tuple[Hashable, S], ...]


class ExtensiveGame(Protocol[S]):
    name: str
    num_players: int
    player_names: tuple[str, ...]

    def root(self) -> S: ...

    def expand(self, state: S) -> Terminal | Chance[S] | Decision[S]: ...


class GameDefinitionError(ValueError):
    pass


@dataclass(frozen=True, eq=False)
class GameTree:
    name: str
    num_players: int
    player_names: tuple[str, ...]
    node_player: np.ndarray
    node_infoset: np.ndarray
    node_parent: np.ndarray
    node_depth: np.ndarray
    first_child: np.ndarray
    num_children: np.ndarray
    node_slot: np.ndarray
    node_chance_prob: np.ndarray
    node_chance_cdf: np.ndarray
    node_terminal: np.ndarray
    terminal_utility: np.ndarray
    infoset_player: np.ndarray
    infoset_offset: np.ndarray
    infoset_level: np.ndarray
    slot_infoset: np.ndarray
    infoset_keys: tuple[Hashable, ...]
    action_labels: tuple[tuple[Hashable, ...], ...]
    key_to_infoset: dict[Hashable, int] = field(repr=False)
    meta: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def num_nodes(self) -> int:
        return int(self.node_player.shape[0])

    @property
    def num_infosets(self) -> int:
        return int(self.infoset_player.shape[0])

    @property
    def num_slots(self) -> int:
        return int(self.infoset_offset[-1])

    @property
    def max_depth(self) -> int:
        return int(self.node_depth.max())

    @property
    def max_actions(self) -> int:
        return int(self.num_children.max())

    @property
    def is_zero_sum(self) -> bool:
        return bool(np.allclose(self.terminal_utility.sum(axis=1), 0.0, atol=1e-9))

    def infoset(self, key: Hashable) -> int:
        return self.key_to_infoset[key]

    def slots(self, infoset: int) -> slice:
        return slice(int(self.infoset_offset[infoset]), int(self.infoset_offset[infoset + 1]))

    def player_slot_mask(self, player: int) -> np.ndarray:
        return self.infoset_player[self.slot_infoset] == player

    def kernel_args(self) -> tuple[np.ndarray, ...]:
        """Arrays in the order the numba kernels expect them."""
        return (
            self.node_player,
            self.node_infoset,
            self.node_parent,
            self.first_child,
            self.num_children,
            self.node_slot,
            self.node_chance_prob,
            self.node_chance_cdf,
            self.node_terminal,
            self.terminal_utility,
            self.infoset_offset,
        )


def compile_game(game: ExtensiveGame[Any], *, max_nodes: int = 20_000_000) -> GameTree:
    """Walk ``game`` breadth-first and return its dense `GameTree`."""
    num_players = game.num_players
    player: list[int] = []
    infoset: list[int] = []
    parent: list[int] = []
    depth: list[int] = []
    slot: list[int] = []
    chance_prob: list[float] = []
    chance_cdf: list[float] = []
    terminal: list[int] = []
    utilities: list[tuple[float, ...]] = []
    first_child: list[int] = []
    num_children: list[int] = []
    # Per node, per player: the slot of that player's most recent own action.
    own_prev: list[tuple[int, ...]] = []
    own_level: list[tuple[int, ...]] = []

    key_to_infoset: dict[Hashable, int] = {}
    infoset_keys: list[Hashable] = []
    infoset_player: list[int] = []
    infoset_actions: list[tuple[Hashable, ...]] = []
    infoset_offset: list[int] = [0]
    infoset_prev: list[int] = []
    infoset_level: list[int] = []

    states: list[Any] = [game.root()]
    player.append(0)
    infoset.append(-1)
    parent.append(-1)
    depth.append(0)
    slot.append(-1)
    chance_prob.append(1.0)
    chance_cdf.append(1.0)
    own_prev.append(tuple([-1] * num_players))
    own_level.append(tuple([0] * num_players))

    head = 0
    while head < len(states):
        state = states[head]
        states[head] = None
        node = game.expand(state)
        base = len(states)
        if isinstance(node, Terminal):
            if len(node.utility) != num_players:
                raise GameDefinitionError(f"terminal utility has {len(node.utility)} entries, expected {num_players}")
            player[head] = TERMINAL
            terminal.append(len(utilities))
            utilities.append(tuple(float(u) for u in node.utility))
            first_child.append(base)
            num_children.append(0)
        elif isinstance(node, Chance):
            if not node.outcomes:
                raise GameDefinitionError("chance node without outcomes")
            total = sum(p for _, p, _ in node.outcomes)
            if abs(total - 1.0) > 1e-9:
                raise GameDefinitionError(f"chance probabilities sum to {total}")
            player[head] = CHANCE
            terminal.append(-1)
            first_child.append(base)
            num_children.append(len(node.outcomes))
            cumulative = 0.0
            for _, prob, child in node.outcomes:
                if prob <= 0.0:
                    raise GameDefinitionError("chance outcomes must have positive probability")
                cumulative += prob
                states.append(child)
                player.append(0)
                infoset.append(-1)
                parent.append(head)
                depth.append(depth[head] + 1)
                slot.append(-1)
                chance_prob.append(float(prob))
                chance_cdf.append(cumulative)
                own_prev.append(own_prev[head])
                own_level.append(own_level[head])
            chance_cdf[-1] = 1.0
        elif isinstance(node, Decision):
            p = node.player
            if not 0 <= p < num_players:
                raise GameDefinitionError(f"invalid player {p}")
            if not node.actions:
                raise GameDefinitionError("decision node without actions")
            labels = tuple(label for label, _ in node.actions)
            iset = key_to_infoset.get(node.infoset)
            if iset is None:
                iset = len(infoset_keys)
                key_to_infoset[node.infoset] = iset
                infoset_keys.append(node.infoset)
                infoset_player.append(p)
                infoset_actions.append(labels)
                infoset_offset.append(infoset_offset[-1] + len(labels))
                infoset_prev.append(own_prev[head][p])
                infoset_level.append(own_level[head][p])
            else:
                if infoset_player[iset] != p or infoset_actions[iset] != labels:
                    raise GameDefinitionError(f"information set {node.infoset!r} has inconsistent player or actions")
                if infoset_prev[iset] != own_prev[head][p]:
                    raise GameDefinitionError(f"imperfect recall at information set {node.infoset!r}")
            player[head] = p
            infoset[head] = iset
            terminal.append(-1)
            first_child.append(base)
            num_children.append(len(labels))
            offset = infoset_offset[iset]
            for a, (_, child) in enumerate(node.actions):
                states.append(child)
                player.append(0)
                infoset.append(-1)
                parent.append(head)
                depth.append(depth[head] + 1)
                slot.append(offset + a)
                chance_prob.append(1.0)
                chance_cdf.append(1.0)
                prev = list(own_prev[head])
                prev[p] = offset + a
                own_prev.append(tuple(prev))
                level = list(own_level[head])
                level[p] += 1
                own_level.append(tuple(level))
        else:
            raise GameDefinitionError(f"expand() returned {type(node).__name__}")
        if len(states) > max_nodes:
            raise GameDefinitionError(f"game exceeds {max_nodes} nodes")
        head += 1

    num_slots = infoset_offset[-1]
    slot_infoset = np.empty(num_slots, dtype=np.int32)
    for i in range(len(infoset_keys)):
        slot_infoset[infoset_offset[i] : infoset_offset[i + 1]] = i

    term = np.asarray(utilities, dtype=np.float64).reshape(len(utilities), num_players)
    return GameTree(
        name=game.name,
        num_players=num_players,
        player_names=tuple(game.player_names),
        node_player=np.asarray(player, dtype=np.int32),
        node_infoset=np.asarray(infoset, dtype=np.int32),
        node_parent=np.asarray(parent, dtype=np.int32),
        node_depth=np.asarray(depth, dtype=np.int32),
        first_child=np.asarray(first_child, dtype=np.int32),
        num_children=np.asarray(num_children, dtype=np.int32),
        node_slot=np.asarray(slot, dtype=np.int32),
        node_chance_prob=np.asarray(chance_prob, dtype=np.float64),
        node_chance_cdf=np.asarray(chance_cdf, dtype=np.float64),
        node_terminal=np.asarray(terminal, dtype=np.int32),
        terminal_utility=term,
        infoset_player=np.asarray(infoset_player, dtype=np.int32),
        infoset_offset=np.asarray(infoset_offset, dtype=np.int32),
        infoset_level=np.asarray(infoset_level, dtype=np.int32),
        slot_infoset=slot_infoset,
        infoset_keys=tuple(infoset_keys),
        action_labels=tuple(infoset_actions),
        key_to_infoset=key_to_infoset,
        meta=dict(getattr(game, "meta", {}) or {}),
    )
