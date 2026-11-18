"""What exists in the world.

Tables define what can exist:
  FIELDS            the values every thing has (one numpy array per field, per layer)
  LAYERS            the three places in every spot: block, cover, creature
  CONTACTS          who touches whom, so energy can flow between them
  DEFAULT_RECIPES   the recipe book: what each material is usually like

WorldState holds every thing's values as numpy arrays: that is the single source of truth. It
also keeps the energy books, so energy can only enter or leave the world through a named line
(start, sun, radiation, nudge). Entity is a read-only window onto one thing, found by its id.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

# Every thing, in every layer, has these values. Adding a property = adding a line here;
# arrays, Entity attributes, moving, copying and saving all follow automatically.
FIELDS: dict[str, type] = {
    # identity
    "kind": np.int16,  # index into the world's recipe book; 0 means the slot is empty
    "id": np.int64,  # unique per thing, travels with it, never reused; 0 means empty
    # energy: both count toward the conserved total
    "energy": np.float64,  # loose energy, which flows to and from neighbours
    "stored": np.float64,  # energy held in the store, released all at once when it pops
    # rolled per thing from its recipe when it is made, then fixed
    "capacity": np.float64,  # loose energy at which warmth reaches 1
    "conductivity": np.float64,  # how easily energy crosses its faces
    "mass": np.float64,  # 1.0 = the most the body can lift
    "store_size": np.float64,  # the most its store can hold; 0 = it has no store
    "release_at": np.float64,  # warmth at which its store pops (unused without a store)
    "soften_at": np.float64,  # warmth at which it goes soft; infinite = it never softens
    "tone": np.float64,  # visible: -1 darker .. +1 lighter than usual for its kind (random, a fingerprint)
    # changing state
    "soft": np.bool_,  # softened right now (it can weld to its neighbours)
}

ENERGY_FIELDS = ("energy", "stored")  # changed only by transfers, spawn/remove and nudges


@dataclass(frozen=True)
class Layer:
    """One of the three places in every spot.

    A spot can hold a block, a cover (like moss) and a creature (like the body) at the same time,
    but never two blocks. The rules here are data: support and movement laws read them instead
    of naming kinds of thing.
    """

    name: str
    supported_by: tuple[str, ...]  # layers it can rest on (in the spot directly below)
    blocks_movement: bool  # creatures can't walk through it
    shares_with: tuple[str, ...]  # layers that may be in the same spot at the same time


# Blocks and cover can rest on a creature: that is how the body holds something (on its head).
# Creatures stand on blocks only, never on each other. A block fills its spot completely; moss
# and a creature can share one (the body standing on moss).
LAYERS: dict[str, Layer] = {
    layer.name: layer
    for layer in [
        Layer("block", supported_by=("block", "creature"), blocks_movement=True, shares_with=()),
        Layer("cover", supported_by=("block", "creature"), blocks_movement=False, shares_with=("creature",)),
        Layer("creature", supported_by=("block",), blocks_movement=True, shares_with=("cover",)),
    ]
}


_SIDES = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]

# Which things touch, so energy can flow between them: (layer A, layer B, offset) means a thing
# in A at p touches a thing in B at p + offset. Every touching pair is listed exactly once.
# One table, shared by every law.
CONTACTS: list[tuple[str, str, tuple[int, int, int]]] = [
    # blocks touch blocks on all six faces
    ("block", "block", (1, 0, 0)),
    ("block", "block", (0, 1, 0)),
    ("block", "block", (0, 0, 1)),
    # moss touches neighbouring moss, the block under it, and the blocks beside it
    ("cover", "cover", (1, 0, 0)),
    ("cover", "cover", (0, 1, 0)),
    ("block", "cover", (0, 0, 1)),
    *[("cover", "block", side) for side in _SIDES],
    # the body touches blocks on every side: under its feet, on its head, and all around it
    ("block", "creature", (0, 0, 1)),
    ("creature", "block", (0, 0, 1)),
    *[("creature", "block", side) for side in _SIDES],
    # ...the moss it stands in, and moss carried on its head
    ("cover", "creature", (0, 0, 0)),
    ("creature", "cover", (0, 0, 1)),
    # creatures touch side by side
    ("creature", "creature", (1, 0, 0)),
    ("creature", "creature", (0, 1, 0)),
]


Spread = tuple[float, float]  # (usual value, give or take)

# The properties each thing rolls from its recipe. A recipe that doesn't list one of the optional
# ones gets its fixed "off" value instead, never a roll.
VARYING = ("mass", "capacity", "conductivity", "store_size", "release_at", "soften_at")
OFF: dict[str, float] = {"store_size": 0.0, "release_at": 0.0, "soften_at": math.inf}


@dataclass(frozen=True)
class Kind:
    """A material's recipe: which layer it lives in, and what its things are usually like.

    A kind is not a thing itself. Each varying property is a (usual value, give or take) pair,
    and every thing made from the recipe rolls its own values near it. That is how two sponges
    can differ slightly, so the only way to know a particular sponge is to test it.

    Materials without a feature simply don't list it: no store_size means no store, no
    soften_at means it never softens, no grow_at means it isn't alive. Laws then read the
    numbers and never name materials.
    """

    name: str
    layer: str
    mass: Spread
    capacity: Spread
    conductivity: Spread
    store_size: Spread | None = None  # has a store that fills and pops
    release_at: Spread | None = None  # ...and the warmth at which it pops (needed with a store)
    soften_at: Spread | None = None  # goes soft when this warm
    # fixed for the whole material, not rolled per thing
    refill_rate: float = 0.0  # loose energy moved into the store per tick, times warmth
    consumed_on_pop: bool = False  # popping uses it up: it disappears (burning)
    grow_at: float | None = None  # alive: spreads to a neighbouring spot when warmer than this
    grow_cost: float = 0.0  # ...a new patch takes this much energy from its parent (and starts with it)
    die_at: float = 0.0  # ...and it dies when colder than this
    color: str = "#888888"  # base look for the viewer

    def __post_init__(self):
        if self.layer not in LAYERS:
            raise ValueError(f"{self.name}: unknown layer {self.layer!r}")
        if (self.store_size is None) != (self.release_at is None):
            raise ValueError(f"{self.name}: a store needs both store_size and release_at")
        if self.consumed_on_pop and self.store_size is None:
            raise ValueError(f"{self.name}: only something with a store can be consumed when it pops")
        for prop in VARYING:
            spread = getattr(self, prop)
            if spread is not None and (spread[0] <= 0 or spread[1] < 0):
                raise ValueError(f"{self.name}: {prop} needs a positive usual value and a non-negative spread")
        if self.grow_at is not None:
            if self.store_size is None:
                raise ValueError(f"{self.name}: living things need a store (growth is paid from it)")
            # A new patch must survive (above die_at) but not spread straight away (below grow_at).
            newborn = self.grow_cost / self.capacity[0]
            if not 0 <= self.die_at < newborn < self.grow_at:
                raise ValueError(
                    f"{self.name}: needs die_at < grow_cost / usual capacity < grow_at "
                    f"(got {self.die_at} < {newborn:.3f} < {self.grow_at})"
                )

    @property
    def alive(self) -> bool:
        return self.grow_at is not None

    def sample(self, rng: np.random.Generator | None) -> dict[str, float]:
        """Roll one thing's properties. Without an rng, every thing is exactly usual.

        Only listed properties are rolled; unlisted ones get their OFF value. Rolls come from a
        normal distribution around the usual value, clipped so nothing ends up zero or negative
        (a zero capacity would make warmth undefined).
        """
        values = {}
        for prop in VARYING:
            spread = getattr(self, prop)
            if spread is None:
                values[prop] = OFF[prop]
                continue
            usual, give_or_take = spread
            value = usual if rng is None else rng.normal(usual, give_or_take)
            values[prop] = max(value, 0.1 * usual)
        return values

    @staticmethod
    def tone_for(rng: np.random.Generator | None) -> float:
        """How light or dark this thing looks compared with others of its kind.

        Purely random and fixed for life: a fingerprint that lets a thing be recognized again,
        but says nothing about its hidden properties. (Without an rng, every thing looks usual.)
        """
        return 0.0 if rng is None else float(rng.uniform(-1, 1))


# The default recipe book. Index 0 is reserved for "empty". Each world carries its own copy of a
# book (saved with it, frozen once the world is made); this is the one every world uses for now.
# Store, softening and refill numbers are first guesses, to be tuned when those laws exist.
DEFAULT_RECIPES: tuple[Kind | None, ...] = (
    None,
    Kind("bedrock", "block", mass=(1000.0, 0.0), capacity=(20.0, 0.0), conductivity=(0.03, 0.0), color="#2b2a33"),
    Kind("soil", "block", mass=(0.5, 0.1), capacity=(10.0, 2.0), conductivity=(0.05, 0.01), color="#8a5a3b"),
    Kind(
        "stone", "block", mass=(1.1, 0.3), capacity=(6.0, 1.0), conductivity=(0.15, 0.03),
        soften_at=(0.8, 0.05), color="#8d929b",
    ),
    Kind("sponge", "block", mass=(0.2, 0.05), capacity=(40.0, 8.0), conductivity=(0.01, 0.003), color="#e6dcb8"),
    # Crystal: charges up and pops (a spark), then recharges. Its pop can set off its neighbours.
    Kind(
        "crystal", "block", mass=(0.6, 0.1), capacity=(2.0, 0.4), conductivity=(0.12, 0.02),
        store_size=(2.5, 0.4), release_at=(0.42, 0.04), refill_rate=0.05, color="#9fd8e8",
    ),
    # Moss: alive. Spreads when well fed, dies when starving, slowly banks fuel, and burns
    # (pops and is used up) when something pushes it past its pop point.
    Kind(
        "moss", "cover", mass=(0.05, 0.01), capacity=(1.0, 0.2), conductivity=(0.02, 0.005),
        store_size=(1.3, 0.15), release_at=(0.74, 0.03), refill_rate=0.002, consumed_on_pop=True,
        grow_at=0.55, grow_cost=0.4, die_at=0.08, color="#6fa84a",
    ),
    # The body's low conductivity means touching ordinary ground only trickles energy in or out;
    # the sun is what really charges it.
    Kind("body", "creature", mass=(0.7, 0.0), capacity=(10.0, 0.0), conductivity=(0.005, 0.0), color="#f2a65a"),
)

Pos = tuple[int, int, int]  # (x, y, z), z points up

# The named lines in the energy books: the only ways energy enters or leaves the world.
#   start      energy given to things as the world is built
#   sun        sunlight soaking in
#   radiation  heat lost to the sky (recorded as a negative amount)
#   nudge      the experimenter adding or taking away energy by hand
OUTSIDE = ("start", "sun", "radiation", "nudge")

TOLERANCE = 1e-9  # how far below zero rounding may push an energy before it counts as a bug
SAVE_VERSION = 1  # bump when the save format changes, so old saves are refused instead of misread


def _as_pos(pos) -> Pos:
    """Normalize a position to plain Python ints (numpy integers sneak in from array math)."""
    x, y, z = pos
    return (int(x), int(y), int(z))


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------


class Entity:
    """A read-only window onto one thing, found by its id. It holds no data of its own.

    It follows the thing wherever it moves, because every read looks up where the thing is
    right now. If the thing no longer exists (moss that died), any read fails loudly with
    "#12 no longer exists"; code that expects things to vanish checks `exists` first.

    Every name in FIELDS is available as an attribute (entity.energy, entity.mass, ...); they are
    attached automatically below the class. They can't be written: energy only changes through
    WorldState.transfer and the other books-keeping methods, so it can't appear from nowhere.
    """

    __slots__ = ("_state", "_id")  # only these attributes exist, so typos raise errors

    # The FIELDS attributes, declared so editors know they exist. The real attributes are
    # attached from FIELDS below; a check there makes sure this list and FIELDS stay identical.
    kind: int
    id: int
    energy: float
    stored: float
    capacity: float
    conductivity: float
    mass: float
    store_size: float
    release_at: float
    soften_at: float
    tone: float
    soft: bool

    def __init__(self, state: WorldState, entity_id: int):
        self._state = state
        self._id = int(entity_id)

    @property
    def id(self) -> int:  # type: ignore[override]
        """Its unique id (readable even after it is gone)."""
        return self._id

    @property
    def exists(self) -> bool:
        """Whether it is still in the world."""
        return self._id in self._state.where

    @property
    def layer(self) -> str:
        return self._state.locate(self._id)[0]

    @property
    def pos(self) -> Pos:
        return self._state.locate(self._id)[1]

    def _read(self, field: str):
        """Read one field as a plain Python value, from wherever the thing is now."""
        layer, pos = self._state.locate(self._id)
        return self._state.layers[layer][field][pos].item()

    @property
    def kind_info(self) -> Kind:
        """The recipe it was made from (its layer, usual values, color), from its world's book."""
        return self._state.recipes[self.kind]

    @property
    def name(self) -> str:
        """The kind's name, e.g. "sponge"."""
        return self.kind_info.name

    @property
    def warmth(self) -> float:
        """How full of loose energy it is: 0 = cold, 1 = at capacity.

        Computed on every read rather than stored, so it can never be out of date.
        """
        return self.energy / self.capacity

    def __repr__(self) -> str:  # how it prints in the console
        if not self.exists:
            return f"Entity(#{self._id}, gone)"
        return (
            f"Entity(#{self._id} {self.name}, {self.layer} @ {self.pos}, "
            f"warmth {self.warmth:.2f}, energy {self.energy:.2f}, mass {self.mass:.2f})"
        )


def _field_property(field: str) -> property:
    """Make a read-only attribute for one field, e.g. entity.energy."""

    def write(self, value):
        raise AttributeError(
            f"{field} is read-only: change energy with WorldState.transfer, or the world by hand with a nudge"
        )

    return property(lambda self: self._read(field), write)


# Turn every FIELDS entry into an attribute (id is special: it lives on the Entity itself).
for _field in FIELDS:
    if _field != "id":
        setattr(Entity, _field, _field_property(_field))

# Adding a field to FIELDS also means declaring it on Entity; fail at startup if they drift apart.
if set(Entity.__annotations__) != set(FIELDS):
    raise TypeError(
        f"Entity's declared fields {sorted(Entity.__annotations__)} don't match FIELDS {sorted(FIELDS)}"
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

Target = int | Iterable[int]  # one id, or several
Account = int | str  # a thing's id, or a named line in the books ("start" or "nudge")


class WorldState:
    """All the world's values. Arrays are shaped (width, depth, height), one set per layer:

        state.layers["block"]["energy"][x, y, z]

    Besides the arrays it keeps:
      - an id index (`where`), so any thing can be found instantly wherever it has moved
      - the world's recipe book (`recipes`), frozen once the world is made
      - the energy books (`books`): total energy always equals the sum of the named lines
      - a log of every nudge made by hand

    WorldState stores and moves things and keeps the books; it knows nothing about physics.
    Laws read the arrays freely, and change energy only through move_energy (energy moving
    around inside) and exchange_with_outside (sun in, radiation out).
    """

    def __init__(self, shape: tuple[int, int, int], recipes: tuple[Kind | None, ...] = DEFAULT_RECIPES):
        self.shape: tuple[int, int, int] = _as_pos(shape)
        self.recipes = _checked_recipes(recipes)
        self.kind_index: dict[str, int] = {kind.name: i for i, kind in enumerate(self.recipes) if kind is not None}
        self.layers: dict[str, dict[str, np.ndarray]] = {
            layer: {field: np.zeros(self.shape, dtype) for field, dtype in FIELDS.items()} for layer in LAYERS
        }
        self.where: dict[int, tuple[str, Pos]] = {}  # thing id -> (layer, pos)
        self.next_id = 1
        self.books: dict[str, float] = {line: 0.0 for line in OUTSIDE}
        self.log: list[dict] = []  # every nudge, in order

    # --- looking things up ---

    def in_bounds(self, pos: Pos) -> bool:
        """Whether pos lies inside the grid."""
        return all(0 <= p < n for p, n in zip(pos, self.shape))

    def occupied(self, layer: str) -> np.ndarray:
        """Boolean array: which spots hold something in this layer."""
        return self.layers[layer]["kind"] != 0

    def entity_at(self, layer: str, pos: Pos) -> Entity | None:
        """A view of whatever is in this layer at pos, or None if the spot is empty or outside."""
        pos = _as_pos(pos)
        if not self.in_bounds(pos) or self.layers[layer]["kind"][pos] == 0:
            return None
        return Entity(self, int(self.layers[layer]["id"][pos]))

    def entity_by_id(self, entity_id: int) -> Entity | None:
        """A view of the thing with this id, or None if it no longer exists."""
        return Entity(self, entity_id) if entity_id in self.where else None

    def locate(self, entity_id: int) -> tuple[str, Pos]:
        """Where the thing with this id is: (layer, pos). Fails loudly if it no longer exists."""
        location = self.where.get(entity_id)
        if location is None:
            raise LookupError(f"#{entity_id} no longer exists")
        return location

    def recipe(self, kind_name: str) -> Kind:
        """The recipe for a material, from this world's book."""
        if kind_name not in self.kind_index:
            raise KeyError(f"no material called {kind_name!r} in this world's recipe book")
        return self.recipes[self.kind_index[kind_name]]  # type: ignore[return-value]

    def neighbours(self, entity_id: int) -> list[int]:
        """Ids of everything this thing touches (by the CONTACTS table), in a fixed order."""
        layer, (x, y, z) = self.locate(entity_id)
        found = []
        for layer_a, layer_b, (dx, dy, dz) in CONTACTS:
            if layer == layer_a and (other := self.entity_at(layer_b, (x + dx, y + dy, z + dz))):
                found.append(other.id)
            if layer == layer_b and (other := self.entity_at(layer_a, (x - dx, y - dy, z - dz))):
                found.append(other.id)
        return [i for i in dict.fromkeys(found) if i != entity_id]  # each once, never itself

    def has_room(self, layer: str, pos: Pos) -> bool:
        """Whether a thing of this layer could be put at pos: inside the world, its own place
        free, and nothing there it can't share a spot with."""
        pos = _as_pos(pos)
        if not self.in_bounds(pos) or self.layers[layer]["kind"][pos] != 0:
            return False
        return all(
            self.layers[other]["kind"][pos] == 0
            for other in LAYERS
            if other != layer and other not in LAYERS[layer].shares_with
        )

    def region(self, lo: Pos, hi: Pos, layers: Iterable[str] | None = None) -> list[int]:
        """Ids of everything in the box from corner lo to corner hi (both included)."""
        box = tuple(slice(max(0, min(a, b)), max(a, b) + 1) for a, b in zip(lo, hi))
        ids = []
        for layer in layers or LAYERS:
            found = self.layers[layer]["id"][box]
            ids.extend(int(i) for i in found[found != 0])
        return ids

    # --- the energy books ---

    def total_energy(self) -> float:
        """All energy in the world, loose and stored, across every layer."""
        return float(sum(arrays[f].sum() for arrays in self.layers.values() for f in ENERGY_FIELDS))

    def expected_energy(self) -> float:
        """What the books say the total should be: start + sun + radiation (negative) + nudges."""
        return sum(self.books.values())

    def check_books(self, tolerance: float = 1e-6) -> None:
        """Raise if the world holds a different amount of energy than the books say."""
        actual, expected = self.total_energy(), self.expected_energy()
        if abs(actual - expected) > tolerance * max(1.0, abs(expected)):
            raise AssertionError(f"energy books don't balance: world has {actual}, books expect {expected}")

    def transfer(self, source: int, dest: int, amount: float) -> None:
        """Move loose energy from one thing to another. A negative amount moves it the other way.

        Nothing is created or lost, so the books don't change. Fails loudly (and changes nothing)
        if either side would end up below zero.
        """
        if amount == 0:
            return
        (a_layer, a_pos), (b_layer, b_pos) = self.locate(source), self.locate(dest)
        a, b = self.layers[a_layer]["energy"], self.layers[b_layer]["energy"]
        if a[a_pos] - amount < -TOLERANCE or b[b_pos] + amount < -TOLERANCE:
            raise ValueError(f"transfer of {amount} between #{source} and #{dest} would leave one below zero")
        a[a_pos] -= amount
        b[b_pos] += amount

    def move_energy(self, changes: dict[tuple[str, str], np.ndarray]) -> None:
        """Apply energy changes that only move energy around inside the world (flow, stores
        filling, pops).

        `changes` maps (layer, field) to an array of changes, where field is "energy" (loose) or
        "stored". Together they must add up to zero, only touch occupied spots, and leave nothing
        below zero; anything else is a bug and fails here, before anything changes.
        """
        total = sum(float(d.sum()) for d in changes.values())
        size = sum(float(np.abs(d).sum()) for d in changes.values())
        if abs(total) > 1e-9 * max(1.0, size):
            raise AssertionError(f"energy moved inside the world doesn't add up to zero (off by {total})")
        for (layer, field), delta in changes.items():
            if field not in ENERGY_FIELDS:
                raise ValueError(f"move_energy only changes {ENERGY_FIELDS}, not {field!r}")
            if np.any(delta[~self.occupied(layer)] != 0):
                raise AssertionError(f"energy moved into empty {layer} spots")
            if np.any(self.layers[layer][field] + delta < -TOLERANCE):
                raise AssertionError(f"moving energy would leave some {layer} {field} below zero")
        for (layer, field), delta in changes.items():
            self.layers[layer][field] += delta

    def exchange_with_outside(self, layer: str, delta: np.ndarray, line: str) -> None:
        """Energy entering from outside the world (positive) or leaving it (negative), recorded
        under `line` in the books: "sun" for sunlight in, "radiation" for heat lost to the sky."""
        if line not in ("sun", "radiation"):
            raise ValueError(f"{line!r} isn't a natural line; spawn, remove and nudges handle the others")
        energy = self.layers[layer]["energy"]
        if np.any(delta[~self.occupied(layer)] != 0):
            raise AssertionError(f"{line} reached empty {layer} spots")
        if np.any(energy + delta < -TOLERANCE):
            raise AssertionError(f"{line} would leave some {layer} below zero energy")
        energy += delta
        self.books[line] += float(delta.sum())

    # --- changing what exists ---

    def spawn(
        self,
        kind_name: str,
        pos: Pos,
        *,
        source: Account,
        energy: float = 0.0,
        warmth: float | None = None,
        rng: np.random.Generator | None = None,
        **values,
    ) -> Entity:
        """Make a new thing in its kind's layer, and say where its starting energy comes from.

        `source` pays for the energy: another thing's id (moss growing pays from the parent),
        "start" (worldgen building the world), or "nudge" (the experimenter, logged). Give the
        energy directly, or as a warmth (energy = warmth * its rolled capacity).

        Properties are rolled from the recipe unless given in `values`. Used by worldgen, nudges,
        tests and (later) growth. Never by the body's actions: an agent moves matter, never makes it.
        """
        kind = self.recipe(kind_name)
        pos = _as_pos(pos)
        if not self.in_bounds(pos):
            raise ValueError(f"{pos} is outside the world")
        if self.layers[kind.layer]["kind"][pos] != 0:
            raise ValueError(f"{kind.layer} spot {pos} is already occupied")
        unknown = set(values) - set(FIELDS)
        not_allowed = set(values) & {"kind", "id", *ENERGY_FIELDS}
        if unknown or not_allowed:
            raise ValueError(f"spawn can't set {sorted(unknown | not_allowed)}")

        record = {field: 0 for field in FIELDS}
        record.update(kind.sample(rng))
        record.update(values)
        if "tone" not in values:
            record["tone"] = kind.tone_for(rng)
        if warmth is not None:
            energy = warmth * record["capacity"]
        if energy < 0:
            raise ValueError("a new thing can't start with negative energy")
        record["energy"] = energy
        record["kind"] = self.kind_index[kind_name]
        record["id"] = self.next_id

        self._withdraw(source, energy)  # pay first: if the source can't, nothing has changed yet
        self.next_id += 1
        self._write_record(kind.layer, pos, record)
        if source == "nudge":
            self._log("spawn", ids=[record["id"]], kind=kind_name, energy=energy)
        return Entity(self, record["id"])

    def remove(self, entity_id: int, *, to: Account | list[int]) -> dict:
        """Take a thing out of existence, and say where its energy (loose and stored) goes.

        `to` receives it: another thing's id (buried moss feeds the block that landed on it), a
        list of ids that share it equally (moss dying or burning feeds everything it touches),
        "start" (worldgen reshaping the world), or "nudge" (the experimenter, logged).
        Returns the thing's values as a record.
        """
        layer, pos = self.locate(entity_id)
        receivers = to if isinstance(to, list) else [to]
        if not receivers:
            raise ValueError(f"#{entity_id}'s energy has to go somewhere")
        for receiver in receivers:
            if receiver == entity_id:
                raise ValueError(f"#{entity_id} can't pass its energy to itself as it is removed")
            if isinstance(receiver, str):
                if receiver not in ("start", "nudge") or len(receivers) > 1:
                    raise ValueError(f"energy can't go to {to!r}")
            else:
                self.locate(receiver)  # check every receiver exists before anything changes
        record = self._take_record(layer, pos)
        amount = record["energy"] + record["stored"]
        for receiver in receivers:
            self._deposit(receiver, amount / len(receivers))
        if to == "nudge":
            self._log("remove", ids=[entity_id], energy=amount)
        return record

    def move(self, layer: str, src: Pos, dst: Pos) -> None:
        """Move a thing within its layer, carrying every field with it.

        Fails loudly if there is nothing at src or something already at dst, so two things
        can never end up in one spot.
        """
        src, dst = _as_pos(src), _as_pos(dst)
        if self.layers[layer]["kind"][src] == 0:
            raise ValueError(f"nothing in {layer} at {src}")
        if not self.in_bounds(dst):
            raise ValueError(f"{dst} is outside the world")
        if self.layers[layer]["kind"][dst] != 0:
            raise ValueError(f"{layer} spot {dst} is already occupied")
        self._write_record(layer, dst, self._take_record(layer, src))

    # --- nudges: the experimenter changing the world by hand (never the agent) ---

    def nudge_energy(self, targets: Target, amount: float) -> float:
        """Add `amount` of loose energy to each target, from outside the world (or take it away,
        if negative; taking stops at zero). Recorded under "nudge" and logged.

        Returns the total energy actually added (negative if taken).
        """
        ids = _ids(targets)
        changed = 0.0
        for entity_id in ids:
            layer, pos = self.locate(entity_id)
            energy = self.layers[layer]["energy"]
            change = max(amount, -float(energy[pos]))
            energy[pos] += change
            changed += change
        self.books["nudge"] += changed
        self._log("energy", ids=ids, amount=amount, changed=changed)
        return changed

    def nudge_property(self, targets: Target, field: str, value: float | bool) -> None:
        """Set one property of each target by hand, e.g. make a block heavier. Logged.

        Energy can't be set this way (use nudge_energy, so the books stay right), and neither can
        what a thing is (kind) or who it is (id).
        """
        if field not in FIELDS or field in ("kind", "id", *ENERGY_FIELDS):
            raise ValueError(f"can't nudge {field!r}")
        ids = _ids(targets)
        value = value.item() if isinstance(value, np.generic) else value  # plain value, so the log saves
        for entity_id in ids:
            layer, pos = self.locate(entity_id)
            self.layers[layer][field][pos] = value
        self._log("property", ids=ids, field=field, value=value)

    def nudge_recipe(self, kind_name: str, **changes) -> None:
        """Change a material's recipe in this world's book. Logged.

        Only things made from now on roll from the new recipe; existing things keep their values.
        """
        index = self.kind_index[kind_name]
        new = dataclasses.replace(self.recipes[index], **changes)  # its checks run again here
        if new.name != kind_name or new.layer != self.recipes[index].layer:  # type: ignore[union-attr]
            raise ValueError("a recipe's name and layer can't be changed")
        self.recipes = self.recipes[:index] + (new,) + self.recipes[index + 1 :]
        self._log("recipe", kind=kind_name, changes=changes)

    # --- copying and saving ---

    def copy(self) -> WorldState:
        """A fully independent copy: changing the copy never affects the original."""
        clone = WorldState(self.shape, self.recipes)
        clone.layers = {layer: {f: a.copy() for f, a in arrays.items()} for layer, arrays in self.layers.items()}
        clone.where = dict(self.where)
        clone.next_id = self.next_id
        clone.books = dict(self.books)
        clone.log = json.loads(json.dumps(self.log))
        return clone

    def save(self, path: str | Path) -> None:
        """Save everything to one .npz file, so this exact state can be loaded again later."""
        arrays, meta = self.pack()
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> WorldState:
        """Load a state saved with save()."""
        with np.load(path, allow_pickle=False) as data:
            return cls.unpack(data, json.loads(str(data["meta"])))

    def pack(self) -> tuple[dict[str, np.ndarray], dict]:
        """Everything in the state as (named arrays, plain JSON-able values), ready to be written
        into a file (save() does that; World.save adds its own values alongside)."""
        meta = {
            "version": SAVE_VERSION,
            "shape": list(self.shape),
            "next_id": self.next_id,
            "books": self.books,
            "log": self.log,
            "recipes": [None if kind is None else dataclasses.asdict(kind) for kind in self.recipes],
        }
        arrays = {f"{layer}.{field}": array for layer, fields in self.layers.items() for field, array in fields.items()}
        return arrays, meta

    @classmethod
    def unpack(cls, arrays, meta: dict) -> WorldState:
        """Rebuild a state from what pack() produced. The id index is rebuilt from the arrays,
        then the whole state is checked."""
        if meta["version"] != SAVE_VERSION:
            raise ValueError(f"save is version {meta['version']}, this code reads version {SAVE_VERSION}")
        recipes = tuple(None if d is None else _kind_from_dict(d) for d in meta["recipes"])
        state = cls(tuple(meta["shape"]), recipes)
        for layer, fields in state.layers.items():
            for field, dtype in FIELDS.items():
                array = arrays[f"{layer}.{field}"]
                if array.shape != state.shape:
                    raise ValueError(f"saved {layer}.{field} has the wrong shape")
                fields[field] = np.array(array, dtype=dtype)
        for layer, fields in state.layers.items():
            for pos in zip(*np.nonzero(fields["kind"])):
                pos = _as_pos(pos)
                state.where[int(fields["id"][pos])] = (layer, pos)
        state.next_id = meta["next_id"]
        state.books = {line: float(meta["books"][line]) for line in OUTSIDE}
        state.log = meta["log"]
        state.check()
        return state

    # --- consistency ---

    def check(self) -> None:
        """Raise if anything is inconsistent. Catches bookkeeping bugs right when they happen,
        instead of as strange behaviour much later. Checks that:
          - empty spots hold nothing but zeros
          - every filled spot has a real kind, which belongs in that layer
          - no energy is below zero, and no store holds more than its size
          - the id index matches the arrays exactly, and ids are never reused
          - the energy books balance
        """
        seen = set()
        for layer, arrays in self.layers.items():
            kinds = arrays["kind"]
            empty = kinds == 0
            filled = ~empty
            for field, array in arrays.items():
                if np.any(array[empty] != 0):
                    raise AssertionError(f"empty {layer} spots have nonzero {field}")
            for index in np.unique(kinds[filled]):
                kind = self.recipes[index] if 0 < index < len(self.recipes) else None
                if kind is None or kind.layer != layer:
                    raise AssertionError(f"{layer} holds kind {index}, which doesn't belong there")
            for field in ENERGY_FIELDS:
                if np.any(arrays[field][filled] < -TOLERANCE):
                    raise AssertionError(f"some {layer} has negative {field}")
            if np.any(arrays["stored"][filled] > arrays["store_size"][filled] + TOLERANCE):
                raise AssertionError(f"some {layer} stores more than its store size")
            for pos in zip(*np.nonzero(filled)):
                pos = _as_pos(pos)
                entity_id = int(arrays["id"][pos])
                if self.where.get(entity_id) != (layer, pos):
                    raise AssertionError(f"id index wrong for #{entity_id} at {layer} {pos}")
                seen.add(entity_id)
        if seen != set(self.where):
            raise AssertionError("id index lists things that don't exist")
        if seen and self.next_id <= max(seen):
            raise AssertionError("next_id would reuse an existing id")
        self.check_books()

    # --- internals ---

    def _withdraw(self, account: Account, amount: float) -> None:
        """Take energy out of an account: a thing's loose energy, or a line in the books."""
        if isinstance(account, str):
            if account not in ("start", "nudge"):
                raise ValueError(f"energy can't come from {account!r}")
            self.books[account] += amount  # energy entering the world from this line
            return
        layer, pos = self.locate(account)
        energy = self.layers[layer]["energy"]
        if energy[pos] - amount < -TOLERANCE:
            raise ValueError(f"#{account} doesn't have {amount} energy to give")
        energy[pos] -= amount

    def _deposit(self, account: Account, amount: float) -> None:
        """Put energy into an account: a thing's loose energy, or a line in the books."""
        if isinstance(account, str):
            self.books[account] -= amount  # energy leaving the world through this line
            return
        layer, pos = self.locate(account)
        self.layers[layer]["energy"][pos] += amount

    def _log(self, what: str, **details) -> None:
        self.log.append({"nudge": what, **details})

    # A "record" is one thing's values packed in a dict while it moves.

    def _take_record(self, layer: str, pos: Pos) -> dict:
        """Pack a thing's values into a record and clear its spot.

        One of the only two places that touch every field at once (with _write_record), so a
        moved thing can never lose one of its values along the way.
        """
        arrays = self.layers[layer]
        record = {field: arrays[field][pos].item() for field in FIELDS}
        for field in FIELDS:
            arrays[field][pos] = 0
        del self.where[record["id"]]
        return record

    def _write_record(self, layer: str, pos: Pos, record: dict) -> None:
        """Unpack a record into a spot and update the id index to point there."""
        arrays = self.layers[layer]
        for field in FIELDS:
            arrays[field][pos] = record[field]
        self.where[record["id"]] = (layer, pos)


def _ids(targets: Target) -> list[int]:
    """One id or several, as a list of plain ints."""
    return [int(targets)] if isinstance(targets, (int, np.integer)) else [int(t) for t in targets]


def _checked_recipes(recipes: tuple[Kind | None, ...]) -> tuple[Kind | None, ...]:
    """A recipe book must reserve index 0 for "empty" and name each material once."""
    recipes = tuple(recipes)
    if not recipes or recipes[0] is not None:
        raise ValueError("a recipe book's first entry must be None (index 0 means empty)")
    names = [kind.name for kind in recipes[1:] if kind is not None]
    if None in recipes[1:] or len(names) != len(set(names)):
        raise ValueError("every recipe after the first must be a material, each with its own name")
    return recipes


def _kind_from_dict(values: dict) -> Kind:
    """Rebuild a recipe from its saved form (JSON turns tuples into lists)."""
    values = dict(values)
    for prop in VARYING:
        if values.get(prop) is not None:
            values[prop] = tuple(values[prop])
    return Kind(**values)
