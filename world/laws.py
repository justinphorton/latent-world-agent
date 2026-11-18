"""How the world changes on its own.

Each law is a function that reads a WorldState and changes it only through the state's
books-keeping methods (move_energy, exchange_with_outside, spawn, remove, move). Laws never call
each other, never depend on another law being switched on, and never name materials ("if moss"
is banned): they read numbers from the recipe book and rules from the layer table.

The laws, in the order they run each tick (world.py runs them; each one can be switched off):
  gravity    things with nothing holding them up fall, all the way, within the tick
  sun        the top of every column soaks up sunlight: 2/3 of each day is light, 1/3 dark
  flow       energy moves between touching things, from warmer to cooler
  pop        stores fill; a full store on a warm enough thing pops into its neighbours
  life       living things spread when well fed and die when starving
  radiation  faces open to the air lose energy to the sky
(Welding is designed but not built yet; it will run between life and radiation.)

Rungs are suggested sets of switches, each adding one new kind of behaviour:
  1 base physics   2 + living matter   3 + store and pop   (4 + welding, later)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from world.state import CONTACTS, ENERGY_FIELDS, LAYERS, Entity, Pos, WorldState

LAWS = ("gravity", "sun", "flow", "pop", "life", "radiation")  # the order they run in
RUNGS: dict[int, frozenset[str]] = {
    1: frozenset({"gravity", "sun", "flow", "radiation"}),
    2: frozenset({"gravity", "sun", "flow", "radiation", "life"}),
    3: frozenset({"gravity", "sun", "flow", "radiation", "life", "pop"}),
}

DAY_LENGTH = 900  # ticks per full day and night
DAYLIGHT = 2 / 3  # fraction of each day the sun is up
SUN_POWER = 0.025  # energy per tick reaching the top of a column at noon
RADIATION = 0.02  # energy lost per tick per exposed face, times warmth
MAX_FLOW_STIFFNESS = 0.4  # flow is split into substeps if one tick would move energy too abruptly
GROUND_PULL = 0.01  # added to every spot's weight when living things pick where to spread

_SIDES = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # north, east, south, west: the fixed order for sliding


@dataclass(frozen=True)
class Fall:
    """A thing dropped from one spot to another (the viewer animates these)."""

    entity_id: int
    start: Pos
    end: Pos


# ---------------------------------------------------------------------------
# Gravity
# ---------------------------------------------------------------------------


def is_supported(state: WorldState, layer: str, pos: Pos) -> bool:
    """Whether something in this layer at pos would be held up.

    It is held up if it sits on the world's floor (z = 0), or if the spot directly below holds
    something from one of the layers it can rest on (LAYERS[layer].supported_by).

    One exception: a creature that has fallen onto another creature slides off to the side, but
    if every side is blocked it has nowhere to go, so it rests on the other's head.
    """
    x, y, z = pos
    if z == 0:
        return True
    below = (x, y, z - 1)
    if any(state.layers[support]["kind"][below] != 0 for support in LAYERS[layer].supported_by):
        return True
    return layer == "creature" and state.layers["creature"]["kind"][below] != 0 and _free_side(state, pos) is None


def settle(state: WorldState) -> list[Fall]:
    """Let everything that isn't supported fall straight down until it is.

    Works from the bottom of the world upward, so a whole unsupported stack falls together:
    the lowest piece lands first and the pieces above land on it. Everything is resolved within
    one call, so after settle() nothing in the world is floating.
    """
    events: list[Fall] = []
    _, _, height = state.shape
    for z in range(1, height):
        for layer in LAYERS:
            occupied = state.occupied(layer)[:, :, z]
            for x, y in zip(*np.nonzero(occupied)):
                pos = (int(x), int(y), z)
                if state.layers[layer]["kind"][pos] != 0 and not is_supported(state, layer, pos):
                    events.extend(_drop(state, layer, pos))
    return events


def _drop(state: WorldState, layer: str, pos: Pos) -> list[Fall]:
    """Move one unsupported thing down to where it comes to rest.

    Falling straight down, it stops at the first spot where it would be supported. Special
    landings, all keeping the energy in the world:
      - a block landing where moss is buries it: the moss's energy goes into the block
      - moss landing where moss already is merges into it
      - a creature about to land on another creature slides off to the first free side
        (north, east, south, west) and keeps falling from there
    """
    x, y, z = pos
    falling = state.entity_at(layer, pos)
    assert falling is not None

    land = z
    while land > 0 and not is_supported(state, layer, (x, y, land)):
        if layer == "creature" and state.layers["creature"]["kind"][x, y, land - 1] != 0:
            side = _free_side(state, (x, y, land))
            assert side is not None  # if it were boxed in, is_supported would have said so
            state.move(layer, pos, side)
            return [Fall(falling.id, pos, side), *_drop(state, layer, side)]
        land -= 1
    target = (x, y, land)

    already_there = state.entity_at("cover", target)
    if layer == "cover" and already_there is not None:
        absorb(state, layer, pos, into=already_there)  # moss onto moss: they merge
        return []
    if layer == "block" and already_there is not None:
        absorb(state, "cover", target, into=falling)  # a block onto moss: the moss is buried

    state.move(layer, pos, target)
    return [Fall(falling.id, pos, target)]


def _free_side(state: WorldState, pos: Pos) -> Pos | None:
    """The first neighbouring spot (north, east, south, west) a creature could move into."""
    x, y, z = pos
    for dx, dy in _SIDES:
        if state.has_room("creature", (x + dx, y + dy, z)):
            return (x + dx, y + dy, z)
    return None


def absorb(state: WorldState, layer: str, pos: Pos, into: Entity) -> None:
    """Remove the thing at pos and pass its energy into `into`, so no energy is lost."""
    gone = state.entity_at(layer, pos)
    assert gone is not None
    state.remove(gone.id, to=into.id)


# ---------------------------------------------------------------------------
# Sun, flow, radiation
# ---------------------------------------------------------------------------


def sun_level(tick: int) -> float:
    """How strong the sun is at this tick: rising from 0 at sunrise to 1 at noon and back to 0
    at sunset, then 0 all night.

    Tick 0 is sunrise. The first DAYLIGHT (2/3) of each day is light, the rest is night.
    """
    phase = (tick % DAY_LENGTH) / DAY_LENGTH
    return math.sin(math.pi * phase / DAYLIGHT) if phase < DAYLIGHT else 0.0


def shine(state: WorldState, tick: int) -> None:
    """The sun warms the topmost thing of every column.

    Only the top gets sunlight: anything underneath is in the shade. When a creature and a cover
    share the top spot (the body standing on moss), the creature catches the light.
    """
    energy_in = SUN_POWER * sun_level(tick)
    if energy_in == 0:
        return
    occupied = {layer: state.occupied(layer) for layer in LAYERS}
    anything = np.logical_or.reduce(list(occupied.values()))
    has_top = anything.any(axis=2)
    height = state.shape[2]
    top_z = height - 1 - np.argmax(anything[:, :, ::-1], axis=2)  # highest occupied spot per column

    unclaimed = has_top.copy()
    xs, ys = np.indices(has_top.shape)
    for layer in ("creature", "cover", "block"):
        hit = unclaimed & occupied[layer][xs, ys, top_z]
        delta = np.zeros(state.shape)
        delta[xs[hit], ys[hit], top_z[hit]] = energy_in
        state.exchange_with_outside(layer, delta, "sun")
        unclaimed &= ~hit


def flow(state: WorldState) -> None:
    """Energy moves between touching things, from warmer to cooler.

    For each contact in CONTACTS: flow = k * (warmth_a - warmth_b), where k is the harmonic mean
    of the two conductivities (the poorer conductor dominates). All flows in a step are computed
    from the same starting warmths, then applied together, so the result doesn't depend on any
    order and energy is conserved exactly.

    If the step would be too abrupt for some low-capacity thing (it could overshoot and
    oscillate), the tick is split into equal substeps.
    """
    substeps = max(1, math.ceil(_flow_stiffness(state) / MAX_FLOW_STIFFNESS))
    for _ in range(substeps):
        delta = {layer: np.zeros(state.shape) for layer in LAYERS}
        warmth = {layer: _warmth(state, layer) for layer in LAYERS}
        for layer_a, layer_b, offset in CONTACTS:
            a, b = _contact_slices(state.shape, offset)
            k = _conductance(state, layer_a, a, layer_b, b)
            moved = k * (warmth[layer_a][a] - warmth[layer_b][b]) / substeps
            delta[layer_a][a] -= moved
            delta[layer_b][b] += moved
        state.move_energy({(layer, "energy"): d for layer, d in delta.items()})


def radiate(state: WorldState) -> None:
    """Every exposed face loses energy to the sky, in proportion to the thing's warmth.

    A face is exposed when the spot next to it holds nothing solid (no block or creature).
    Blocks radiate from all six faces, so compact clumps keep their heat and thin towers lose it.
    Cover and creatures only radiate upward. Faces at the edge of the world don't count:
    the world's boundary is insulated.
    """
    solid = state.occupied("block") | state.occupied("creature")
    open_above = ~_shifted(solid, (0, 0, 1), fill=True)
    for layer in LAYERS:
        if layer == "block":
            faces = sum(~_shifted(solid, offset, fill=True) for offset in _SIX_FACES).astype(float)
        else:
            faces = open_above.astype(float)
        occupied = state.occupied(layer)
        energy = state.layers[layer]["energy"]
        loss = np.where(occupied, RADIATION * _warmth(state, layer) * faces, 0.0)
        loss = np.minimum(loss, energy)  # can't radiate away more than it has
        state.exchange_with_outside(layer, -loss, "radiation")


# ---------------------------------------------------------------------------
# Store and pop
# ---------------------------------------------------------------------------


def pop(state: WorldState) -> list[int]:
    """Stores fill, and full stores on warm enough things pop. Returns the ids that popped.

    1. Filling: each thing with a store moves refill_rate * warmth of its loose energy into its
       store each tick, until the store is full. A charging thing dims a little as it fills.
    2. Popping: a thing pops when its store is full AND its warmth has reached its pop point
       (release_at). Needing a full store means it can't pop again until it has refilled, which
       is what makes chain reactions travel as clean waves instead of bouncing back and forth.
    3. Releasing: everything in the store goes out at once, split equally among everything it
       touches. A thing that is consumed_on_pop (moss burning) also gives away its loose energy
       and disappears.
    All pops in a tick are decided from the same starting state and applied together, so a chain
    reaction spreads one ring per tick, whatever order things are listed in.
    """
    _fill_stores(state)

    poppers = []
    for layer, arrays in state.layers.items():
        has_store = arrays["store_size"] > 0
        full = has_store & (arrays["stored"] >= arrays["store_size"] * (1 - 1e-9))
        hot = _warmth(state, layer) >= arrays["release_at"]
        poppers.extend(int(i) for i in arrays["id"][full & hot])
    if not poppers:
        return []

    changes = {(layer, field): np.zeros(state.shape) for layer in LAYERS for field in ENERGY_FIELDS}
    consumed = []
    for entity_id in poppers:
        layer, pos = state.locate(entity_id)
        arrays = state.layers[layer]
        amount = arrays["stored"][pos]
        changes[(layer, "stored")][pos] -= amount
        if state.recipes[arrays["kind"][pos]].consumed_on_pop:  # type: ignore[union-attr]
            amount += arrays["energy"][pos]
            changes[(layer, "energy")][pos] -= arrays["energy"][pos]
            consumed.append(entity_id)
        receivers = state.neighbours(entity_id) or [entity_id]  # touching nothing: it keeps the heat
        for receiver in receivers:
            r_layer, r_pos = state.locate(receiver)
            changes[(r_layer, "energy")][r_pos] += amount / len(receivers)
    state.move_energy({key: delta for key, delta in changes.items() if delta.any()})

    # Things used up by popping disappear, passing on anything they were given this tick.
    for entity_id in sorted(consumed):
        state.remove(entity_id, to=state.neighbours(entity_id))
    return poppers


def _fill_stores(state: WorldState) -> None:
    """Each thing with a store moves refill_rate * warmth of its loose energy into it."""
    refill_rate = _per_kind(state, "refill_rate")
    changes = {}
    for layer, arrays in state.layers.items():
        rate = refill_rate[arrays["kind"]]
        amount = np.minimum(rate * _warmth(state, layer), arrays["store_size"] - arrays["stored"])
        amount = np.clip(np.minimum(amount, arrays["energy"]), 0, None)
        if amount.any():
            changes[(layer, "energy")] = -amount
            changes[(layer, "stored")] = amount
    if changes:
        state.move_energy(changes)


# ---------------------------------------------------------------------------
# Life
# ---------------------------------------------------------------------------


def life(state: WorldState, rng: np.random.Generator) -> None:
    """Living things (those whose recipe has grow_at) die when starving and spread when well fed.

    1. Dying: anything alive colder than its die_at disappears, its energy split among
       everything it touches.
    2. Spreading: anything alive that is mature (its store is full) and warmer than its grow_at
       puts one new patch into a free neighbouring spot of its own layer, on the same level or
       one step down, never up: life flows downhill, so each terrace fills on its own and higher
       ground stays bare unless something carries it up. The new patch's energy, grow_cost, comes out of
       the parent's store. Stores fill slowly, so this sets the pace: a parent must refill
       before spreading again, and a newborn must fill its store (about a day) before it can
       spread at all. It also means a patch that just spread has less fuel to burn.
       It only grows from the ground (resting on a block), never while carried on a head, and
       only onto ground.
    Which spot it picks is a roll of the world's seeded dice, weighted toward warm ground, so
    moss looks natural, tends to grow toward warmth, and the same seed always gives the same
    result. Growers take turns in a dice-shuffled order, so no direction is favoured.
    """
    grow_at = _per_kind(state, "grow_at")
    die_at = _per_kind(state, "die_at")
    alive = _per_kind(state, "alive").astype(bool)

    dying, growing = [], []
    for layer, arrays in state.layers.items():
        kind = arrays["kind"]
        living = alive[kind] & (kind != 0)
        warmth = _warmth(state, layer)
        mature = arrays["stored"] >= arrays["store_size"] * (1 - 1e-9)
        dying.extend(int(i) for i in arrays["id"][living & (warmth < die_at[kind])])
        growing.extend(int(i) for i in arrays["id"][living & mature & (warmth >= grow_at[kind])])

    for entity_id in sorted(dying):
        state.remove(entity_id, to=state.neighbours(entity_id))

    for entity_id in rng.permutation(growing) if growing else []:
        _spread(state, int(entity_id), rng)


def _spread(state: WorldState, parent_id: int, rng: np.random.Generator) -> None:
    """One living thing puts one new patch into a free neighbouring spot, if it can."""
    layer, (x, y, z) = state.locate(parent_id)
    parent = state.entity_by_id(parent_id)
    assert parent is not None
    kind = parent.kind_info
    if not _on_ground(state, (x, y, z)) or parent.stored < kind.grow_cost:
        return

    spots = []
    for dx, dy in _SIDES:
        for dz in (0, -1):  # level or one step down, never up
            spot = (x + dx, y + dy, z + dz)
            # going down, the way there must be open: no block in the way at the parent's level
            path = (x + dx, y + dy, z)
            if dz == -1 and state.in_bounds(path) and state.layers["block"]["kind"][path] != 0:
                continue
            if state.has_room(layer, spot) and _on_ground(state, spot):
                spots.append(spot)
    if not spots:
        return

    weights = np.array([_ground_warmth(state, spot) + GROUND_PULL for spot in spots])
    spot = spots[rng.choice(len(spots), p=weights / weights.sum())]
    # The cost comes out of the parent's store: move it to loose energy, then hand it over.
    cost = np.zeros(state.shape)
    cost[x, y, z] = kind.grow_cost
    state.move_energy({(layer, "stored"): -cost, (layer, "energy"): cost})
    state.spawn(kind.name, spot, source=parent_id, energy=kind.grow_cost, rng=rng)


def _on_ground(state: WorldState, pos: Pos) -> bool:
    """Whether the spot directly below pos holds a block."""
    x, y, z = pos
    return z > 0 and state.layers["block"]["kind"][x, y, z - 1] != 0


def _ground_warmth(state: WorldState, pos: Pos) -> float:
    """Warmth of the block directly below pos."""
    below = state.entity_at("block", (pos[0], pos[1], pos[2] - 1))
    return 0.0 if below is None else max(0.0, below.warmth)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIX_FACES = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def _per_kind(state: WorldState, attribute: str) -> np.ndarray:
    """One recipe number per kind index, for looking up with a kind array: e.g.
    _per_kind(state, "refill_rate")[arrays["kind"]]. Empty (index 0) gets 0; a missing grow_at
    gets infinity (never grows)."""
    values = []
    for kind in state.recipes:
        value = 0.0 if kind is None else getattr(kind, attribute)
        values.append(math.inf if value is None else float(value))
    return np.array(values)


def _warmth(state: WorldState, layer: str) -> np.ndarray:
    """energy / capacity for every spot of a layer (0 where empty)."""
    arrays = state.layers[layer]
    capacity = arrays["capacity"]
    return np.divide(arrays["energy"], capacity, out=np.zeros(state.shape), where=capacity > 0)


def _contact_slices(shape, offset) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """Array slices pairing every spot p (first) with p + offset (second), within the grid."""
    a, b = [], []
    for d, n in zip(offset, shape):
        a.append(slice(0, n - d) if d >= 0 else slice(-d, n))
        b.append(slice(d, n) if d >= 0 else slice(0, n + d))
    return tuple(a), tuple(b)


def _conductance(state: WorldState, layer_a: str, a, layer_b: str, b) -> np.ndarray:
    """Harmonic-mean conductivity for every contact pair (0 where either side is empty)."""
    ca = state.layers[layer_a]["conductivity"][a]
    cb = state.layers[layer_b]["conductivity"][b]
    total = ca + cb
    return np.divide(2 * ca * cb, total, out=np.zeros_like(total), where=total > 0)


def _flow_stiffness(state: WorldState) -> float:
    """The largest fraction of its warmth gap any thing could move in one tick.

    Above about 0.5 an explicit step can overshoot; flow() substeps to stay under the limit.
    """
    total_k = {layer: np.zeros(state.shape) for layer in LAYERS}
    for layer_a, layer_b, offset in CONTACTS:
        a, b = _contact_slices(state.shape, offset)
        k = _conductance(state, layer_a, a, layer_b, b)
        total_k[layer_a][a] += k
        total_k[layer_b][b] += k
    worst = 0.0
    for layer in LAYERS:
        capacity = state.layers[layer]["capacity"]
        ratio = np.divide(total_k[layer], capacity, out=np.zeros(state.shape), where=capacity > 0)
        worst = max(worst, float(ratio.max()))
    return worst


def _shifted(mask: np.ndarray, offset, fill: bool) -> np.ndarray:
    """mask[p + offset] for every p; spots whose neighbour is outside the grid get `fill`."""
    out = np.full(mask.shape, fill)
    a, b = _contact_slices(mask.shape, offset)
    out[a] = mask[b]
    return out
