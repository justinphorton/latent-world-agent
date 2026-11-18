"""What an agent perceives.

One doorway: observe() works out everything the agent can perceive this tick, as an Observation.
Two converters then pack the same facts differently: to_grid() for neural networks (a block of
numbers around the body, neighbours next to neighbours) and to_list() for readable or rule-based
agents (one entry per visible thing). Both come from the same Observation, so neither can show
something the other doesn't.

Sight:
  - a window of columns around the body (SenseSettings.window), from a few levels below the feet
    to a few above the head; or the whole island. Always north-up, like the actions.
  - surfaces only: a block is seen if at least one of its faces touches open air (no block
    there). Buried blocks stay hidden until something uncovers them. Moss and creatures always
    lie in open air, so they're always seen when in the window.
  - each seen thing shows: its position, its layer, its color (which material it looks like, as
    a number, never a name), its tone (the fixed fingerprint) and its glow (how warm it looks).
  - events have no special signal: they show up as changes (glow jumping, things appearing and
    disappearing), and noticing change is the agent's job.
Always known: time of day, its own position, its own energy.
Touch: after each action, the body's Feedback.sensed(): strain, exact warmth, energy.

Every hard choice is a dial in SenseSettings, so experiments can start easy (whole island, ids
given, exact glow and tone) and get harder one dial at a time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world.body import Feedback
from world.laws import DAY_LENGTH
from world.state import LAYERS, Pos, WorldState

LAYER_ORDER = tuple(LAYERS)  # block, cover, creature
GRID_CHANNELS = ("present", "color", "tone", "glow")  # per layer, in to_grid()


@dataclass(frozen=True)
class SenseSettings:
    """The perception dials. The defaults are the intended (harder) setting."""

    window: tuple[int, int] = (15, 15)  # columns seen east-west x north-south, centred on the body
    below: int = 3  # levels seen below the feet
    above: int = 4  # levels seen above the feet (the head is +1)
    whole_island: bool = False  # see every surface on the island instead of a window
    glow_levels: int | None = 10  # glow is seen in this many levels; None = exact
    tone_levels: int | None = 8  # tone is seen in this many levels; None = exact
    give_ids: bool = False  # hand the agent every thing's true id (normally it must recognize things)


EASY = SenseSettings(whole_island=True, glow_levels=None, tone_levels=None, give_ids=True)


@dataclass(frozen=True)
class Seen:
    """One visible thing. Positions are island coordinates (the agent knows where it is)."""

    pos: Pos
    layer: str
    color: int  # which material it looks like (the same number for every crystal), not its name
    tone: float  # a level 0..tone_levels-1, or the exact value -1..1
    glow: float  # a level 0..glow_levels-1, or the exact warmth
    id: int | None = None  # only with give_ids


@dataclass(frozen=True)
class Observation:
    """Everything the agent perceives in one tick."""

    time_of_day: float  # 0 at sunrise, rising to 1 just before the next sunrise
    position: Pos  # where the body stands
    energy: float  # the body's own energy
    touch: tuple[float, float | None, float] | None  # (strain, warmth, energy) from the last action
    seen: tuple[Seen, ...]
    lo: Pos  # the window's lowest corner, in island coordinates
    size: tuple[int, int, int]  # the window's size


def observe(state: WorldState, body_id: int, tick: int, settings: SenseSettings = SenseSettings(),
            feedback: Feedback | None = None) -> Observation:
    """What the body with this id perceives right now."""
    _, (bx, by, bz) = state.locate(body_id)
    lo, size = _window(state, (bx, by, bz), settings)
    hi = tuple(l + s for l, s in zip(lo, size))
    clip_lo = tuple(max(0, l) for l in lo)
    clip_hi = tuple(min(n, h) for n, h in zip(state.shape, hi))
    box = tuple(slice(a, b) for a, b in zip(clip_lo, clip_hi))

    visible = visible_mask(state)
    seen = []
    for layer in LAYER_ORDER:
        arrays = state.layers[layer]
        here = visible[layer][box]
        for offset in zip(*np.nonzero(here)):
            pos = tuple(int(o + c) for o, c in zip(offset, clip_lo))
            capacity = arrays["capacity"][pos]
            warmth = float(arrays["energy"][pos] / capacity) if capacity > 0 else 0.0
            seen.append(
                Seen(
                    pos=pos,  # type: ignore[arg-type]
                    layer=layer,
                    color=int(arrays["kind"][pos]),
                    tone=_level(float(arrays["tone"][pos]), -1.0, 1.0, settings.tone_levels),
                    glow=_level(warmth, 0.0, 1.0, settings.glow_levels),
                    id=int(arrays["id"][pos]) if settings.give_ids else None,
                )
            )
    entity = state.entity_by_id(body_id)
    assert entity is not None
    return Observation(
        time_of_day=(tick % DAY_LENGTH) / DAY_LENGTH,
        position=(bx, by, bz),
        energy=entity.energy,
        touch=feedback.sensed() if feedback is not None else None,
        seen=tuple(seen),
        lo=lo,  # type: ignore[arg-type]
        size=size,  # type: ignore[arg-type]
    )


def visible_mask(state: WorldState) -> dict[str, np.ndarray]:
    """Which spots of each layer hold something with a face open to the air (surfaces only).

    A block is visible if any of its six neighbouring spots has no block in it (the edge of the
    world counts as open). Moss and creatures always sit in open air, so they're always visible.
    """
    block = state.occupied("block")
    padded = np.pad(block, 1, constant_values=False)  # outside the world is open air
    enclosed = np.ones_like(block)
    for axis in range(3):
        for step in (-1, 1):
            neighbour = np.roll(padded, -step, axis=axis)[1:-1, 1:-1, 1:-1]
            enclosed &= neighbour
    return {
        "block": block & ~enclosed,
        "cover": state.occupied("cover"),
        "creature": state.occupied("creature"),
    }


def to_list(observation: Observation) -> list[dict]:
    """One readable entry per visible thing (for rule-based or language agents)."""
    entries = []
    for thing in observation.seen:
        entry = {"pos": thing.pos, "layer": thing.layer, "color": thing.color, "tone": thing.tone, "glow": thing.glow}
        if thing.id is not None:
            entry["id"] = thing.id
        entries.append(entry)
    return entries


def to_grid(observation: Observation) -> np.ndarray:
    """The window as numbers: shape (layers x channels, width, depth, height), channels-first.

    For each layer: present (1 where something visible is), color, tone and glow. Spots with
    nothing visible (empty, buried, or outside the island) are all zeros. The body is always at
    the same place in the grid (the middle of the window, at `below` levels up).
    """
    grid = np.zeros((len(LAYER_ORDER) * len(GRID_CHANNELS), *observation.size), dtype=np.float32)
    for thing in observation.seen:
        offset = tuple(p - l for p, l in zip(thing.pos, observation.lo))
        base = LAYER_ORDER.index(thing.layer) * len(GRID_CHANNELS)
        grid[(base, *offset)] = 1.0
        grid[(base + 1, *offset)] = thing.color
        grid[(base + 2, *offset)] = thing.tone
        grid[(base + 3, *offset)] = thing.glow
    return grid


def _window(state: WorldState, body: Pos, settings: SenseSettings) -> tuple[Pos, tuple[int, int, int]]:
    """The window's lowest corner and size, in island coordinates (it may stick out of the world)."""
    if settings.whole_island:
        return (0, 0, 0), state.shape
    bx, by, bz = body
    wx, wy = settings.window
    lo = (bx - wx // 2, by - wy // 2, bz - settings.below)
    return lo, (wx, wy, settings.below + settings.above + 1)


def _level(value: float, low: float, high: float, levels: int | None) -> float:
    """value seen exactly (levels=None), or as one of `levels` steps between low and high."""
    if levels is None:
        return value
    fraction = (min(max(value, low), high) - low) / (high - low)
    return float(min(levels - 1, int(fraction * levels)))
