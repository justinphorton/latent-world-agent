"""How a world begins.

build_island(seed) makes an island from a fixed list of ingredients. Every seed gets every
ingredient, placed somewhere different, so islands share the same materials and laws but never
the same map (the "tier 2" variation: did an agent learn the world, or just this map?).

Worlds worth keeping (a starter island, interesting islands found later) are saved as files in
world/worlds/ and loaded with World.load, so they stay frozen even when this code changes. Later, a
bigger pipeline can sit on top of this: generate many islands, simulate them, score how much
happens and how balanced it is, and save the best ones.
"""

from __future__ import annotations

import numpy as np

from world.state import WorldState

SIZE = (32, 32, 8)  # width, depth, height
START_WARMTH = 0.3  # everything begins slightly warm, so the world is alive before the first noon
CENTER = ((SIZE[0] - 1) / 2, (SIZE[1] - 1) / 2)

# The ingredients every island gets.
CRYSTAL_GROUPS = 10  # small standing crystal groups scattered through the meadows
LOOSE_BLOCKS = ("soil", "stone", "sponge", "crystal")  # starter toys near the body
SCATTERED_BLOCKS = 16  # more loose blocks, of random materials, lying all over the island
MOSS_PATCHES = 3  # starting moss, on different levels where possible


def build_island(seed: int = 0, recipes: tuple | None = None) -> tuple[WorldState, int]:
    """Build an island from its seed (and optionally a different recipe book, for trying other
    numbers). Returns the state and the id of the body.

    Built in passes; `heights` tracks how many blocks each column has, so later passes know where
    the top is, and `taken` keeps the big features from landing on top of each other:
      1. hills of soil on a bedrock floor (moss flows downhill, so the hills shape where it lives)
      2. a stone outcrop: heavy, the building material (push vs pick up; welding later)
      3. a sponge pocket: the battery, holding the day's warmth into the night
      4. a crystal cluster: where pops can ripple from crystal to crystal (chain reactions)
      5. small standing crystal groups through the meadows: moss can never grow over a standing
         crystal, so they keep popping on hot afternoons and burn the mature moss around them,
         which is what keeps the moss in check
      6. loose blocks: one of each material near the middle, and more scattered everywhere
      7. moss patches, on different levels where possible (where moss starts decides where it
         can ever flow to)
      8. the body, standing in the middle
    """
    rng = np.random.default_rng(seed)
    state = WorldState(SIZE) if recipes is None else WorldState(SIZE, recipes)
    width, depth, _ = SIZE
    taken: list[tuple[tuple[int, int], float]] = []  # (center, radius) of each big feature

    # 1. Fill each island column: bedrock at the bottom, soil above it.
    heights = _hill_heights(rng)  # number of blocks in each column; 0 = void
    for x in range(width):
        for y in range(depth):
            for z in range(heights[x, y]):
                _place(state, "bedrock" if z == 0 else "soil", (x, y, z), rng)

    # 2. A stone outcrop: replaces the top few soil blocks with stone, plus one more on top.
    for x, y in _disc(_free_spot(rng, heights, taken, radius=2.5, near=(4, 10)), radius=2.5):
        top = heights[x, y]
        for z in range(max(1, top - 2), top + 1):
            _replace(state, "stone", (x, y, z), rng)
        heights[x, y] = top + 1

    # 3. A pocket of sponge in the ground (swaps the top soil block of each column).
    for x, y in _disc(_free_spot(rng, heights, taken, radius=1.5, near=(4, 10)), radius=1.5):
        _replace(state, "sponge", (x, y, heights[x, y] - 1), rng)

    # 4. A crystal cluster: set into the ground, with a few standing on top.
    cluster = _free_spot(rng, heights, taken, radius=2, near=(4, 10))
    cells = _disc(cluster, radius=2)
    for x, y in cells:
        _replace(state, "crystal", (x, y, heights[x, y] - 1), rng)
    for i in rng.choice(len(cells), size=3, replace=False):
        x, y = cells[i]
        _place(state, "crystal", (x, y, heights[x, y]), rng)
        heights[x, y] += 1

    # 5. Small standing crystal groups scattered through the meadows.
    for _ in range(CRYSTAL_GROUPS):
        angle, distance = rng.uniform(0, 2 * np.pi), rng.uniform(3, 11)
        center = (int(round(CENTER[0] + distance * np.cos(angle))), int(round(CENTER[1] + distance * np.sin(angle))))
        for x, y in _disc(center, radius=float(rng.choice([0.5, 1.0]))):
            if _top_is(state, heights, x, y, "soil") and heights[x, y] < SIZE[2] - 1:
                _place(state, "crystal", (x, y, heights[x, y]), rng)
                heights[x, y] += 1

    # 6. Loose blocks lying on the ground: one of each near the middle, to discover pushing and
    #    carrying early, and more scattered everywhere, so there is usually something to work with.
    scattered = tuple(rng.choice(LOOSE_BLOCKS, size=SCATTERED_BLOCKS))
    for i, kind in enumerate(LOOSE_BLOCKS + scattered):
        near = (2, 5) if i < len(LOOSE_BLOCKS) else (3, 13)
        x, y = _free_spot(rng, heights, taken, radius=0.5, near=near)
        _place(state, str(kind), (x, y, heights[x, y]), rng)
        heights[x, y] += 1

    # 7. Moss patches, each on a level not used yet where possible.
    levels_used: set[int] = set()
    for _ in range(MOSS_PATCHES):
        options = [_free_spot(rng, heights, [], radius=2, near=(3, 11)) for _ in range(8)]
        fresh = [spot for spot in options if heights[spot] not in levels_used]
        center = (fresh or options)[0]
        levels_used.add(int(heights[center]))
        for x, y in _disc(center, radius=2):
            if _top_is(state, heights, x, y, "soil") and state.has_room("cover", (x, y, heights[x, y])):
                _place(state, "moss", (x, y, heights[x, y]), rng)

    # 8. The body stands in the empty spot above the middle column (off any moss or toy).
    bx, by = int(CENTER[0] + 0.5), int(CENTER[1] + 0.5)
    if state.entity_at("cover", (bx, by, heights[bx, by])) is not None:
        state.remove(state.entity_at("cover", (bx, by, heights[bx, by])).id, to="start")
    body = _place(state, "body", (bx, by, heights[bx, by]), rng)
    return state, body.id


def _free_spot(rng, heights, taken, radius: float, near: tuple[float, float]) -> tuple[int, int]:
    """A random island column `near` (min, max) distance from the middle, whose disc of `radius`
    lies fully on the island and clear of everything in `taken`. Records it in `taken`."""
    for _ in range(500):
        angle, distance = rng.uniform(0, 2 * np.pi), rng.uniform(*near)
        center = (int(round(CENTER[0] + distance * np.cos(angle))), int(round(CENTER[1] + distance * np.sin(angle))))
        cells = _disc(center, radius)
        on_island = all(0 <= x < SIZE[0] and 0 <= y < SIZE[1] and heights[x, y] >= 2 for x, y in cells)
        clear = all(np.hypot(center[0] - c[0], center[1] - c[1]) > radius + r + 1 for c, r in taken)
        middle = np.hypot(center[0] - CENTER[0], center[1] - CENTER[1]) > radius + 1  # leave the body room
        if on_island and clear and middle:
            taken.append((center, radius))
            return center
    raise RuntimeError("couldn't find room for a feature on this island")


def _top_is(state: WorldState, heights, x: int, y: int, kind: str) -> bool:
    """Whether the top block of column (x, y) is of this kind."""
    if not (0 <= x < SIZE[0] and 0 <= y < SIZE[1]) or heights[x, y] == 0:
        return False
    top = state.entity_at("block", (x, y, heights[x, y] - 1))
    return top is not None and top.name == kind


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hill_heights(rng: np.random.Generator) -> np.ndarray:
    """Gentle hills (2 to 5 blocks tall) on a roughly round island; columns off the island are 0.

    The hills are a few overlapping sine waves with random directions and phases, rescaled to
    0..1 and then mapped to 2..5 blocks. Everything farther than about 14 from the center is
    cut away, with a little noise on that radius so the shoreline isn't a perfect circle.
    """
    width, depth, _ = SIZE
    x, y = np.meshgrid(np.arange(width), np.arange(depth), indexing="ij")

    hills = np.zeros((width, depth))
    for _ in range(3):  # a few overlapping waves with random direction and phase
        fx, fy = rng.uniform(0.1, 0.3, size=2)
        hills += np.sin(fx * x + rng.uniform(0, 2 * np.pi)) * np.sin(fy * y + rng.uniform(0, 2 * np.pi))
    hills = (hills - hills.min()) / (hills.max() - hills.min())
    heights = np.round(2 + 3 * hills).astype(int)

    center = (np.array(SIZE[:2]) - 1) / 2
    distance = np.hypot(x - center[0], y - center[1])
    edge = 14 + rng.normal(0, 0.6, size=(width, depth))  # slightly ragged shoreline
    heights[distance > edge] = 0
    return heights


def _disc(center: tuple[int, int], radius: float) -> list[tuple[int, int]]:
    """All (x, y) columns within radius of center: a filled circle on the grid."""
    cx, cy = center
    r = int(np.ceil(radius))
    return [
        (x, y)
        for x in range(cx - r, cx + r + 1)
        for y in range(cy - r, cy + r + 1)
        if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
    ]


def _place(state: WorldState, kind: str, pos, rng: np.random.Generator):
    """Spawn one entity with its individual properties sampled, starting at START_WARMTH.

    Its starting energy is paid from the "start" line of the books, so it is not energy
    "from nowhere".
    """
    return state.spawn(kind, pos, source="start", warmth=START_WARMTH, rng=rng)


def _replace(state: WorldState, kind: str, pos, rng: np.random.Generator) -> None:
    """Swap whatever block is at pos for a new entity of another kind."""
    _remove(state, pos)
    _place(state, kind, pos, rng)


def _remove(state: WorldState, pos) -> None:
    """Delete the block at pos (worldgen only: actions never destroy matter).

    Its energy goes back to the "start" line, as if it had never been given out.
    """
    if (block := state.entity_at("block", pos)) is not None:
        state.remove(block.id, to="start")


def print_height_map(state: WorldState) -> None:
    """Top-down text view: each cell shows the kind on top of its column and that block's height.

    A debugging tool for seeing the island before any graphics exist. North (high y) is at the
    top of the printout. Creatures are shown over cover, and cover over blocks.
    """
    letters = {"bedrock": "B", "soil": ".", "stone": "S", "sponge": "P", "crystal": "C", "moss": "m", "body": "@"}
    width, depth, height = state.shape
    for y in reversed(range(depth)):
        row = ""
        for x in range(width):
            top = None
            for layer in ("creature", "cover", "block"):
                for z in reversed(range(height)):
                    entity = state.entity_at(layer, (x, y, z))
                    if entity is not None:
                        top = (entity, z)
                        break
                if top:
                    break
            row += "  " if top is None else f"{letters[top[0].name]}{top[1]}"
        print(row)


if __name__ == "__main__":
    island, body_id = build_island()
    island.check()
    print_height_map(island)
    print(f"\n{len(island.where)} entities, body #{body_id}, total energy {island.total_energy():.1f}")
