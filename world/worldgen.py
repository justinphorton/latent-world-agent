"""How a world begins.

For now there is one hand-built test island. Its layout is fixed so we can check that the rules
behave the way we intend; the seed only varies individual properties and the hill shapes.
"""

from __future__ import annotations

import numpy as np

from world.state import WorldState

SIZE = (32, 32, 8)  # width, depth, height
START_WARMTH = 0.3  # everything begins slightly warm, so the world is alive before the first noon
CRYSTAL_GROUPS = 10  # small crystal groups scattered through the meadows


def build_test_island(seed: int = 0) -> tuple[WorldState, int]:
    """Build the test island. Returns the state and the id of the body.

    The island is built in passes, each one a feature to test the rules against:
      1. hills of soil on a bedrock floor
      2. a stone outcrop (heavy: tests push/lift limits)
      3. a sponge pocket (high capacity: should act like a battery)
      4. a crystal cluster (low capacity: will matter once cracking exists)
      5. a pit (tests falling in, digging and pillaring out)
      6. loose blocks on flat ground (tests pushing and carrying)
      7. moss patches (tests the cover layer and trampling)
      8. the body, standing in the middle

    `heights` tracks how many blocks each column has, so later passes know where the top is.
    """
    rng = np.random.default_rng(seed)
    state = WorldState(SIZE)
    width, depth, _ = SIZE

    # 1. Fill each island column: bedrock at the bottom, soil above it.
    heights = _hill_heights(rng)  # number of blocks in each column; 0 = void
    for x in range(width):
        for y in range(depth):
            for z in range(heights[x, y]):
                _place(state, "bedrock" if z == 0 else "soil", (x, y, z), rng)

    # 2. A stone outcrop: a raised, heavy lump. Some stones are too heavy to lift.
    #    Replaces the top few soil blocks with stone, then adds one more stone on top.
    for x, y in _disc((8, 22), radius=2.5):
        top = heights[x, y]
        for z in range(max(1, top - 2), top + 1):
            _replace(state, "stone", (x, y, z), rng)
        heights[x, y] = top + 1

    # 3. A pocket of sponge-like blocks in the ground (swaps the top soil block of each column).
    for x, y in _disc((22, 9), radius=1.5):
        _replace(state, "sponge", (x, y, heights[x, y] - 1), rng)

    # 4. A crystal cluster, some set into the ground and some standing on it.
    for x, y in _disc((23, 22), radius=2):
        _replace(state, "crystal", (x, y, heights[x, y] - 1), rng)
    for x, y in [(23, 22), (24, 22), (23, 23)]:
        _place(state, "crystal", (x, y, heights[x, y]), rng)
        heights[x, y] += 1

    # 4b. Small crystal groups scattered through the meadows, standing on the ground. Moss spreads
    #     only level or downhill, so it can never grow over a standing crystal: they stay in the
    #     sun, keep popping on hot afternoons, and set fire to whatever mature moss surrounds them.
    #     That is what keeps the moss population in check.
    for _ in range(CRYSTAL_GROUPS):
        angle, distance = rng.uniform(0, 2 * np.pi), rng.uniform(3, 11)
        center = (int(round(15.5 + distance * np.cos(angle))), int(round(15.5 + distance * np.sin(angle))))
        for x, y in _disc(center, radius=float(rng.choice([0.5, 1.0]))):
            top = state.entity_at("block", (x, y, heights[x, y] - 1))
            if top is not None and top.name == "soil" and heights[x, y] < SIZE[2]:
                _place(state, "crystal", (x, y, heights[x, y]), rng)
                heights[x, y] += 1

    # 5. A 2x2 pit, two blocks deep (removes the top two blocks of four columns).
    for x, y in [(11, 10), (12, 10), (11, 11), (12, 11)]:
        for _ in range(2):
            _remove(state, (x, y, heights[x, y] - 1))
            heights[x, y] -= 1

    # 6. Loose blocks lying on flat ground near the middle, to push and carry.
    for (x, y), kind in [((14, 16), "soil"), ((17, 15), "stone"), ((15, 19), "sponge"), ((18, 18), "crystal")]:
        _place(state, kind, (x, y, heights[x, y]), rng)
        heights[x, y] += 1

    # 7. Moss patches, only where the top block is bare soil.
    #    Moss is a cover entity: it sits in the empty slot just above the top block.
    for center in [(10, 16), (20, 13)]:
        for x, y in _disc(center, radius=2):
            top = state.entity_at("block", (x, y, heights[x, y] - 1))
            if top is not None and top.name == "soil":
                _place(state, "moss", (x, y, heights[x, y]), rng)

    # 8. The body stands in the empty slot above the middle column.
    body = _place(state, "body", (16, 16, heights[16, 16]), rng)
    return state, body.id


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
    island, body_id = build_test_island()
    island.check()
    print_height_map(island)
    print(f"\n{len(island.where)} entities, body #{body_id}, total energy {island.total_energy():.1f}")
