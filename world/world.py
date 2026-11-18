"""The world as a whole: state + body + laws, advanced one tick at a time.

This is the only place that knows the order things happen in. Each tick:
    1. the body acts            (move / grasp / release / hit / wait)
    2. the laws that are on, in laws.LAWS order: gravity, sun, flow, pop, life, radiation
    3. checks                   (state consistent, energy books balance, nothing floating)
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from world import laws
from world.body import Action, Body, Feedback
from world.laws import LAWS, RUNGS
from world.state import LAYERS, WorldState
from world.worldgen import build_island

WORLD_SAVE_VERSION = 1  # bump when the world save format changes


@dataclass
class StepResult:
    """What happened during one tick: what the body felt, what fell, and what popped."""

    feedback: Feedback
    events: list = field(default_factory=list)  # Falls, for the viewer to animate
    popped: list[int] = field(default_factory=list)  # ids of everything that popped


@dataclass(frozen=True)
class Snapshot:
    """Everything needed to put a world back exactly as it was at one moment.

    The body needs nothing of its own: it has no hidden state, so the state holds all of it.
    Viewer settings (speed, facing, who drives) are not part of the world and aren't kept.
    """

    state: WorldState  # a private copy: never handed out or changed
    tick: int
    dice: dict  # the exact position of the world's random number generator
    seed: int
    laws_on: frozenset[str]
    body_id: int


class World:
    """One running world.

    Which laws run: pass `rung` (1, 2 or 3, see laws.RUNGS) or an explicit set of law names in
    `laws`. Any combination is valid, because no law depends on another being on.

    `check_every` sets how often every invariant is verified: 1 = after every tick (the default,
    a few milliseconds on the test island), 100 = every hundredth tick, 0 = never (for big, fast
    runs without the viewer).
    """

    def __init__(self, seed: int = 0, rung: int = 3, laws: Iterable[str] | None = None, check_every: int = 1):
        chosen = frozenset(laws) if laws is not None else RUNGS[rung]
        unknown = chosen - set(LAWS)
        if unknown:
            raise ValueError(f"unknown laws: {sorted(unknown)}")
        self.laws_on = chosen
        self.check_every = check_every
        self.reset(seed)

    def reset(self, seed: int) -> None:
        """Build a fresh island from a seed and start the clock at sunrise."""
        self.seed = seed
        self.state, body_id = build_island(seed)
        self.body = Body(self.state, body_id)
        self.rng = np.random.default_rng([seed, 1])  # the world's own dice (growth), apart from worldgen's
        self.tick = 0

    @property
    def sun(self) -> float:
        """Current sun strength, 0 (night) to 1 (noon)."""
        return laws.sun_level(self.tick)

    def step(self, action: Action | None = None) -> StepResult:
        """Advance one tick. No action means the body waits."""
        on = self.laws_on
        result = StepResult(self.body.act(action or Action("wait")))
        if "gravity" in on:
            result.events = laws.settle(self.state)
        if "sun" in on:
            laws.shine(self.state, self.tick)
        if "flow" in on:
            laws.flow(self.state)
        if "pop" in on:
            result.popped = laws.pop(self.state)
        if "life" in on:
            laws.life(self.state, self.rng)
        if "radiation" in on:
            laws.radiate(self.state)
        self.tick += 1
        if self.check_every and self.tick % self.check_every == 0:
            self.verify()
        return result

    def verify(self) -> None:
        """Raise if any invariant is broken: bookkeeping and energy books (state.check), or,
        while gravity is on, anything left floating."""
        self.state.check()
        if "gravity" not in self.laws_on:
            return
        for layer in LAYERS:
            for pos in zip(*self.state.occupied(layer).nonzero()):
                pos = tuple(int(p) for p in pos)
                if not laws.is_supported(self.state, layer, pos):
                    raise AssertionError(f"{layer} at {pos} is floating")

    # --- snapshots and saving ---

    def snapshot(self) -> Snapshot:
        """Capture this exact moment in memory (fast). restore() puts it back; replaying the same
        actions from there gives exactly the same world, tick for tick."""
        return Snapshot(
            state=self.state.copy(),
            tick=self.tick,
            dice=copy.deepcopy(self.rng.bit_generator.state),
            seed=self.seed,
            laws_on=self.laws_on,
            body_id=self.body.entity_id,
        )

    def restore(self, snapshot: Snapshot) -> None:
        """Go back to a snapshot. The snapshot itself stays untouched, so it can be reused."""
        self.state = snapshot.state.copy()
        self.body = Body(self.state, snapshot.body_id)
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = copy.deepcopy(snapshot.dice)
        self.tick = snapshot.tick
        self.seed = snapshot.seed
        self.laws_on = snapshot.laws_on

    def save(self, path: str | Path) -> None:
        """Save the whole world (state, time, dice, settings) to one .npz file."""
        arrays, state_meta = self.state.pack()
        world_meta = {
            "version": WORLD_SAVE_VERSION,
            "tick": self.tick,
            "dice": self.rng.bit_generator.state,
            "seed": self.seed,
            "laws_on": sorted(self.laws_on),
            "body_id": self.body.entity_id,
        }
        np.savez_compressed(
            path, meta=np.array(json.dumps(state_meta)), world=np.array(json.dumps(world_meta)), **arrays
        )

    @classmethod
    def load(cls, path: str | Path, check_every: int = 1) -> World:
        """Load a world saved with save(), ready to keep running exactly where it left off."""
        with np.load(path, allow_pickle=False) as data:
            state = WorldState.unpack(data, json.loads(str(data["meta"])))
            meta = json.loads(str(data["world"]))
        if meta["version"] != WORLD_SAVE_VERSION:
            raise ValueError(f"world save is version {meta['version']}, this code reads {WORLD_SAVE_VERSION}")
        world = cls.__new__(cls)
        world.check_every = check_every
        world.restore(
            Snapshot(
                state=state,
                tick=meta["tick"],
                dice=meta["dice"],
                seed=meta["seed"],
                laws_on=frozenset(meta["laws_on"]),
                body_id=meta["body_id"],
            )
        )
        return world
