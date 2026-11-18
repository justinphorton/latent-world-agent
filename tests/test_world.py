"""Whole-world snapshots and saves: going back and replaying must give exactly the same world.

This is what makes replay trustworthy. If anything in the world used randomness or state from
outside what a snapshot captures, these tests would catch it.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.agents import RandomAgent
from world.world import World

BEFORE, AFTER = 1500, 900  # snapshot at tick 1500 (moss is growing by then), then 900 more ticks


def random_actions(count: int, seed: int = 5):
    agent = RandomAgent(seed)
    return [agent.act(None) for _ in range(count)]


def run(world: World, actions) -> None:
    for action in actions:
        world.step(action)


def assert_same(a: World, b: World) -> None:
    """Two worlds are identical, down to the last bit of every value."""
    assert a.tick == b.tick and a.seed == b.seed and a.laws_on == b.laws_on
    assert a.body.entity_id == b.body.entity_id
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
    for layer, fields in a.state.layers.items():
        for field, array in fields.items():
            assert np.array_equal(array, b.state.layers[layer][field]), f"{layer}.{field} differs"
    assert a.state.where == b.state.where
    assert a.state.books == b.state.books
    assert a.state.next_id == b.state.next_id
    assert a.state.recipes == b.state.recipes


@pytest.mark.slow
def test_same_seed_and_actions_give_the_same_world():
    actions = random_actions(BEFORE + AFTER)
    a, b = World(seed=3, check_every=0), World(seed=3, check_every=0)
    run(a, actions)
    run(b, actions)
    assert_same(a, b)


@pytest.mark.slow
def test_restoring_a_snapshot_and_replaying_gives_exactly_what_happened():
    actions = random_actions(BEFORE + AFTER)
    world = World(seed=3, check_every=0)
    run(world, actions[:BEFORE])
    snapshot = world.snapshot()
    dice_then = world.rng.bit_generator.state
    run(world, actions[BEFORE:])
    assert world.rng.bit_generator.state != dice_then  # the dice really were used after the snapshot
    original = world.snapshot()

    world.restore(snapshot)
    assert world.tick == BEFORE
    run(world, actions[BEFORE:])
    replayed = World.__new__(World)
    replayed.check_every = 0
    replayed.restore(original)
    assert_same(world, replayed)


def test_a_snapshot_is_unaffected_by_the_world_moving_on():
    world = World(seed=3, check_every=0)
    run(world, random_actions(200))
    snapshot = world.snapshot()
    energy = snapshot.state.total_energy()
    run(world, random_actions(200, seed=6))
    assert snapshot.state.total_energy() == energy and snapshot.tick == 200


@pytest.mark.slow
def test_saving_and_loading_continues_exactly_where_it_left_off(tmp_path):
    actions = random_actions(BEFORE + AFTER)
    world = World(seed=3, check_every=0)
    run(world, actions[:BEFORE])
    world.save(tmp_path / "world.npz")
    run(world, actions[BEFORE:])

    loaded = World.load(tmp_path / "world.npz", check_every=0)
    assert loaded.tick == BEFORE
    run(loaded, actions[BEFORE:])
    assert_same(world, loaded)
