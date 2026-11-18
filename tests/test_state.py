"""The world's bookkeeping: energy can only enter or leave through a named line, things are
read-only views that follow their block, and nudges are the only way to change the world by hand."""

from __future__ import annotations

import numpy as np
import pytest

from world.state import DEFAULT_RECIPES, Kind, WorldState


def small_world() -> WorldState:
    state = WorldState((6, 6, 4))
    for x in range(6):
        for y in range(6):
            state.spawn("bedrock", (x, y, 0), source="start", warmth=0.3)
    return state


def put(state, kind, pos, **kw):
    return state.spawn(kind, pos, source="start", warmth=0.3, **kw)


# --- the energy books ---------------------------------------------------------------


def test_spawning_from_start_is_booked_and_balanced():
    state = small_world()
    state.check()
    assert state.total_energy() == pytest.approx(state.books["start"])


def test_spawn_paid_by_a_parent_takes_its_energy():
    state = small_world()
    parent = put(state, "soil", (2, 2, 1))
    before = parent.energy
    child = state.spawn("moss", (2, 2, 2), source=parent.id, energy=0.4)
    assert parent.energy == pytest.approx(before - 0.4) and child.energy == pytest.approx(0.4)
    state.check()
    with pytest.raises(ValueError):
        state.spawn("moss", (3, 3, 1), source=parent.id, energy=1e6)  # can't pay
    state.check()


def test_remove_splits_energy_equally_among_receivers():
    state = small_world()
    a, b, gone = put(state, "soil", (1, 1, 1)), put(state, "soil", (3, 3, 1)), put(state, "soil", (2, 2, 1))
    ea, eb, amount = a.energy, b.energy, gone.energy
    state.remove(gone.id, to=[a.id, b.id])
    assert a.energy == pytest.approx(ea + amount / 2) and b.energy == pytest.approx(eb + amount / 2)
    state.check()


def test_remove_needs_somewhere_for_the_energy_to_go():
    state = small_world()
    gone = put(state, "soil", (2, 2, 1))
    with pytest.raises(ValueError):
        state.remove(gone.id, to=[])
    with pytest.raises(ValueError):
        state.remove(gone.id, to="sun")  # only start or nudge, or other things


def test_transfer_moves_energy_and_never_below_zero():
    state = small_world()
    a, b = put(state, "soil", (1, 1, 1)), put(state, "soil", (2, 1, 1))
    total = state.total_energy()
    state.transfer(a.id, b.id, 0.5)
    state.transfer(a.id, b.id, -0.2)  # negative goes the other way
    assert state.total_energy() == pytest.approx(total)
    with pytest.raises(ValueError):
        state.transfer(a.id, b.id, 1e6)
    state.check()


def test_move_energy_refuses_anything_that_creates_energy_or_hits_empty_spots():
    state = small_world()
    soil = put(state, "soil", (1, 1, 1))
    delta = np.zeros(state.shape)
    delta[1, 1, 1] = 1.0
    with pytest.raises(AssertionError):
        state.move_energy({("block", "energy"): delta})  # doesn't add up to zero
    delta = np.zeros(state.shape)
    delta[1, 1, 1], delta[4, 4, 3] = -0.1, 0.1  # (4, 4, 3) is empty
    with pytest.raises(AssertionError):
        state.move_energy({("block", "energy"): delta})
    assert soil.energy == pytest.approx(0.3 * soil.capacity)


def test_outside_exchange_only_through_sun_or_radiation():
    state = small_world()
    delta = np.zeros(state.shape)
    delta[0, 0, 0] = 0.1
    with pytest.raises(ValueError):
        state.exchange_with_outside("block", delta, "nudge")
    state.exchange_with_outside("block", delta, "sun")
    assert state.books["sun"] == pytest.approx(0.1)
    state.check()


def test_nudges_are_logged_and_balanced():
    state = small_world()
    soil = put(state, "soil", (1, 1, 1))
    state.nudge_energy(soil.id, 2.0)
    state.nudge_property(soil.id, "mass", 0.9)
    state.nudge_recipe("crystal", refill_rate=0.07)
    assert [entry["nudge"] for entry in state.log] == ["energy", "property", "recipe"]
    assert soil.mass == pytest.approx(0.9) and state.recipe("crystal").refill_rate == 0.07
    state.check()
    with pytest.raises(ValueError):
        state.nudge_property(soil.id, "energy", 5.0)  # energy only through nudge_energy


# --- entities -------------------------------------------------------------------------


def test_entities_are_read_only_follow_their_block_and_fail_loudly_when_gone():
    state = small_world()
    soil = put(state, "soil", (1, 1, 1))
    with pytest.raises(AttributeError):
        soil.energy = 5  # type: ignore[misc]
    state.move("block", (1, 1, 1), (2, 1, 1))
    assert soil.pos == (2, 1, 1)
    state.remove(soil.id, to="start")
    assert not soil.exists
    with pytest.raises(LookupError):
        _ = soil.energy


def test_neighbours_are_mutual():
    state = small_world()
    put(state, "soil", (2, 2, 1))
    put(state, "soil", (3, 2, 1))
    put(state, "moss", (2, 2, 2))
    state.spawn("body", (3, 2, 2), source="start", warmth=0.3)
    for entity_id in state.where:
        for other in state.neighbours(entity_id):
            assert entity_id in state.neighbours(other)


def test_has_room_follows_the_layer_sharing_rules():
    state = small_world()
    put(state, "soil", (2, 2, 1))
    put(state, "moss", (2, 2, 2))
    assert state.has_room("creature", (2, 2, 2))  # a creature can stand on moss
    assert not state.has_room("block", (2, 2, 2))  # a block can't share with moss
    assert not state.has_room("cover", (2, 2, 1))  # nor moss with a block


# --- recipes --------------------------------------------------------------------------


def test_recipes_refuse_inconsistent_numbers():
    with pytest.raises(ValueError):
        Kind("thing", "block", mass=(1, 0), capacity=(1, 0), conductivity=(1, 0), store_size=(1, 0))  # no pop point
    with pytest.raises(ValueError):
        Kind("thing", "block", mass=(1, 0), capacity=(1, 0), conductivity=(1, 0), consumed_on_pop=True)  # no store
    with pytest.raises(ValueError):
        Kind("thing", "cover", mass=(1, 0), capacity=(1, 0), conductivity=(1, 0), store_size=(1, 0),
             release_at=(1, 0), grow_at=0.5, grow_cost=0.9, die_at=0.1)  # a newborn would spread at once


def test_every_default_recipe_is_valid_and_unlisted_features_are_off():
    state = small_world()
    for kind in DEFAULT_RECIPES[1:]:
        values = kind.sample(np.random.default_rng(0))
        assert all(v > 0 or prop in ("store_size", "release_at") for prop, v in values.items())
    soil = put(state, "soil", (1, 1, 1))
    assert soil.store_size == 0 and soil.soften_at == np.inf
