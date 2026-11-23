"""The laws, one behaviour per test: every law keeps the books balanced on its own, and each does
what it says (flow evens warmth, the sun only warms tops by day, stores pop when full and warm and
then need to refill, moss flows downhill and pays to grow, gravity resolves fully)."""

from __future__ import annotations

import numpy as np
import pytest

from world import laws
from world.state import Tempo, WorldState
from world.world import World


def flat(size=(7, 7, 6), top="soil") -> WorldState:
    state = WorldState(size)
    for x in range(size[0]):
        for y in range(size[1]):
            state.spawn("bedrock", (x, y, 0), source="start", warmth=0.3)
            state.spawn(top, (x, y, 1), source="start", warmth=0.3)
    return state


def put(state, kind, pos, warmth=0.3, **kw):
    return state.spawn(kind, pos, source="start", warmth=warmth, **kw)


def fill_store(state, thing):
    need = thing.store_size - thing.stored
    state.nudge_energy(thing.id, need)
    layer, pos = state.locate(thing.id)
    delta = np.zeros(state.shape)
    delta[pos] = need
    state.move_energy({(layer, "energy"): -delta, (layer, "stored"): delta})


# --- every law alone ------------------------------------------------------------------


@pytest.mark.parametrize("law", laws.LAWS)
def test_each_law_alone_keeps_the_books_balanced(law):
    world = World(seed=0, laws={law}, check_every=1)  # check_every=1: books checked every tick
    for _ in range(400):
        world.step()


def test_every_rung_runs_cleanly():
    for rung in laws.RUNGS:
        world = World(seed=1, rung=rung, check_every=1)
        for _ in range(150):
            world.step()


# --- sun, flow, radiation ---------------------------------------------------------------


@pytest.mark.parametrize("tempo", [Tempo(), Tempo(day_length=300, daylight=0.5)])
def test_the_day_is_light_for_the_daylight_share_then_dark(tempo):
    levels = [laws.sun_level(t, tempo) for t in range(tempo.day_length)]
    lit = sum(level > 0 for level in levels)
    assert abs(lit / tempo.day_length - tempo.daylight) < 0.01
    assert max(levels) == pytest.approx(1.0, abs=1e-3)


def test_the_sun_warms_only_the_top_of_each_column():
    state = flat()
    day = state.tempo.day_length
    put(state, "soil", (3, 3, 2))
    under = state.entity_at("block", (3, 3, 1))
    top = state.entity_at("block", (3, 3, 2))
    before_under, before_top = under.energy, top.energy
    laws.shine(state, day // 3)  # noon
    assert top.energy > before_top and under.energy == before_under
    laws.shine(state, int(day * 0.9))  # night
    state.check()


def test_the_body_catches_more_sunlight_than_the_ground():
    state = flat()
    body = put(state, "body", (3, 3, 2))
    ground = state.entity_at("block", (5, 5, 1))
    body_before, ground_before = body.energy, ground.energy
    laws.shine(state, state.tempo.day_length // 3)  # noon
    caught = body.kind_info.sun_catch
    assert caught > 1
    assert body.energy - body_before == pytest.approx(caught * (ground.energy - ground_before))
    state.check()  # the extra still comes from the sun line of the books


def test_a_world_keeps_its_tempo_through_copy_and_save(tmp_path):
    state = flat()
    state.tempo = Tempo(day_length=300, sun_power=0.05)
    assert state.copy().tempo == state.tempo
    state.save(tmp_path / "s.npz")
    assert WorldState.load(tmp_path / "s.npz").tempo == state.tempo


def test_flow_evens_out_warmth_and_the_poorer_conductor_sets_the_pace():
    state = flat(top="bedrock")
    hot = put(state, "stone", (3, 3, 2), warmth=0.9)
    cold = put(state, "stone", (4, 3, 2), warmth=0.1)
    gap = hot.warmth - cold.warmth
    for _ in range(30):
        laws.flow(state)
    assert 0 < hot.warmth - cold.warmth < gap
    state.check()

    fast, slow = flat(top="bedrock"), flat(top="bedrock")
    for s, kind in ((fast, "stone"), (slow, "sponge")):
        put(s, "stone", (3, 3, 2), warmth=0.9)
        put(s, kind, (4, 3, 2), warmth=0.1)
        laws.flow(s)
    assert fast.entity_at("block", (3, 3, 2)).energy < slow.entity_at("block", (3, 3, 2)).energy


def test_exposed_faces_lose_more_heat_than_buried_ones():
    state = flat()
    lone = put(state, "soil", (3, 3, 2), warmth=0.6)
    put(state, "soil", (5, 5, 2), warmth=0.6)
    put(state, "soil", (5, 5, 3), warmth=0.6)
    tucked = state.entity_at("block", (5, 5, 2))
    before_lone, before_tucked = lone.energy, tucked.energy
    laws.radiate(state)
    assert before_lone - lone.energy > before_tucked - tucked.energy
    state.check()


# --- store and pop ------------------------------------------------------------------------


def test_stores_fill_from_loose_energy():
    state = flat()
    crystal = put(state, "crystal", (3, 3, 2), warmth=0.4)
    before_energy = crystal.energy
    laws.pop(state)
    assert crystal.stored > 0 and crystal.energy < before_energy
    state.check()


def test_a_crystal_pops_only_when_full_and_warm_and_then_must_refill():
    state = flat()
    crystal = put(state, "crystal", (3, 3, 2), warmth=crystal_warm(state))
    assert crystal.id not in laws.pop(state)  # warm but not full
    fill_store(state, crystal)
    neighbours = {i: state.entity_by_id(i).energy for i in state.neighbours(crystal.id)}
    assert crystal.id in laws.pop(state)  # full and warm
    assert crystal.stored == 0
    assert all(state.entity_by_id(i).energy > e for i, e in neighbours.items())  # they got the burst
    assert crystal.id not in laws.pop(state)  # can't pop again until refilled
    state.check()


def crystal_warm(state) -> float:
    return state.recipe("crystal").release_at[0] + 0.2


def test_a_line_of_crystals_pops_one_ring_per_tick():
    state = flat(top="bedrock")
    line = [put(state, "crystal", (x, 3, 2), warmth=0.4) for x in range(1, 6)]
    for crystal in line:
        fill_store(state, crystal)
    state.nudge_energy(line[0].id, 2.0)  # set off the first one
    order = []
    for _ in range(8):
        popped = laws.pop(state)
        order.append([c.pos[0] for c in line if c.id in popped])
    first_pop = {x: next(t for t, xs in enumerate(order) if x in xs) for x in range(1, 6) if any(x in xs for xs in order)}
    assert first_pop[1] == 0 and len(first_pop) >= 3
    assert all(first_pop[x] <= first_pop[x + 1] for x in range(1, max(first_pop)))  # it travels outward


def test_burning_moss_is_used_up_and_passes_everything_to_its_neighbours():
    state = flat()
    moss = put(state, "moss", (3, 3, 2), warmth=0.5)
    fill_store(state, moss)
    state.nudge_energy(moss.id, 1.0)  # a jolt past its pop point
    total = state.total_energy()
    assert moss.id in laws.pop(state)
    assert not moss.exists and state.total_energy() == pytest.approx(total)
    state.check()


# --- life ----------------------------------------------------------------------------------


def mature_moss(state, pos):
    moss = put(state, "moss", pos, warmth=0.8)
    fill_store(state, moss)
    return moss


def test_mature_well_fed_moss_spreads_and_pays_from_its_store():
    state = flat()
    moss = mature_moss(state, (3, 3, 2))
    stored = moss.stored
    laws.life(state, np.random.default_rng(0))
    assert len([i for i in state.where if state.entity_by_id(i).name == "moss"]) == 2
    assert moss.stored == pytest.approx(stored - state.recipe("moss").grow_cost)
    state.check()


def test_young_moss_does_not_spread():
    state = flat()
    put(state, "moss", (3, 3, 2), warmth=0.8)  # warm but its store is empty
    laws.life(state, np.random.default_rng(0))
    assert len([i for i in state.where if state.entity_by_id(i).name == "moss"]) == 1


def test_moss_flows_level_or_downhill_never_up():
    state = flat()
    for x in range(7):
        for y in range(7):
            if (x, y) != (3, 3):
                put(state, "soil", (x, y, 2))  # a plateau one step up all around
    mature_moss(state, (3, 3, 2))  # moss in a one-deep hole
    for seed in range(20):
        laws.life(state, np.random.default_rng(seed))
    assert [i for i in state.where if state.entity_by_id(i).name == "moss"] and all(
        state.entity_by_id(i).pos == (3, 3, 2) for i in state.where if state.entity_by_id(i).name == "moss"
    )

    down = flat()
    put(down, "soil", (3, 3, 2))
    mature_moss(down, (3, 3, 3))  # moss on a pillar one step up
    laws.life(down, np.random.default_rng(0))
    assert any(down.entity_by_id(i).pos[2] == 2 for i in down.where if down.entity_by_id(i).name == "moss")


def test_starving_moss_dies_and_feeds_what_it_touches():
    state = flat()
    moss = put(state, "moss", (3, 3, 2), warmth=0.01)
    total = state.total_energy()
    laws.life(state, np.random.default_rng(0))
    assert not moss.exists and state.total_energy() == pytest.approx(total)
    state.check()


# --- gravity --------------------------------------------------------------------------------


def test_gravity_resolves_everything_within_one_call():
    state = flat()
    a = put(state, "soil", (3, 3, 5))
    b = put(state, "sponge", (3, 3, 4))
    moss = put(state, "moss", (3, 3, 3))  # floating moss
    events = laws.settle(state)
    assert b.pos == (3, 3, 2) and a.pos == (3, 3, 3) and not moss.exists  # moss fell first, then got buried
    assert events
    for layer in ("block", "cover", "creature"):
        for pos in zip(*np.nonzero(state.occupied(layer))):
            assert laws.is_supported(state, layer, tuple(int(p) for p in pos))
    state.check()
