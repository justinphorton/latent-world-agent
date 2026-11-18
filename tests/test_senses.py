"""What the agent perceives: surfaces only, the window really cuts off, the two formats always
agree, and the hard settings never leak exact values or ids."""

from __future__ import annotations

import numpy as np

from world.senses import EASY, GRID_CHANNELS, LAYER_ORDER, SenseSettings, observe, to_grid, to_list
from world.state import WorldState
from world.world import World

BODY = (3, 3, 3)


def make_world(size=(8, 8, 6)) -> tuple[WorldState, int]:
    """Bedrock, then two layers of soil, with the body standing on top at (3, 3, 3)."""
    state = WorldState(size)
    for x in range(size[0]):
        for y in range(size[1]):
            state.spawn("bedrock", (x, y, 0), source="start", warmth=0.3)
            state.spawn("soil", (x, y, 1), source="start", warmth=0.3)
            state.spawn("soil", (x, y, 2), source="start", warmth=0.3)
    body = state.spawn("body", BODY, source="start", warmth=0.5)
    return state, body.id


def seen_positions(observation, layer=None):
    return {thing.pos for thing in observation.seen if layer is None or thing.layer == layer}


def test_buried_blocks_are_never_seen():
    state, body = make_world()
    obs = observe(state, body, 0, EASY)
    assert (3, 3, 1) not in seen_positions(obs)  # surrounded on all sides by blocks
    assert (3, 3, 2) in seen_positions(obs)  # the ground surface
    # dig it out and the block below shows up
    state.remove(state.entity_at("block", (4, 3, 2)).id, to="start")
    assert (4, 3, 1) in seen_positions(observe(state, body, 0, EASY))


def test_moss_and_the_body_itself_are_seen():
    state, body = make_world()
    state.spawn("moss", (4, 3, 3), source="start", warmth=0.3)
    obs = observe(state, body, 0, EASY)
    assert (4, 3, 3) in seen_positions(obs, "cover") and BODY in seen_positions(obs, "creature")


def test_the_window_really_cuts_off():
    state, body = make_world()
    settings = SenseSettings(window=(3, 3), below=1, above=1)
    obs = observe(state, body, 0, settings)
    assert obs.size == (3, 3, 3) and obs.lo == (2, 2, 2)
    for x, y, z in seen_positions(obs):
        assert 2 <= x <= 4 and 2 <= y <= 4 and 2 <= z <= 4
    assert (2, 2, 2) in seen_positions(obs) and (5, 3, 2) not in seen_positions(obs)


def test_hard_settings_never_leak_exact_values_or_ids():
    state, body = make_world()
    obs = observe(state, body, 0, SenseSettings(whole_island=True))
    assert obs.seen
    for thing in obs.seen:
        assert thing.id is None
        assert float(thing.tone).is_integer() and 0 <= thing.tone <= 7
        assert float(thing.glow).is_integer() and 0 <= thing.glow <= 9


def test_easy_settings_show_exact_values_and_ids():
    state, body = make_world()
    obs = observe(state, body, 0, EASY)
    soil = state.entity_at("block", (3, 3, 2))
    thing = next(t for t in obs.seen if t.pos == (3, 3, 2))
    assert thing.id == soil.id and thing.tone == soil.tone and thing.glow == soil.warmth


def test_glow_levels_rise_with_warmth():
    state, body = make_world()
    soil = state.entity_at("block", (4, 3, 2))
    before = next(t.glow for t in observe(state, body, 0).seen if t.pos == (4, 3, 2))
    state.nudge_energy(soil.id, 4.0)
    after = next(t.glow for t in observe(state, body, 0).seen if t.pos == (4, 3, 2))
    assert after > before


def test_grid_and_list_always_show_the_same_facts():
    world = World(seed=0, check_every=0)
    for _ in range(50):
        world.step()
    obs = observe(world.state, world.body.entity_id, world.tick)
    grid, entries = to_grid(obs), to_list(obs)
    present = sum(int(grid[i * len(GRID_CHANNELS)].sum()) for i in range(len(LAYER_ORDER)))
    assert present == len(entries)
    for entry in entries:
        offset = tuple(p - l for p, l in zip(entry["pos"], obs.lo))
        base = LAYER_ORDER.index(entry["layer"]) * len(GRID_CHANNELS)
        assert grid[(base, *offset)] == 1
        assert [grid[(base + k, *offset)] for k in (1, 2, 3)] == [entry["color"], entry["tone"], entry["glow"]]


def test_the_body_is_always_at_the_same_place_in_the_grid():
    world = World(seed=0, check_every=0)
    settings = SenseSettings()
    centre = (settings.window[0] // 2, settings.window[1] // 2, settings.below)
    obs = observe(world.state, world.body.entity_id, 0, settings)
    creature = LAYER_ORDER.index("creature") * len(GRID_CHANNELS)
    assert to_grid(obs)[(creature, *centre)] == 1


def test_time_of_day_position_and_energy_are_always_known():
    world = World(seed=0, check_every=0)
    for _ in range(450):
        world.step()
    obs = observe(world.state, world.body.entity_id, world.tick)
    assert obs.time_of_day == 0.5 and obs.position == world.body.pos
    assert np.isclose(obs.energy, world.body.entity.energy)
