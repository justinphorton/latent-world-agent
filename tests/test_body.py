"""The body's edge cases: one small, readable test per weird situation.

Each test builds a tiny flat world (bedrock floor, one layer of soil, the body standing on it at
(3, 3, 2)), sets up a situation, acts, lets gravity run, and checks the result. Every action
also checks the whole state (bookkeeping and energy books) via `do`.

Directions: e = +x, w = -x, n = +y, s = -y. Things are made at their recipe's usual values.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.agents import RandomAgent
from world import laws
from world.body import HIT_ENERGY, LIFT_STRENGTH, PUSH_STRENGTH, Action, Body
from world.state import WorldState
from world.world import World

BODY = (3, 3, 2)


def make_world(size=(8, 8, 6), body_at=BODY) -> tuple[WorldState, Body]:
    state = WorldState(size)
    for x in range(size[0]):
        for y in range(size[1]):
            state.spawn("bedrock", (x, y, 0), source="start", warmth=0.3)
            state.spawn("soil", (x, y, 1), source="start", warmth=0.3)
    body = state.spawn("body", body_at, source="start", warmth=0.5)
    return state, Body(state, body.id)


def put(state: WorldState, kind: str, pos, **values):
    return state.spawn(kind, pos, source="start", warmth=0.3, **values)


def do(body: Body, verb: str, direction: str | None = None):
    """Act, let gravity run, and check that everything is still consistent."""
    feedback = body.act(Action(verb, direction))
    laws.settle(body.state)
    body.state.check()
    return feedback


def exhaust(state: WorldState, body: Body) -> None:
    state.nudge_energy(body.entity_id, -1e9)


def fill_store(state: WorldState, thing) -> None:
    """Fill a thing's store, paid for by a nudge, keeping its loose energy as it was."""
    need = thing.store_size - thing.stored
    state.nudge_energy(thing.id, need)
    layer, pos = state.locate(thing.id)
    delta = np.zeros(state.shape)
    delta[pos] = need
    state.move_energy({(layer, "energy"): -delta, (layer, "stored"): delta})


# --- move -------------------------------------------------------------------


def test_walk_onto_moss_shares_its_spot():
    state, body = make_world()
    moss = put(state, "moss", (4, 3, 2))
    fb = do(body, "move", "e")
    assert fb.success and body.pos == (4, 3, 2) and moss.pos == (4, 3, 2)
    assert fb.warmth is not None  # the feet feel the ground


def test_climb_while_carrying_needs_headroom():
    state, body = make_world()
    put(state, "sponge", (3, 4, 2))
    do(body, "pick_up", "n")
    put(state, "soil", (4, 3, 2))  # a one-step climb
    assert do(body, "move", "e").moved and body.pos == (4, 3, 3) and body.carried() == [("block", (4, 3, 4))]

    state, body = make_world()
    put(state, "sponge", (3, 4, 2))
    do(body, "pick_up", "n")
    put(state, "soil", (4, 3, 2))
    put(state, "soil", (4, 3, 4), mass=0.5)  # floating ledge above the step: no headroom
    laws.settle(state)
    fb = body.act(Action("move", "e"))
    assert not fb.moved


def test_wall_two_high_blocks_with_full_strain():
    state, body = make_world()
    put(state, "soil", (4, 3, 2))
    put(state, "soil", (4, 3, 3))
    fb = do(body, "move", "e")
    assert not fb.moved and fb.strain == 1.0 and body.pos == BODY


def test_drop_off_a_ledge_while_carrying():
    state, body = make_world()
    put(state, "soil", (3, 4, 2))
    do(body, "pick_up", "n")
    do(body, "put_down", "here")  # pillar up: now standing at z = 3
    assert body.pos == (3, 3, 3)
    put(state, "sponge", (4, 3, 2))
    do(body, "pick_up", "e")  # reaches down one level
    fb = do(body, "move", "w")
    assert fb.moved and body.pos == (2, 3, 2) and body.carried() == [("block", (2, 3, 3))]


def test_cannot_walk_off_the_island():
    state, body = make_world(body_at=(0, 3, 2))
    fb = do(body, "move", "w")
    assert not fb.moved and body.pos == (0, 3, 2)


def test_exhausted_body_walks_level_but_cannot_climb():
    state, body = make_world()
    exhaust(state, body)
    assert do(body, "move", "e").moved  # level walking is free when exhausted
    assert do(body, "move", "e").moved  # every tick, not every other tick
    put(state, "soil", (6, 3, 2))
    assert not do(body, "move", "e").moved


# --- push -------------------------------------------------------------------


def test_push_moves_a_whole_row_within_the_limit():
    state, body = make_world()
    a, b = put(state, "soil", (4, 3, 2)), put(state, "soil", (5, 3, 2))
    fb = do(body, "push", "e")
    assert fb.success and a.pos == (5, 3, 2) and b.pos == (6, 3, 2) and body.pos == (4, 3, 2)
    assert fb.strain == pytest.approx((a.mass + b.mass) / PUSH_STRENGTH)


def test_push_too_heavy_wont_budge():
    state, body = make_world()
    a = put(state, "stone", (4, 3, 2), mass=2.0)
    put(state, "stone", (5, 3, 2), mass=1.5)
    fb = do(body, "push", "e")
    assert not fb.success and fb.strain == 1.0 and a.pos == (4, 3, 2) and body.pos == BODY


def test_push_into_a_gap_and_it_falls():
    state, body = make_world()
    state.remove(state.entity_at("block", (5, 3, 1)).id, to="start")
    block = put(state, "soil", (4, 3, 2))
    do(body, "push", "e")
    assert block.pos == (5, 3, 1)


def test_push_refused_off_the_edge_or_into_a_creature():
    state, body = make_world(body_at=(5, 3, 2))
    block = put(state, "soil", (6, 3, 2))
    put(state, "soil", (7, 3, 2))  # the row reaches the island's last column
    assert not do(body, "push", "e").success and block.pos == (6, 3, 2)

    state, body = make_world()
    block = put(state, "soil", (4, 3, 2))
    put(state, "body", (5, 3, 2))
    assert not do(body, "push", "e").success and block.pos == (4, 3, 2)


def test_whatever_rests_on_the_row_rides_along_and_counts_toward_the_weight():
    state, body = make_world()
    bottom = put(state, "soil", (4, 3, 2))
    top = put(state, "sponge", (4, 3, 3))
    moss = put(state, "moss", (4, 3, 4))
    fb = do(body, "push", "e")
    assert bottom.pos == (5, 3, 2) and top.pos == (5, 3, 3) and moss.pos == (5, 3, 4)
    assert body.pos == (4, 3, 2) and body.carried() == []  # nothing lands on your head
    assert fb.strain == pytest.approx((bottom.mass + top.mass + moss.mass) / PUSH_STRENGTH)

    state, body = make_world()
    light = put(state, "soil", (4, 3, 2))
    put(state, "stone", (4, 3, 3), mass=1.4)
    put(state, "stone", (4, 3, 4), mass=1.4)  # the pile is too heavy, though the row alone isn't
    fb = do(body, "push", "e")
    assert not fb.success and fb.strain == 1.0 and light.pos == (4, 3, 2)


def test_push_while_carrying_takes_the_load_along():
    state, body = make_world()
    held = put(state, "sponge", (3, 4, 2))
    do(body, "pick_up", "n")
    put(state, "soil", (4, 3, 2))
    assert do(body, "push", "e").success
    assert body.pos == (4, 3, 2) and held.pos == (4, 3, 3)


def test_push_onto_moss_buries_it():
    state, body = make_world()
    block = put(state, "soil", (4, 3, 2))
    moss = put(state, "moss", (5, 3, 2))
    before = block.energy + moss.energy
    do(body, "push", "e")
    assert not moss.exists and block.pos == (5, 3, 2)
    assert block.energy >= before  # the moss's energy went into the block (plus the push's heat)


# --- pick up ----------------------------------------------------------------


def test_pick_up_strain_and_weight_limit():
    state, body = make_world()
    light = put(state, "stone", (4, 3, 2), mass=1.2)
    fb = do(body, "pick_up", "e")
    assert fb.success and fb.strain == pytest.approx(1.2 / LIFT_STRENGTH) and light.pos == (3, 3, 3)

    state, body = make_world()
    heavy = put(state, "stone", (4, 3, 2), mass=1.6)
    fb = do(body, "pick_up", "e")
    assert not fb.success and fb.strain == 1.0 and heavy.pos == (4, 3, 2)


def test_pick_up_fails_with_full_hands_or_no_headroom():
    state, body = make_world()
    put(state, "sponge", (4, 3, 2))
    put(state, "sponge", (2, 3, 2))
    do(body, "pick_up", "e")
    assert not do(body, "pick_up", "w").success

    state, body = make_world(size=(8, 8, 3))  # the head is at the top of the world
    put(state, "sponge", (4, 3, 2))
    assert not do(body, "pick_up", "e").success


def test_carried_moss_does_not_grow():
    state, body = make_world()
    moss = put(state, "moss", (4, 3, 2))
    do(body, "pick_up", "e")
    assert moss.pos == (3, 3, 3)
    state.nudge_energy(moss.id, 2.0)  # very warm
    fill_store(state, moss)  # and mature
    count = len(state.where)
    laws.life(state, World(0, check_every=0).rng)
    assert len(state.where) == count


def test_pick_up_a_block_with_something_on_top():
    state, body = make_world()
    put(state, "soil", (4, 3, 2))
    top = put(state, "sponge", (4, 3, 3))
    do(body, "pick_up", "e")
    assert top.pos == (4, 3, 2)  # it fell into the gap


def test_pick_up_here_digs_out_the_block_underfoot():
    state, body = make_world()
    soil = state.entity_at("block", (3, 3, 1))
    do(body, "pick_up", "here")
    assert body.pos == (3, 3, 1) and soil.pos == (3, 3, 2)


def test_cannot_pick_up_a_creature():
    state, body = make_world()
    other = put(state, "body", (4, 3, 2))
    assert not do(body, "pick_up", "e").success and other.pos == (4, 3, 2)


# --- put down ---------------------------------------------------------------


def test_put_down_in_front_lands_falls_or_goes_on_top():
    state, body = make_world()
    held = put(state, "sponge", (3, 4, 2))
    do(body, "pick_up", "n")
    do(body, "put_down", "e")
    assert held.pos == (4, 3, 2)

    state, body = make_world()
    state.remove(state.entity_at("block", (4, 3, 1)).id, to="start")
    held = put(state, "sponge", (3, 4, 2))
    do(body, "pick_up", "n")
    do(body, "put_down", "e")
    assert held.pos == (4, 3, 1)  # fell into the gap

    state, body = make_world()
    put(state, "soil", (4, 3, 2))  # one step high
    held = put(state, "sponge", (3, 4, 2))
    do(body, "pick_up", "n")
    do(body, "put_down", "e")
    assert held.pos == (4, 3, 3)


def test_put_block_on_moss_buries_it_and_moss_on_moss_merges():
    state, body = make_world()
    moss = put(state, "moss", (4, 3, 2))
    held = put(state, "sponge", (3, 4, 2))
    do(body, "pick_up", "n")
    do(body, "put_down", "e")
    assert not moss.exists and held.pos == (4, 3, 2)

    state, body = make_world()
    below = put(state, "moss", (4, 3, 2))
    carried = put(state, "moss", (3, 4, 2))
    do(body, "pick_up", "n")
    do(body, "put_down", "e")
    assert below.exists and not carried.exists  # merged into the one already there


def test_put_down_here_pillars_up():
    state, body = make_world()
    held = put(state, "soil", (3, 4, 2))
    do(body, "pick_up", "n")
    fb = do(body, "put_down", "here")
    assert fb.moved and body.pos == (3, 3, 3) and held.pos == (3, 3, 2)


def test_exhausted_body_can_still_put_things_down():
    state, body = make_world()
    held = put(state, "sponge", (3, 4, 2))
    do(body, "pick_up", "n")
    exhaust(state, body)
    fb = do(body, "put_down", "e")
    assert fb.success and fb.spent == 0 and held.pos == (4, 3, 2)


# --- hit --------------------------------------------------------------------


def test_bare_hit_jolts_energy_in_and_recoils():
    state, body = make_world()
    target = put(state, "stone", (4, 3, 2))
    before_body, before_target = body.entity.energy, target.energy
    fb = do(body, "hit", "e")
    jolt = HIT_ENERGY * body.entity.mass
    assert target.energy == pytest.approx(before_target + jolt)
    assert body.entity.energy == pytest.approx(before_body - jolt)
    assert fb.strain == pytest.approx(target.mass / PUSH_STRENGTH)


def test_hitting_with_a_block_hits_harder():
    state, body = make_world()
    tool = put(state, "stone", (3, 4, 2))
    do(body, "pick_up", "n")
    target = put(state, "soil", (4, 3, 2))
    before = target.energy
    do(body, "hit", "e")
    assert target.energy == pytest.approx(before + HIT_ENERGY * tool.mass)


def test_hitting_a_charged_crystal_sets_it_off():
    state, body = make_world()
    crystal = put(state, "crystal", (4, 3, 2))
    fill_store(state, crystal)
    assert crystal.warmth < crystal.release_at  # it wouldn't pop on its own
    do(body, "hit", "e")
    assert crystal.id in laws.pop(state)


def test_hit_fails_with_nothing_in_reach_or_when_exhausted():
    state, body = make_world(body_at=(7, 3, 2))  # at the island's edge: nothing out there
    assert not do(body, "hit", "e").success
    state, body = make_world()
    put(state, "soil", (4, 3, 2))
    exhaust(state, body)
    assert not do(body, "hit", "e").success


# --- always -----------------------------------------------------------------


def test_the_agent_only_senses_strain_warmth_and_energy():
    state, body = make_world()
    put(state, "stone", (4, 3, 2))
    sensed = do(body, "hit", "e").sensed()
    assert len(sensed) == 3 and all(isinstance(v, float) for v in sensed)


def test_a_block_falling_on_your_head_is_carried_and_can_pin_you():
    state, body = make_world()
    falling = put(state, "stone", (3, 3, 4), mass=2.0)
    laws.settle(state)
    assert falling.pos == (3, 3, 3) and body.carried() == [("block", (3, 3, 3))]
    fb = do(body, "move", "e")
    assert not fb.moved and fb.strain == 1.0  # pinned until it's put down
    assert do(body, "put_down", "e").success
    assert do(body, "move", "w").moved


def test_random_play_on_the_island_never_breaks_anything():
    """Thousands of random actions on the real island, with every invariant checked every tick."""
    world = World(seed=1, rung=3, check_every=1)
    agent = RandomAgent(seed=2)
    feedback = None
    for _ in range(3000):
        feedback = world.step(agent.act(feedback)).feedback
