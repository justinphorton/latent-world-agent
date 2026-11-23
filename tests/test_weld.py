"""Welding (rung 4): softening with a gap, welds forming, welded groups standing, falling, being
pushed and pulled as one piece, and coming apart (soft welds pull apart, hard hits break them)."""

from __future__ import annotations

import numpy as np
import pytest

from world import laws
from world.body import Action, Body
from world.state import WorldState
from world.world import World

BODY = (3, 3, 2)


def make_world(size=(9, 9, 7)) -> tuple[WorldState, Body]:
    state = WorldState(size)
    for x in range(size[0]):
        for y in range(size[1]):
            state.spawn("bedrock", (x, y, 0), source="start", warmth=0.3)
            state.spawn("soil", (x, y, 1), source="start", warmth=0.3)
    body = state.spawn("body", BODY, source="start", warmth=0.5)
    return state, Body(state, body.id)


def put(state, kind, pos, **values):
    return state.spawn(kind, pos, source="start", warmth=0.3, **values)


def do(body: Body, verb: str, direction: str | None = None):
    feedback = body.act(Action(verb, direction))
    laws.settle(body.state)
    body.state.check()
    return feedback


def set_warmth(state, thing, warmth):
    state.nudge_energy(thing.id, warmth * thing.capacity - thing.energy)


def welded(state, a, b) -> bool:
    return b.pos in state.welded_to(a.pos)


# --- softening and welding ----------------------------------------------------


def test_stone_softens_hot_and_only_hardens_once_well_below():
    state, _ = make_world()
    stone = put(state, "stone", (6, 6, 2), soften_at=0.8)
    set_warmth(state, stone, 0.85)
    laws.weld(state)
    assert stone.soft
    set_warmth(state, stone, 0.75)  # cooler, but within the gap: still soft
    laws.weld(state)
    assert stone.soft
    set_warmth(state, stone, 0.65)  # well below: hard again
    laws.weld(state)
    assert not stone.soft


def test_materials_that_never_soften_stay_hard():
    state, _ = make_world()
    soil = put(state, "soil", (6, 6, 2))
    set_warmth(state, soil, 5.0)
    laws.weld(state)
    assert not soil.soft


def test_a_soft_block_welds_to_every_block_it_touches():
    state, _ = make_world()
    stone = put(state, "stone", (6, 6, 2), soften_at=0.8)
    sponge = put(state, "sponge", (7, 6, 2))
    ground = state.entity_at("block", (6, 6, 1))
    set_warmth(state, stone, 0.9)
    laws.weld(state)
    assert welded(state, stone, sponge) and welded(state, stone, ground)
    set_warmth(state, stone, 0.3)  # cooling hardens it, but the welds stay
    laws.weld(state)
    assert not stone.soft and welded(state, stone, sponge)


# --- gravity --------------------------------------------------------------------


def test_a_welded_overhang_stands():
    state, _ = make_world()
    put(state, "stone", (5, 5, 2))
    top = put(state, "stone", (5, 5, 3))
    ledge = put(state, "stone", (6, 5, 3))  # sticks out over empty air
    state.weld(top.pos, ledge.pos)
    laws.settle(state)
    assert ledge.pos == (6, 5, 3)

    state, _ = make_world()
    put(state, "stone", (5, 5, 2))
    put(state, "stone", (5, 5, 3))
    loose = put(state, "stone", (6, 5, 3))  # not welded: it falls
    laws.settle(state)
    assert loose.pos == (6, 5, 2)


def test_an_unsupported_welded_group_falls_as_one_piece():
    state, _ = make_world()
    put(state, "stone", (5, 5, 2))  # a bump under one end
    a = put(state, "stone", (5, 5, 5))
    b = put(state, "stone", (6, 5, 5))
    state.weld(a.pos, b.pos)
    laws.settle(state)
    assert a.pos == (5, 5, 3) and b.pos == (6, 5, 3)  # landed on the bump, still level
    assert welded(state, a, b)
    state.check()


def test_a_falling_group_buries_moss():
    state, _ = make_world()
    moss = put(state, "moss", (6, 5, 2))
    a, b = put(state, "stone", (5, 5, 4)), put(state, "stone", (6, 5, 4))
    state.weld(a.pos, b.pos)
    laws.settle(state)
    assert not moss.exists and b.pos == (6, 5, 2)


# --- push and pull --------------------------------------------------------------


def test_push_moves_a_welded_group_as_one_piece():
    state, body = make_world()
    front, side = put(state, "soil", (4, 3, 2)), put(state, "soil", (4, 4, 2))
    state.weld(front.pos, side.pos)
    assert do(body, "push", "e").success
    assert front.pos == (5, 3, 2) and side.pos == (5, 4, 2) and welded(state, front, side)


def test_a_soft_weld_pulls_apart_when_pushed():
    state, body = make_world()
    front, side = put(state, "soil", (4, 3, 2)), put(state, "stone", (4, 4, 2))
    state.weld(front.pos, side.pos)
    state.nudge_property(side.id, "soft", True)
    assert do(body, "push", "e").success
    assert front.pos == (5, 3, 2) and side.pos == (4, 4, 2) and not welded(state, front, side)


def test_a_block_welded_to_the_ground_wont_budge():
    state, body = make_world()
    front = put(state, "soil", (4, 3, 2))
    state.weld(front.pos, (4, 3, 1))
    fb = do(body, "push", "e")
    assert not fb.success and fb.strain == 1.0 and front.pos == (4, 3, 2)


def test_pull_steps_back_and_brings_the_block_and_its_welded_group():
    state, body = make_world()
    front, side = put(state, "soil", (4, 3, 2)), put(state, "soil", (4, 4, 2))
    state.weld(front.pos, side.pos)
    fb = do(body, "pull", "e")
    assert fb.success and body.pos == (2, 3, 2)
    assert front.pos == (3, 3, 2) and side.pos == (3, 4, 2) and welded(state, front, side)


def test_pull_brings_only_what_is_attached_not_the_rest_of_the_row():
    state, body = make_world()
    front, behind = put(state, "soil", (4, 3, 2)), put(state, "soil", (5, 3, 2))
    assert do(body, "pull", "e").success
    assert front.pos == (3, 3, 2) and behind.pos == (5, 3, 2)


def test_pull_needs_room_to_step_back():
    state, body = make_world()
    front = put(state, "soil", (4, 3, 2))
    put(state, "soil", (2, 3, 2))  # a wall behind
    fb = do(body, "pull", "e")
    assert not fb.success and front.pos == (4, 3, 2) and body.pos == BODY


def test_pulling_onto_moss_underfoot_buries_it():
    state, body = make_world()
    moss = put(state, "moss", BODY)  # the body stands on moss
    front = put(state, "soil", (4, 3, 2))
    do(body, "pull", "e")
    assert front.pos == BODY and not moss.exists


# --- breaking welds and lifting ---------------------------------------------------


def test_a_hard_hit_breaks_welds_but_bare_hands_dont():
    state, body = make_world()
    front, side = put(state, "soil", (4, 3, 2)), put(state, "soil", (4, 4, 2))
    state.weld(front.pos, side.pos)
    do(body, "hit", "e")  # bare hands: 1.5 x 0.7 = 1.05, under WELD_STRENGTH
    assert welded(state, front, side)
    put(state, "stone", (3, 2, 2), mass=1.2)
    do(body, "pick_up", "s")  # a stone to hit with: 1.5 x 1.2 = 1.8
    fb = do(body, "hit", "e")
    assert not welded(state, front, side) and "broke loose" in fb.note


def test_welded_blocks_cant_be_picked_up():
    state, body = make_world()
    front, side = put(state, "soil", (4, 3, 2)), put(state, "soil", (4, 4, 2))
    state.weld(front.pos, side.pos)
    fb = do(body, "pick_up", "e")
    assert not fb.success and front.pos == (4, 3, 2)


# --- bookkeeping ------------------------------------------------------------------


def test_copy_and_save_keep_welds(tmp_path):
    state, _ = make_world()
    a, b = put(state, "stone", (5, 5, 2)), put(state, "stone", (6, 5, 2))
    state.weld(a.pos, b.pos)
    assert (6, 5, 2) in state.copy().welded_to((5, 5, 2))
    state.save(tmp_path / "s.npz")
    assert (6, 5, 2) in WorldState.load(tmp_path / "s.npz").welded_to((5, 5, 2))


def test_check_catches_a_weld_that_doesnt_join_two_blocks():
    state, _ = make_world()
    state.welds[0][5, 5, 4] = True  # floating in the air
    with pytest.raises(AssertionError):
        state.check()


def test_moving_or_removing_a_block_on_its_own_breaks_its_welds():
    state, _ = make_world()
    a, b = put(state, "stone", (5, 5, 2)), put(state, "stone", (6, 5, 2))
    state.weld(a.pos, b.pos)
    state.move("block", a.pos, (5, 5, 3))
    assert not state.has_welds(b.pos)
    state.weld(b.pos, (6, 5, 1))
    state.remove(b.id, to="start")
    assert not state.has_welds((6, 5, 1))
    state.check()


# --- the law in the running world -----------------------------------------------


def _hot_stone(world):
    stone = next(world.state.entity_by_id(i) for i in world.state.where if world.state.entity_by_id(i).name == "stone")
    world.state.nudge_energy(stone.id, 20.0)
    return stone


def test_welding_only_happens_with_the_weld_law_on():
    on, off = World(seed=0, rung=4, check_every=0), World(seed=0, rung=3, check_every=0)
    hot_on, hot_off = _hot_stone(on), _hot_stone(off)
    on.step(), off.step()
    assert hot_on.soft and on.state.has_welds(hot_on.pos)
    assert not hot_off.soft and not any(w.any() for w in off.state.welds)


def test_sunlight_alone_never_softens_stone():
    # pops off: a crystal popping against stone can soften it, and that's allowed (it's heat, not sunlight)
    world = World(seed=0, laws=laws.RUNGS[4] - {"pop"}, check_every=0)
    for _ in range(2 * world.state.tempo.day_length):
        world.step()
    assert not world.state.layers["block"]["soft"].any()
    assert not any(w.any() for w in world.state.welds)
