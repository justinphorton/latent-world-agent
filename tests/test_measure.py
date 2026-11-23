"""The measuring tool: branch pairs must only count what the action really caused."""

from __future__ import annotations

import json

from world.body import Action
from world.measure import EVENT_KINDS, _run_branch, _seen_picture, _pictures_differ, measure, step_with_events
from world.world import World


def fresh() -> World:
    return World(seed=4, check_every=0)


def test_waiting_against_waiting_shows_no_difference():
    world = fresh()
    start = world.snapshot()
    a, _ = _run_branch(world, Action("wait"), 10)
    pa = _seen_picture(world, 10)
    world.restore(start)
    b, _ = _run_branch(world, Action("wait"), 10)
    assert a == b and not _pictures_differ(pa, _seen_picture(world, 10))


def test_picking_something_up_counts_as_an_effect_and_a_move():
    world = fresh()
    for direction in ("n", "e", "s", "w", "here"):
        start = world.snapshot()
        result, events = step_with_events(world, Action("pick_up", direction))
        if result.feedback.success:
            assert any(kind == "moved" for kind, *_ in events)
            return
        world.restore(start)
    raise AssertionError("nothing to pick up next to the body on seed 4")


def test_the_body_itself_never_counts():
    world = fresh()
    body = world.body.entity_id
    for direction in ("n", "e", "s", "w"):
        _, events = step_with_events(world, Action("move", direction))
        assert all(rest[0] != body for kind, *rest in events if kind in ("moved", "fall"))


def test_a_short_card():
    card = measure(fresh, days=0.1, every=5, horizon=5, source="seed 4")
    assert set(card.checklist) == set(EVENT_KINDS)
    assert card.verbs["wait"].effect == 0 if "wait" in card.verbs else True
    assert sum(line.tries for line in card.verbs.values()) == int(0.1 * fresh().state.tempo.day_length)
    json.loads(card.to_json())
    assert "Checklist" in card.report()
