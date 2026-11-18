"""Every seed makes a valid island with every ingredient, and different seeds make different maps."""

from __future__ import annotations

import pytest

from world.worldgen import CRYSTAL_GROUPS, build_island


def kinds_on(state) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entity_id in state.where:
        name = state.entity_by_id(entity_id).name
        counts[name] = counts.get(name, 0) + 1
    return counts


@pytest.mark.parametrize("seed", range(12))
def test_every_seed_gets_every_ingredient(seed):
    state, body_id = build_island(seed)
    state.check()  # consistent, and the energy books balance
    counts = kinds_on(state)
    for kind in ("bedrock", "soil", "stone", "sponge", "crystal", "moss", "body"):
        assert counts.get(kind, 0) > 0, f"seed {seed} has no {kind}"
    assert counts["body"] == 1
    assert counts["crystal"] >= 5 + CRYSTAL_GROUPS  # the cluster plus scattered groups
    body = state.entity_by_id(body_id)
    assert body.pos[:2] == (16, 16) and state.entity_at("block", (16, 16, body.pos[2] - 1)) is not None


def test_different_seeds_make_different_maps():
    def layout(seed):
        state, _ = build_island(seed)
        return {(state.entity_by_id(i).name, state.entity_by_id(i).pos) for i in state.where}

    assert layout(0) != layout(1)
    assert layout(0) == layout(0)  # and the same seed always makes the same map
