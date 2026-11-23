"""How eventful a world is, in numbers: the card from context/worldgen-redesign.md.

Three measurements, all by running the world:

1. Background activity: the body only waits for N days; count discrete events per day.
2. Agent effect: a random agent plays for N days. Every `every` ticks the world is snapshotted
   and split in two: one branch takes the agent's chosen action, the other waits; both then wait
   for `horizon` ticks. Because the world is deterministic and restore is exact, any difference
   between the branches was caused by the action alone. The branches are compared in island
   coordinates (so a step that only shifts the agent's window counts as nothing), leaving the
   body itself out, over every visible thing: a difference is a thing present in one branch and
   not the other, or a thing whose glow sits at a different level (the agent's own glow levels,
   so drift too small to see doesn't count). Reported per verb.
3. Reachability checklist: from the same branch pairs, which kinds of discrete event (a block
   moved, a fall, a pop, a weld made or broken, moss born or died) happened in the action branch
   and not in the wait branch, at least once.

Two leaks are counted rather than hidden:
  - dice: if the action changes where moss can grow, the two branches may draw different
    numbers from the world's dice, and moss can then differ anywhere for no real reason. Pairs
    whose dice ended in different states are counted per verb.
  - shade: the body shades the top of whatever it stands on, so a daytime step can change glow
    under the old and new spot. Effects are split into day and night per verb.

A separate health check runs the world untouched for many days and prints one line per day
(moss alive, born, burned, starved, pops, welds, the biggest welded group, energy), to see
whether the world stays balanced over the long run.

Run it:  python -m world.measure [worlds/starter.npz] [--days 3] [--rung 4]
         python -m world.measure [worlds/starter.npz] --health 20
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from agents.agents import RandomAgent
from world.body import Action
from world.laws import RUNGS
from world.senses import SenseSettings, visible_mask
from world.world import StepResult, World

EVENT_KINDS = ("moved", "fall", "pop", "weld_made", "weld_broken", "moss_born", "moss_died")
WAIT = Action("wait")


@dataclass
class VerbLine:
    """One verb's row on the card."""

    tries: int = 0  # how often the random agent chose it (whole run)
    succeeded: int = 0  # how often the body reported success
    sampled: int = 0  # branch pairs taken for this verb
    effect: int = 0  # pairs where the action changed something the agent can see
    effect_day: int = 0
    sampled_day: int = 0
    dice_split: int = 0  # pairs whose dice ended up in different states
    caused: Counter = field(default_factory=Counter)  # event kind -> pairs where the action caused it


@dataclass
class Card:
    """The measurements for one world."""

    source: str
    laws: list[str]
    days: float
    background_per_day: dict[str, float]  # event kind -> events per day while only waiting
    verbs: dict[str, VerbLine]
    checklist: dict[str, bool]  # event kind -> did random play cause it at least once
    moss_at_end: int  # moss alive at the end of the random run (life sanity check)

    def to_json(self) -> str:
        verbs = {verb: {**vars(line), "caused": dict(line.caused)} for verb, line in self.verbs.items()}
        return json.dumps({**vars(self), "verbs": verbs}, indent=2)

    def report(self) -> str:
        """The card as a short readable table."""
        out = [f"{self.source}   laws: {', '.join(self.laws)}   {self.days:g} days each", ""]
        out.append("Background (body waiting), events per day:")
        out.append("  " + "   ".join(f"{k} {v:.1f}" for k, v in self.background_per_day.items()))
        out.append("")
        out.append(f"{'verb':9}{'tries':>7}{'success':>9}{'pairs':>7}{'effect':>8}{'day':>7}{'night':>7}{'dice':>6}  caused")
        for verb, line in self.verbs.items():
            night_n = line.sampled - line.sampled_day
            out.append(
                f"{verb:9}{line.tries:>7}{_pct(line.succeeded, line.tries):>9}{line.sampled:>7}"
                f"{_pct(line.effect, line.sampled):>8}{_pct(line.effect_day, line.sampled_day):>7}"
                f"{_pct(line.effect - line.effect_day, night_n):>7}{line.dice_split:>6}  "
                + ", ".join(f"{k} {n}" for k, n in sorted(line.caused.items()))
            )
        out.append("")
        out.append("Checklist (caused by random play at least once):")
        out.append("  " + "   ".join(f"{k} {'yes' if v else 'NO'}" for k, v in self.checklist.items()))
        out.append(f"Moss alive at the end: {self.moss_at_end}")
        return "\n".join(out)


def _pct(part: int, whole: int) -> str:
    return f"{100 * part / whole:.0f}%" if whole else "-"


# ---------------------------------------------------------------------------
# Events: what discrete things happened in one tick
# ---------------------------------------------------------------------------


def step_with_events(world: World, action: Action) -> tuple[StepResult, set[tuple]]:
    """Advance one tick and return what discrete events happened, as comparable keys.

    Keys name things by id where the thing existed before (so the same event in two branches
    gets the same key), and by position where it is new (new moss gets a fresh id that can
    differ between branches). The body itself never counts.
    """
    state = world.state
    body = world.body.entity_id
    blocks_before = state.layers["block"]["id"].copy()
    cover_before = state.layers["cover"]["id"].copy()
    welds_before = [w.copy() for w in state.welds]

    result = world.step(action)
    events: set[tuple] = set()

    blocks_after = state.layers["block"]["id"]
    changed = blocks_before != blocks_after
    for entity_id in np.unique(blocks_after[changed]):
        if entity_id != 0 and entity_id != body:
            events.add(("moved", int(entity_id)))
    for fall in result.events:
        if fall.entity_id != body:
            events.add(("fall", fall.entity_id))
    for entity_id in result.popped:
        events.add(("pop", entity_id))

    cover_after = state.layers["cover"]["id"]
    before_ids, after_ids = set(np.unique(cover_before)) - {0}, set(np.unique(cover_after)) - {0}
    for entity_id in before_ids - after_ids:
        events.add(("moss_died", int(entity_id)))
    born = np.isin(cover_after, list(after_ids - before_ids)) if after_ids - before_ids else None
    if born is not None:
        for pos in zip(*np.nonzero(born)):
            events.add(("moss_born", tuple(int(p) for p in pos)))

    for axis, (old, new) in enumerate(zip(welds_before, state.welds)):
        for pos in zip(*np.nonzero(new & ~old)):
            events.add(("weld_made", axis, tuple(int(p) for p in pos)))
        for pos in zip(*np.nonzero(old & ~new)):
            events.add(("weld_broken", axis, tuple(int(p) for p in pos)))
    return result, events


# ---------------------------------------------------------------------------
# What the agent could tell apart between two branches
# ---------------------------------------------------------------------------


def _seen_picture(world: World, glow_levels: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per layer: which material is visible at each spot (0 = nothing seen) and its glow level.
    The body is left out."""
    state = world.state
    visible = visible_mask(state)
    picture = {}
    for layer, arrays in state.layers.items():
        kind = np.where(visible[layer], arrays["kind"], 0)
        if layer == "creature":
            kind = np.where(arrays["id"] == world.body.entity_id, 0, kind)
        capacity = arrays["capacity"]
        warmth = np.divide(arrays["energy"], capacity, out=np.zeros_like(capacity, dtype=float), where=capacity > 0)
        level = np.minimum(glow_levels - 1, (np.clip(warmth, 0.0, 1.0) * glow_levels).astype(int))
        picture[layer] = (kind, np.where(kind != 0, level, 0))
    return picture


def _pictures_differ(a: dict, b: dict) -> bool:
    return any(not (np.array_equal(a[l][0], b[l][0]) and np.array_equal(a[l][1], b[l][1])) for l in a)


def _run_branch(world: World, first: Action, horizon: int) -> tuple[set[tuple], dict]:
    """Take `first`, then wait until `horizon` ticks have passed. Returns every event and the
    world's dice state at the end."""
    events: set[tuple] = set()
    for i in range(horizon):
        _, happened = step_with_events(world, first if i == 0 else WAIT)
        events |= happened
    return events, world.rng.bit_generator.state


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------


def measure(make_world, days: float = 3, every: int = 10, horizon: int = 10, agent_seed: int = 0,
            glow_levels: int | None = None, source: str = "") -> Card:
    """Measure a world. `make_world()` must return a fresh copy of the world each call (it is
    called twice: once for the waiting run, once for the random run)."""
    glow_levels = glow_levels or SenseSettings().glow_levels or 10
    # 1. Background: the body only waits.
    world = make_world()
    ticks = int(days * world.state.tempo.day_length)
    background: Counter = Counter()
    for _ in range(ticks):
        _, events = step_with_events(world, WAIT)
        background.update(kind for kind, *_ in events)

    # 2 and 3. A random agent, with branch pairs every `every` ticks.
    world = make_world()
    agent = RandomAgent(agent_seed)
    verbs: dict[str, VerbLine] = defaultdict(VerbLine)
    for t in range(ticks):
        action = agent.act(None)
        line = verbs[action.verb]
        if t % every == 0:
            daytime = world.sun > 0
            start = world.snapshot()
            acted, dice_acted = _run_branch(world, action, horizon)
            picture_acted = _seen_picture(world, glow_levels)
            world.restore(start)
            waited, dice_waited = _run_branch(world, WAIT, horizon)
            picture_waited = _seen_picture(world, glow_levels)
            world.restore(start)

            line.sampled += 1
            line.sampled_day += daytime
            if _pictures_differ(picture_acted, picture_waited):
                line.effect += 1
                line.effect_day += daytime
            line.dice_split += dice_acted != dice_waited
            line.caused.update({kind for kind, *_ in acted - waited})
        result = world.step(action)
        line.tries += 1
        line.succeeded += result.feedback.success

    caused_any = Counter()
    for line in verbs.values():
        caused_any.update(line.caused)
    ordered = {verb: verbs[verb] for verb in ("move", "push", "pull", "pick_up", "put_down", "hit", "wait") if verb in verbs}
    return Card(
        source=source,
        laws=sorted(world.laws_on),
        days=days,
        background_per_day={k: background[k] / days for k in EVENT_KINDS},
        verbs=ordered,
        checklist={k: caused_any[k] > 0 for k in EVENT_KINDS},
        moss_at_end=int(world.state.occupied("cover").sum()),
    )


# ---------------------------------------------------------------------------
# Health: does the world stay balanced over many days with nobody playing?
# ---------------------------------------------------------------------------


@dataclass
class DayLine:
    """One day of a long, untouched run."""

    day: int
    moss: int  # moss alive at the end of the day
    born: int  # moss born during the day
    burned: int  # moss that died by popping (fire)
    starved: int  # moss that died any other way (cold, buried)
    pops: int  # everything that popped
    welds: int  # welds in the world at the end of the day
    biggest_group: int  # blocks in the largest welded group
    body_energy: float
    total_energy: float


def health(make_world, days: int = 20) -> list[DayLine]:
    """Run the world untouched (the body only waits) and record one line per day."""
    from world.laws import welded_groups

    world = make_world()
    state = world.state
    lines = []
    for day in range(1, days + 1):
        born = burned = starved = pops = 0
        for _ in range(state.tempo.day_length):
            before = set(np.unique(state.layers["cover"]["id"])) - {0}
            result = world.step(WAIT)
            after = set(np.unique(state.layers["cover"]["id"])) - {0}
            popped = set(result.popped)
            died = before - after
            pops += len(popped)
            born += len(after - before)
            burned += len(died & popped)
            starved += len(died - popped)
        groups = welded_groups(state)
        lines.append(DayLine(
            day=day,
            moss=int(state.occupied("cover").sum()),
            born=born, burned=burned, starved=starved, pops=pops,
            welds=int(sum(w.sum() for w in state.welds)),
            biggest_group=max((len(g) for g in groups), default=0),
            body_energy=state.entity_by_id(world.body.entity_id).energy,
            total_energy=state.total_energy(),
        ))
    return lines


def health_report(lines: list[DayLine]) -> str:
    out = [f"{'day':>4}{'moss':>6}{'born':>6}{'burned':>8}{'starved':>9}{'pops':>6}{'welds':>7}{'group':>7}{'body':>7}{'energy':>9}"]
    for l in lines:
        out.append(
            f"{l.day:>4}{l.moss:>6}{l.born:>6}{l.burned:>8}{l.starved:>9}{l.pops:>6}{l.welds:>7}"
            f"{l.biggest_group:>7}{l.body_energy:>7.2f}{l.total_energy:>9.0f}"
        )
    fire_days = sum(l.burned > 0 for l in lines)
    out.append(f"\nfires on {fire_days} of {len(lines)} days; moss ranged {min(l.moss for l in lines)} to "
               f"{max(l.moss for l in lines)}; welds at the end {lines[-1].welds}")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure how eventful a world is.")
    parser.add_argument("world", nargs="?", default="worlds/starter.npz", help="a saved world (.npz) or a seed number")
    parser.add_argument("--days", type=float, default=3)
    parser.add_argument("--rung", type=int, default=None, help="run with this rung's laws instead of the saved ones")
    parser.add_argument("--every", type=int, default=10, help="ticks between branch pairs")
    parser.add_argument("--horizon", type=int, default=10, help="ticks each branch runs")
    parser.add_argument("--health", type=int, default=0, metavar="DAYS",
                        help="instead of the card: run this many untouched days and print one line per day")
    parser.add_argument("--agent-seed", type=int, default=0, help="seed for the random agent")
    parser.add_argument("--json", type=Path, default=None, help="also save the card here")
    args = parser.parse_args()

    def make_world() -> World:
        if args.world.isdigit():
            return World(seed=int(args.world), rung=args.rung or 4, check_every=0)
        world = World.load(args.world, check_every=0)
        if args.rung:
            world.laws_on = RUNGS[args.rung]
        return world

    if args.health:
        print(health_report(health(make_world, args.health)))
        return
    card = measure(make_world, days=args.days, every=args.every, horizon=args.horizon,
                   agent_seed=args.agent_seed, source=args.world)
    print(card.report())
    if args.json:
        args.json.write_text(card.to_json())


if __name__ == "__main__":
    main()
