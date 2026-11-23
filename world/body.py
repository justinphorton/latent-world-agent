"""How a body acts in the world.

The body is an ordinary thing in the creature layer. This file adds what only a body does:
seven verbs, each with one clear job, aimed in one of 4 directions (or "here", its own spot):

    move      moves only the body: walk, climb one step up, drop down any distance
    push      moves blocks sideways: shoves the whole row in front one step, and follows
    pull      the mirror of push: steps back, and the block in front comes along
    pick_up   moves one block (or moss) up onto the head
    put_down  moves what's on the head down: in front, or under the body (pillaring up)
    hit       strikes what's in front: a jolt of energy goes into it (a hard one breaks welds)
    wait      does nothing (the world keeps going)

Push/pull and pick up/put down mirror each other: push and pull are horizontal, for heavy loads,
whole rows and welded pieces, and cheap; pick up is vertical, one loose block at a time, with a
lower weight limit, and costly. Whatever rests on what's pushed or pulled rides along, and blocks
welded to it come too (unless the weld is soft, then it pulls apart). The same 27 actions exist
on every rung: the body never depends on which laws are switched on.

The body is part of the energy economy. Effort is paid from its own energy into whatever it acted
on (a hit warms the target, a push warms the pushed block, a step warms the ground), so acting
never creates or destroys energy. Exhausted, it can still wait, put things down, and walk on level
ground or downhill; anything effortful fails until the sun recharges it. It never dies.

Every action returns Feedback. The agent feels only three numbers (Feedback.sensed): strain (how
hard that was), warmth (of whatever it touched), and its own energy. Everything else it learns by
seeing what happened. The rest of Feedback is for the human's debug panel only.

Holding is physical: a lifted thing sits in the spot above the body's head, resting on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from world.laws import absorb
from world.state import LAYERS, Entity, Pos, WorldState

LIFT_STRENGTH = 1.5  # heaviest thing the body can pick up (or walk around carrying)
PUSH_STRENGTH = 3.0  # heaviest row the body can push; sliding is easier than lifting

# What effort costs, in energy. All of it goes into the thing acted on, never out of the world.
STEP_COST = 0.01  # one step on level ground or down, into the ground underfoot
CLIMB_COST = 0.03  # extra for stepping up
CARRY_COST = 0.03  # extra per unit of mass carried on the head, per step
PUSH_COST = 0.05  # per unit of mass pushed, into the block pushed directly
LIFT_COST = 0.05  # per unit of mass picked up, into what was picked up
PLACE_COST = 0.02  # per unit of mass, to set something down carefully, into it
HIT_ENERGY = 1.5  # per unit of striking mass, out of the body and into the target (bare hands: ~one glow level on soil)
WELD_STRENGTH = 1.5  # a hit with at least this much jolt breaks the target's welds (bare hands: 1.05; a held stone: ~1.65)

# Four directions, matching how the physics works: energy flows across faces, never corners.
DIRECTIONS: dict[str, tuple[int, int]] = {"n": (0, 1), "e": (1, 0), "s": (0, -1), "w": (-1, 0)}
HERE = "here"  # pick_up / put_down only: the body's own column
AIMS: dict[str, tuple[str | None, ...]] = {
    "move": tuple(DIRECTIONS),
    "push": tuple(DIRECTIONS),
    "pull": tuple(DIRECTIONS),
    "pick_up": (*DIRECTIONS, HERE),
    "put_down": (*DIRECTIONS, HERE),
    "hit": tuple(DIRECTIONS),
    "wait": (None,),
}
VERBS = tuple(AIMS)
BLOCKING = tuple(name for name, layer in LAYERS.items() if layer.blocks_movement)


@dataclass(frozen=True)
class Action:
    """One verb, aimed in a direction. The same for a human player and any agent."""

    verb: str
    direction: str | None = None


ACTIONS: tuple[Action, ...] = tuple(Action(verb, aim) for verb, aims in AIMS.items() for aim in aims)  # all 27


@dataclass
class Feedback:
    """What happened in one action.

    What the body feels (all the agent ever gets, via sensed()):
      strain:  0 = effortless, 1 = at or beyond its limit (heft, resistance, recoil, a wall)
      warmth:  warmth of whatever the action touched (hands, or the ground underfoot), or None
      energy:  the body's own energy after the action
    For the human's debug panel only:
      success, moved, note (a short reason), spent (energy paid), spent_into (what received it)
    """

    action: Action
    strain: float = 0.0
    warmth: float | None = None
    energy: float = 0.0
    success: bool = False
    moved: bool = False
    note: str = ""
    spent: float = 0.0
    spent_into: list[str] = field(default_factory=list)

    def sensed(self) -> tuple[float, float | None, float]:
        """The only part an agent may see: (strain, warmth, energy)."""
        return (self.strain, self.warmth, self.energy)


@dataclass
class Body:
    """A body: which thing it is. Everything else (where it is, what's on its head, how much
    energy it has) is read from the world state on demand, so it can never fall out of sync."""

    state: WorldState
    entity_id: int
    receivers: list[str] = field(default_factory=list)  # filled by _spend during one action

    # --- where things are ---

    @property
    def entity(self) -> Entity:
        """The body's own thing (for its energy, warmth, mass)."""
        entity = self.state.entity_by_id(self.entity_id)
        assert entity is not None
        return entity

    @property
    def pos(self) -> Pos:
        """The spot the body stands in."""
        return self.state.locate(self.entity_id)[1]

    def carried(self) -> list[tuple[str, Pos]]:
        """Everything stacked on the body's head, bottom to top, as (layer, pos).

        Usually one block or one moss. More only if something fell onto what it carries.
        """
        return self._stack_on(self.pos)

    def _stack_on(self, pos: Pos) -> list[tuple[str, Pos]]:
        """Everything resting on top of whatever is at pos, bottom to top, as (layer, pos):
        a column of blocks, possibly with moss lying on the top one."""
        x, y, z = pos
        stack = []
        for k in range(z + 1, self.state.shape[2]):
            if self.state.entity_at("block", (x, y, k)):
                stack.append(("block", (x, y, k)))
            elif self.state.entity_at("cover", (x, y, k)):
                stack.append(("cover", (x, y, k)))
                break
            else:
                break
        return stack

    def load(self) -> float:
        """Total mass resting on the head."""
        return sum(entity.mass for layer, pos in self.carried() if (entity := self.state.entity_at(layer, pos)))

    def hand_target(self, direction: str) -> Entity | None:
        """What the hands reach when aimed this way, or None.

        Aimed at a neighbouring column: whatever is there at body height (a wall), otherwise
        whatever lies on top of that column one level down (moss before the block under it).
        Aimed "here": the moss in the body's own spot, otherwise the block under its feet.
        """
        x, y, z = self.pos
        if direction == HERE:
            return self.state.entity_at("cover", (x, y, z)) or self.state.entity_at("block", (x, y, z - 1))
        dx, dy = DIRECTIONS[direction]
        tx, ty = x + dx, y + dy
        for layer in BLOCKING:
            if entity := self.state.entity_at(layer, (tx, ty, z)):
                return entity
        return self.state.entity_at("cover", (tx, ty, z)) or self.state.entity_at("block", (tx, ty, z - 1))

    # --- acting ---

    def act(self, action: Action) -> Feedback:
        """Carry out one action and report what happened.

        Things it leaves unsupported (a block lifted out from under others, a body that dug out
        its own floor) are not dropped here: gravity runs right after actions in each tick.
        """
        self.receivers = []
        feedback = Feedback(action)
        if action.verb not in AIMS or action.direction not in AIMS[action.verb]:
            feedback.note = f"can't {action.verb} toward {action.direction!r}"
        else:
            handler = {
                "move": self._move,
                "push": self._push,
                "pull": self._pull,
                "pick_up": self._pick_up,
                "put_down": self._put_down,
                "hit": self._hit,
                "wait": self._wait,
            }[action.verb]
            handler(action.direction, feedback)
        feedback.energy = self.entity.energy
        feedback.spent_into = self.receivers
        return feedback

    def _wait(self, direction, fb: Feedback) -> None:
        fb.success = True

    def _move(self, direction: str, fb: Feedback) -> None:
        """Step to a neighbouring column: walk across (dropping down to the ground there), or climb
        one step up. Never moves anything else. Strain is the weight being carried."""
        x, y, z = self.pos
        dx, dy = DIRECTIONS[direction]
        tx, ty = x + dx, y + dy
        load = self.load()
        fb.strain = min(1.0, load / LIFT_STRENGTH)

        if not self._is_ground(tx, ty):
            fb.note = "nothing to stand on there"
            return
        if load > LIFT_STRENGTH:
            fb.strain = 1.0
            fb.note = "pinned: too much weight on my head"
            return
        if not self._blocked((tx, ty, z)):
            land = z
            while land > 0 and not self._standable((tx, ty, land - 1)):
                land -= 1
            self._relocate((tx, ty, land), fb, climbing=False)
        elif not self._blocked((tx, ty, z + 1)):
            self._relocate((tx, ty, z + 1), fb, climbing=True)
        else:
            fb.strain = 1.0
            fb.note = "blocked"

    def _push(self, direction: str, fb: Feedback) -> None:
        """Shove the block in front one spot along the ground, and step after it.

        Everything that has to come along moves with it as one piece (see _gather): the rest of
        the row, whatever rests on top, and blocks welded to any of it. All of it counts toward
        PUSH_STRENGTH. It only slides along the ground: never up a step, never off the island.
        Anything pushed over a drop falls (gravity runs after actions); onto moss, it buries it.
        """
        x, y, z = self.pos
        dx, dy = DIRECTIONS[direction]
        front = self.state.entity_at("block", (x + dx, y + dy, z))
        if front is None:
            fb.note = "nothing to push"
            return
        fb.warmth = front.warmth
        offset = (dx, dy, 0)
        things, mass = self._gather(front, offset, shove=True)
        if not self._can_shift(things, mass, offset, fb):
            return
        new_spot = (x + dx, y + dy, z)
        if not all(self._free_after(things, offset, layer, (x + dx, y + dy, p[2])) for layer, p in self.carried()):
            fb.note = "no headroom for what I'm carrying"
            return
        cost = PUSH_COST * mass
        if not self._can_afford(cost + self._step_cost(climbing=False)):
            fb.note = "too tired to push it"
            return

        self._spend(cost, into=front)  # the friction heat goes into the block pushed directly
        self._shift(things, offset)
        self._relocate(new_spot, fb, climbing=False)
        fb.spent += cost
        fb.note = "pushed it"

    def _pull(self, direction: str, fb: Feedback) -> None:
        """The mirror of push: step back one spot, and the block in front comes along into the
        spot just left, with whatever rests on it and whatever is welded to it (but not the rest
        of a row: only what's attached). Stepping back has to be a plain step on level ground."""
        x, y, z = self.pos
        dx, dy = DIRECTIONS[direction]
        front = self.state.entity_at("block", (x + dx, y + dy, z))
        if front is None:
            fb.note = "nothing to pull"
            return
        fb.warmth = front.warmth
        back = (x - dx, y - dy, z)
        if not self._is_ground(back[0], back[1]) or self._blocked(back) or not self._standable((back[0], back[1], z - 1)):
            fb.note = "no room to step back"
            return
        if not all(self._room_for(layer, (back[0], back[1], p[2])) for layer, p in self.carried()):
            fb.note = "no headroom behind me for what I'm carrying"
            return
        offset = (-dx, -dy, 0)
        things, mass = self._gather(front, offset, shove=False)
        if not self._can_shift(things, mass, offset, fb, vacating=self.pos):
            return
        cost = PUSH_COST * mass
        if not self._can_afford(cost + self._step_cost(climbing=False)):
            fb.note = "too tired to pull it"
            return

        self._relocate(back, fb, climbing=False)  # step back first: the pulled block takes my spot
        self._spend(cost, into=front)
        self._shift(things, offset)
        fb.spent += cost
        fb.note = "pulled it"

    def _gather(self, start: Entity, offset: Pos, shove: bool) -> tuple[list[tuple[str, Pos]], float]:
        """Everything that has to move with `start` when it slides by `offset`, and its mass:
          - whatever rests on top of a moving block rides along (blocks, and moss lying on it)
          - blocks welded to a moving block come too, if the weld is set (both sides hard); a
            weld to a soft block is pliable and just pulls apart
          - when shoving, any block in the way of a moving block is shoved too
        Stops gathering as soon as it's heavier than PUSH_STRENGTH (it won't budge anyway)."""
        state = self.state
        soft = state.layers["block"]["soft"]
        things: dict[tuple[str, Pos], None] = {}
        mass = 0.0
        frontier = [("block", start.pos)]
        while frontier and mass <= PUSH_STRENGTH:
            layer, pos = frontier.pop()
            entity = state.entity_at(layer, pos)
            if (layer, pos) in things or entity is None:
                continue
            things[(layer, pos)] = None
            mass += entity.mass
            if layer != "block":
                continue
            above = (pos[0], pos[1], pos[2] + 1)
            if state.in_bounds(above):
                for rider in ("block", "cover"):
                    if state.entity_at(rider, above) is not None:
                        frontier.append((rider, above))
            for other in state.welded_to(pos):
                if not (soft[pos] or soft[other]):
                    frontier.append(("block", other))
            ahead = (pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2])
            if shove and state.in_bounds(ahead) and state.entity_at("block", ahead) is not None:
                frontier.append(("block", ahead))
        return list(things), mass

    def _can_shift(self, things, mass: float, offset: Pos, fb: Feedback, vacating: Pos | None = None) -> bool:
        """Whether everything gathered can slide by offset: light enough, staying on the island,
        and with room to go (moss in the way gets buried; the body's spot counts as free if it is
        stepping out of it). Sets strain and a note."""
        fb.strain = min(1.0, mass / PUSH_STRENGTH)
        if mass > PUSH_STRENGTH:
            fb.note = "it won't budge"
            return False
        for layer, pos in things:
            dst = (pos[0] + offset[0], pos[1] + offset[1], pos[2] + offset[2])
            if not self.state.in_bounds(dst) or (layer == "block" and not self._is_ground(dst[0], dst[1])):
                fb.strain, fb.note = 1.0, "no room to move it there"
                return False
            for other in BLOCKING:
                if dst == vacating and other == "creature":
                    continue
                if self.state.entity_at(other, dst) is not None and (other, dst) not in things:
                    fb.strain, fb.note = 1.0, "no room to move it there"
                    return False
            if layer == "cover" and self.state.entity_at("cover", dst) is not None and ("cover", dst) not in things:
                fb.strain, fb.note = 1.0, "no room to move it there"
                return False
        return True

    def _free_after(self, things, offset: Pos, layer: str, pos: Pos) -> bool:
        """Whether a thing of `layer` would have room at pos once `things` have slid by offset."""
        moving = {p for _, p in things}
        arriving = {(p[0] + offset[0], p[1] + offset[1], p[2] + offset[2]) for _, p in things}
        if pos in arriving:
            return False
        if pos in moving:
            return True
        return self._room_for(layer, pos)

    def _shift(self, things, offset: Pos) -> None:
        """Slide everything gathered by offset as one piece; moss in the way of a block is buried."""
        state = self.state
        moving = set(things)
        for layer, (x, y, z) in things:
            dst = (x + offset[0], y + offset[1], z + offset[2])
            moss = state.entity_at("cover", dst)
            if layer == "block" and moss is not None and ("cover", dst) not in moving:
                state.remove(moss.id, to=int(state.layers["block"]["id"][x, y, z]))
        state.move_together(things, offset)

    def _relocate(self, target: Pos, fb: Feedback, climbing: bool) -> None:
        """Move the body (with everything on its head) to target, if there is room and it has the
        energy. The cost of the step goes into the ground it lands on.

        Exhausted, it can still walk on level ground or downhill for free; climbing or carrying
        anything needs energy.
        """
        x, y, z = self.pos
        tx, ty, tz = target
        shift = tz - z
        carried = self.carried()
        # Everything on the head moves with the body; climbing also passes through the spot just
        # above the current stack, so that must be clear too.
        destinations = [(layer, (tx, ty, p[2] + shift)) for layer, p in carried]
        if not all(self._room_for(layer, dest) for layer, dest in destinations):
            fb.note = "no headroom for what I'm carrying"
            return
        if climbing and carried and self._blocked((x, y, z + len(carried) + 1)):
            fb.note = "no headroom to climb with this"
            return

        cost = self._step_cost(climbing)
        if not self._can_afford(cost):
            if climbing or carried:
                fb.note = "too tired"
                return
            cost = 0.0  # exhausted, but plain walking still works

        self.state.move("creature", (x, y, z), target)
        for (layer, src), (_, dest) in zip(reversed(carried), reversed(destinations)):
            self.state.move(layer, src, dest)
        ground = self._ground_under(target)
        if ground is not None:
            fb.warmth = ground.warmth  # the feet feel the ground
            self._spend(cost, into=ground)
            fb.spent += cost
        fb.success = fb.moved = True
        fb.note = fb.note or ("climbed" if climbing else "")

    def _pick_up(self, direction: str, fb: Feedback) -> None:
        """Lift what the hands reach onto the head, if it's light enough and there's room.

        Strain is its weight against LIFT_STRENGTH. Anything that was resting on it falls into
        the gap (gravity). Picking up "here" takes the moss underfoot, or digs out the block under
        the feet (the body then drops into the hole).
        """
        if self.carried():
            fb.note = "hands are full"
            return
        target = self.hand_target(direction)
        if target is None:
            fb.note = "nothing in reach"
            return
        fb.warmth = target.warmth
        if target.layer == "creature":
            fb.note = "can't pick that up"  # creatures never stand on creatures, heads included
            return
        if target.layer == "block" and self.state.has_welds(target.pos):
            fb.strain = 1.0
            fb.note = "it's stuck to something"  # welded pieces can be pushed or pulled, not lifted
            return
        fb.strain = min(1.0, target.mass / LIFT_STRENGTH)
        if target.mass > LIFT_STRENGTH:
            fb.note = "too heavy"
            return
        x, y, z = self.pos
        head = (x, y, z + 1)
        if not self._room_for(target.layer, head):
            fb.note = "no room above my head"
            return
        cost = LIFT_COST * target.mass
        if not self._can_afford(cost):
            fb.note = "too tired to lift it"
            return
        self._spend(cost, into=target)
        fb.spent = cost
        self.state.move(target.layer, target.pos, head)
        fb.success = True
        fb.note = "picked it up"

    def _put_down(self, direction: str, fb: Feedback) -> None:
        """Put down the bottom of what's on the head.

        Aimed at a neighbouring column, it goes at body height there (or on top of that column if
        it's one step high) and then falls to rest; a block put on moss buries it. Aimed "here",
        a block goes under the body's feet and the body rises onto it (pillaring up); moss is
        laid in the body's own spot. Putting things down never needs strength, and only costs a
        little energy when the body has some to spare (an exhausted body just lets it drop).
        """
        carried = self.carried()
        if not carried:
            fb.note = "nothing to put down"
            return
        layer, held_pos = carried[0]
        held = self.state.entity_at(layer, held_pos)
        assert held is not None
        fb.warmth = held.warmth
        x, y, z = self.pos

        if direction == HERE:
            underfoot = self.state.entity_at("cover", (x, y, z))
            if layer == "cover":
                if underfoot is not None:
                    fb.note = "there's already moss here"
                    return
                self._pay_to_place(held, fb)
                self.state.move(layer, held_pos, (x, y, z))
                fb.success = True
                fb.note = "laid it at my feet"
                return
            if underfoot is not None:
                absorb(self.state, "cover", (x, y, z), into=held)  # a block underfoot buries the moss
            self._pay_to_place(held, fb)
            self.state.move("creature", (x, y, z), (x, y, z + 1))  # rise into the spot the block leaves
            self.state.move(layer, held_pos, (x, y, z))
            fb.success = fb.moved = True
            fb.note = "put it under myself"
            return

        dx, dy = DIRECTIONS[direction]
        tx, ty = x + dx, y + dy
        if not self._is_ground(tx, ty):
            fb.note = "nothing to put it on there"
            return
        for spot in [(tx, ty, z), (tx, ty, z + 1)]:
            if self._room_for(layer, spot, burying=True):
                if layer == "block" and self.state.entity_at("cover", spot) is not None:
                    absorb(self.state, "cover", spot, into=held)  # put on moss: buries it
                self._pay_to_place(held, fb)
                self.state.move(layer, held_pos, spot)
                fb.success = True
                fb.note = "put it down"
                return
        fb.note = "no room to put it there"

    def _hit(self, direction: str, fb: Feedback) -> None:
        """Strike what the hands reach, with whatever is on the head (or the bare body).

        The striker's mass sets the force: HIT_ENERGY per unit of mass moves out of the body and
        into the target, so heavy tools hit harder and tire the body faster. Heavy targets kick
        back (strain). A jolt of at least WELD_STRENGTH (more than bare hands give) breaks the
        target's welds: welds take heat to make and one sharp blow to break.
        """
        target = self.hand_target(direction)
        if target is None:
            fb.note = "nothing in reach"
            return
        carried = self.carried()
        tool = self.state.entity_at(*carried[0]) if carried else None
        striker = tool if tool is not None else self.entity
        jolt = HIT_ENERGY * striker.mass
        if not self._can_afford(jolt):
            fb.note = "too tired to hit"
            return
        self._spend(jolt, into=target)
        fb.spent = jolt
        fb.success = True
        fb.strain = min(1.0, target.mass / PUSH_STRENGTH)  # recoil: heavy things push back
        fb.warmth = target.warmth
        fb.note = f"hit it with {'the ' + tool.name if tool else 'my body'}"
        if target.layer == "block" and jolt >= WELD_STRENGTH and self.state.has_welds(target.pos):
            self.state.break_welds(target.pos)
            fb.note += ", and it broke loose"

    # --- energy ---

    def _step_cost(self, climbing: bool) -> float:
        """What one step costs right now, given what's on the head."""
        return STEP_COST + (CLIMB_COST if climbing else 0.0) + CARRY_COST * self.load()

    def _pay_to_place(self, held: Entity, fb: Feedback) -> None:
        """Setting something down takes a little care; an exhausted body just lets it drop for free."""
        cost = PLACE_COST * held.mass
        if self._can_afford(cost):
            self._spend(cost, into=held)
            fb.spent = cost

    def _can_afford(self, cost: float) -> bool:
        return self.entity.energy >= cost

    def _spend(self, cost: float, into: Entity) -> None:
        """Move `cost` energy from the body into `into`. Nothing is created or lost."""
        if cost <= 0:
            return
        self.state.transfer(self.entity_id, into.id, cost)
        self.receivers.append(into.name)

    # --- small questions about the world around the body ---

    def _blocked(self, pos: Pos) -> bool:
        """Whether a creature can't be at pos: outside the world, or something solid is there."""
        return not self.state.in_bounds(pos) or any(self.state.entity_at(l, pos) for l in BLOCKING)

    def _standable(self, pos: Pos) -> bool:
        """Whether something at pos can hold a creature up."""
        return any(self.state.entity_at(l, pos) for l in LAYERS["creature"].supported_by)

    def _ground_under(self, pos: Pos) -> Entity | None:
        """Whatever the body would be standing on at pos."""
        x, y, z = pos
        for layer in LAYERS["creature"].supported_by:
            if entity := self.state.entity_at(layer, (x, y, z - 1)):
                return entity
        return None

    def _is_ground(self, x: int, y: int) -> bool:
        """Whether (x, y) is part of the island at all (it has a floor). The void around it is not."""
        return self.state.in_bounds((x, y, 0)) and self.state.entity_at("block", (x, y, 0)) is not None

    def _room_for(self, layer: str, pos: Pos, burying: bool = False) -> bool:
        """Whether a thing of this layer can be put at pos. With burying=True a block may also go
        where moss lies (burying it)."""
        if burying and layer == "block":
            return self.state.in_bounds(pos) and not any(
                self.state.entity_at(l, pos) for l in ("block", "creature")
            )
        return self.state.has_room(layer, pos)
