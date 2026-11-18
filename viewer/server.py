"""The bridge between the Python world and the browser.

Runs one World on a clock and talks to browsers over a WebSocket at /ws.

Server -> browser:
    init     once per connection (and after a reset): world size, kinds, seed, day length
    tick     every tick: every entity (id, kind, x, y, z, warmth, tone), the body, sun, last feedback
    inspect  the full values of one entity, when asked
Browser -> server:
    action   {verb, direction}: the human's next action
    facing   {direction}: where the human is facing, so the hands' target can be highlighted
    control  {cmd, value}: play / pause / step / speed / reset / driver (human or random) /
             seek (jump exactly to a recorded tick) / peek (jump to the nearest snapshot, while scrubbing) /
             switch (flip between the original and the try) / keep_try / discard_try
    inspect  {id}: ask for one entity's full values
    perf     the browser's frame rate and render stats, appended to logs/viewer-perf.jsonl

Run with:  uvicorn viewer.server:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import traceback
from contextlib import asynccontextmanager, suppress
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from agents.agents import HumanAgent, RandomAgent
from world.body import DIRECTIONS, Action, Feedback
from world.laws import DAY_LENGTH, DAYLIGHT
from world.state import FIELDS, LAYERS
from world.world import World

BASE_TICKS_PER_SECOND = 5
MAX_UPDATES_PER_SECOND = 15  # world pictures sent to the browser, at most
ERROR_LOG = Path(__file__).resolve().parent.parent / "logs" / "viewer-errors.log"
SNAPSHOT_EVERY = 100  # ticks between replay snapshots
HISTORY_TICKS = 3 * DAY_LENGTH  # how far back replay can go


class History:
    """What happened, so it can be replayed: every action taken, plus a snapshot every
    SNAPSHOT_EVERY ticks. Worlds are deterministic, so a snapshot plus the actions after it
    reproduces any moment exactly. Keeps the last HISTORY_TICKS ticks.

    `head` is the newest tick recorded. When the world is rewound (world.tick < head), playing
    follows the recorded actions until it catches up; acting differently branches: the old
    future is dropped and recording continues from there.
    """

    def __init__(self, world: World):
        self.start = world.tick
        self.head = world.tick
        self.actions: dict[int, Action] = {}  # the action taken at each tick
        self.snapshots = {world.tick: world.snapshot()}

    def record(self, tick: int, action: Action, world: World) -> None:
        """Remember the action taken at `tick` (the world has just stepped past it)."""
        self.actions[tick] = action
        self.head = world.tick
        if world.tick % SNAPSHOT_EVERY == 0:
            self.snapshots[world.tick] = world.snapshot()
        self._forget_old()

    def seek(self, world: World, tick: int) -> None:
        """Put the world at `tick`: restore the nearest snapshot before it, replay up to it."""
        tick = max(self.start, min(self.head, tick))
        base = max(t for t in self.snapshots if t <= tick)
        world.restore(self.snapshots[base])
        for t in range(base, tick):
            world.step(self.actions[t])

    def peek(self, world: World, tick: int) -> None:
        """Put the world at the nearest snapshot at or before `tick`, instantly (no replaying).
        Used while scrubbing: a flipbook of snapshots, then seek() lands exactly on release."""
        tick = max(self.start, min(self.head, tick))
        world.restore(self.snapshots[max(t for t in self.snapshots if t <= tick)])

    def fork(self) -> History:
        """An independent copy to try something different in. Snapshots are never changed once
        taken, so the copy can share them."""
        twin = History.__new__(History)
        twin.start, twin.head = self.start, self.head
        twin.actions, twin.snapshots = dict(self.actions), dict(self.snapshots)
        return twin

    def branch(self, tick: int) -> None:
        """Forget everything recorded after `tick`: a new future starts here."""
        self.actions = {t: a for t, a in self.actions.items() if t < tick}
        self.snapshots = {t: s for t, s in self.snapshots.items() if t <= tick}
        self.head = tick

    def _forget_old(self) -> None:
        """Drop what's older than HISTORY_TICKS, always keeping a snapshot to start from."""
        keep_from = max((t for t in self.snapshots if t <= self.head - HISTORY_TICKS), default=self.start)
        if keep_from > self.start:
            self.snapshots = {t: s for t, s in self.snapshots.items() if t >= keep_from}
            self.actions = {t: a for t, a in self.actions.items() if t >= keep_from}
            self.start = keep_from


class Session:
    """The one running world, who drives it, and who is watching."""

    def __init__(self):
        self.world = World(seed=0)
        self.human = HumanAgent()
        self.random = RandomAgent(seed=0)
        self.driver = "human"
        self.facing = "n"
        self.running = True
        self.speed = 1.0
        self.last_feedback: Feedback | None = None
        self.last_events: list = []
        self.clients: set[WebSocket] = set()
        self.pressed = asyncio.Event()  # set when the human acts, so the clock can respond at once
        self.step_ms = 0.0  # smoothed time to run one world tick
        self.build_ms = 0.0  # smoothed time to build one tick message
        self.error: str | None = None  # the last thing that went wrong, shown in the browser
        self.history = History(self.world)  # the timeline being played right now
        # Trying something different: acting while rewound keeps what really happened, so you
        # can flip between the original and the try, then keep or discard the try. There is
        # always the original plus at most one try; these are only set while a try exists.
        self.original: History | None = None
        self.attempt: History | None = None
        self.try_from: int | None = None  # the tick the try split off at

    # --- the clock ---

    def tick(self) -> None:
        """Advance the world one tick. If anything goes wrong (a failed check, a bug), the world
        pauses and the error is logged and shown, instead of the server crashing.

        While rewound, the world replays what was recorded, unless the human presses something,
        which branches off a new future from here.
        """
        tick = self.world.tick
        rewound = tick < self.history.head
        replaying = rewound and not (self.driver == "human" and self.human.pending is not None)
        if replaying:
            action = self.history.actions[tick]  # what really happened here
        else:
            if rewound:
                self._start_try(tick)
            agent = self.human if self.driver == "human" else self.random
            action = agent.act(self.last_feedback)
        started = time.perf_counter()
        try:
            result = self.world.step(action)
            if not replaying:
                self.history.record(tick, action, self.world)
        except Exception as error:
            self.running = False
            self.error = f"{type(error).__name__}: {error}"
            ERROR_LOG.parent.mkdir(exist_ok=True)
            with ERROR_LOG.open("a") as f:
                f.write(f"{datetime.now().isoformat(timespec='seconds')} tick {self.world.tick}\n")
                f.write(traceback.format_exc() + "\n")
            return
        self.step_ms = _smooth(self.step_ms, (time.perf_counter() - started) * 1000)
        self.last_feedback = result.feedback
        self.last_events = result.events
        action = result.feedback.action
        if action.verb in ("move", "push") and action.direction:
            self.facing = action.direction  # you face the way you last walked or pushed

    def _start_try(self, tick: int) -> None:
        """Acting differently while rewound. Inside the try, this just changes the try's future
        from here. In the original, it starts a fresh try (replacing any earlier one) and leaves
        the original untouched."""
        if self.attempt is not None and self.history is self.attempt:
            self.attempt.branch(tick)
            return
        self.original = self.original or self.history
        self.attempt = self.original.fork()
        self.attempt.branch(tick)
        self.history = self.attempt
        self.try_from = tick

    def _show(self, timeline: History) -> None:
        """Switch to another timeline at the same moment (or as close as it reaches), paused."""
        self.history = timeline
        timeline.seek(self.world, self.world.tick)
        self.running = False
        self.last_feedback, self.last_events = None, []

    def _end_try(self, keep: bool) -> None:
        """Keep the try (it becomes the original) or discard it (back to the original)."""
        if self.attempt is None or self.original is None:
            return
        self._show(self.attempt if keep else self.original)
        self.original = self.attempt = self.try_from = None

    async def run(self) -> None:
        """Tick on a steady clock, but answer a keypress right away instead of waiting for the next
        scheduled tick. Ticks are never closer than half a period, so pressing keys can't speed up
        the world's time."""
        loop = asyncio.get_running_loop()
        last = last_sent = loop.time()
        while True:
            period = 1 / (BASE_TICKS_PER_SECOND * self.speed)
            self.pressed.clear()
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self.pressed.wait(), timeout=max(0.0, last + period - loop.time()))
            pressed = self.pressed.is_set()
            if pressed:
                await asyncio.sleep(max(0.0, last + period / 2 - loop.time()))
            if self.running:
                self.tick()
                last = loop.time()
                # At high speed the world ticks faster than the browser needs pictures: send at most
                # MAX_UPDATES_PER_SECOND (always right away after a keypress, or if the world paused).
                if pressed or not self.running or last - last_sent >= 1 / MAX_UPDATES_PER_SECOND:
                    await self.broadcast(self.tick_message())
                    last_sent = last
            else:
                last = loop.time()
                await asyncio.sleep(0.05)

    # --- messages ---

    def init_message(self) -> dict:
        recipes = self.world.state.recipes
        kinds = [None if k is None else {"name": k.name, "layer": k.layer, "color": k.color} for k in recipes]
        return {
            "type": "init",
            "shape": list(self.world.state.shape),
            "kinds": kinds,
            "seed": self.world.seed,
            "dayLength": DAY_LENGTH,
            "daylight": DAYLIGHT,
            "directions": list(DIRECTIONS),
        }

    def tick_message(self) -> dict:
        started = time.perf_counter()
        message = self._tick_message()
        self.build_ms = _smooth(self.build_ms, (time.perf_counter() - started) * 1000)
        return message

    def _tick_message(self) -> dict:
        world = self.world
        rows = []
        for layer in LAYERS:
            arrays = world.state.layers[layer]
            xs, ys, zs = np.nonzero(arrays["kind"])
            warmth = arrays["energy"][xs, ys, zs] / arrays["capacity"][xs, ys, zs]
            ids, kinds, tones = (arrays[f][xs, ys, zs] for f in ("id", "kind", "tone"))
            rows.append(np.stack([ids, kinds, xs, ys, zs, warmth, tones], axis=1))
        entities = np.concatenate(rows).round(3).tolist()

        body = world.body
        me = body.entity
        target = body.hand_target(self.facing)
        feedback = None
        if self.last_feedback is not None:
            feedback = asdict(self.last_feedback)
            feedback["action"] = asdict(self.last_feedback.action)
        return {
            "type": "tick",
            "tick": world.tick,
            "sun": round(world.sun, 3),
            "running": self.running,
            "speed": self.speed,
            "driver": self.driver,
            "entities": entities,
            "body": {
                "id": body.entity_id,
                "pos": list(body.pos),
                "energy": round(me.energy, 3),
                "capacity": me.capacity,
                "load": round(body.load(), 3),
                "facing": self.facing,
                "target": list(target.pos) if target is not None else None,
            },
            "feedback": feedback,
            "fell": [e.entity_id for e in self.last_events if hasattr(e, "end")],
            "timing": {"stepMs": round(self.step_ms, 2), "buildMs": round(self.build_ms, 2)},
            "error": self.error,
            "timeline": {
                "start": self.history.start,
                "head": self.history.head,
                "now": world.tick,
                "tryFrom": self.try_from,  # set while a try exists
                "onTry": self.attempt is not None and self.history is self.attempt,
            },
        }

    def inspect_message(self, entity_id: int) -> dict:
        entity = self.world.state.entity_by_id(entity_id)
        if entity is None:
            return {"type": "inspect", "entity": None}
        values = {field: _json_value(getattr(entity, field)) for field in FIELDS}
        values.update(name=entity.name, layer=entity.layer, pos=list(entity.pos), warmth=entity.warmth)
        return {"type": "inspect", "entity": values}

    async def broadcast(self, message: dict) -> None:
        text = json.dumps(message)
        for client in list(self.clients):
            try:
                await client.send_text(text)
            except Exception:
                self.clients.discard(client)

    # --- input ---

    async def handle(self, websocket: WebSocket, message: dict) -> None:
        kind = message.get("type")
        if kind == "action":
            self.human.press(Action(message["verb"], message.get("direction")))
            if self.driver == "human":
                self.pressed.set()
        elif kind == "facing" and message.get("direction") in DIRECTIONS:
            self.facing = message["direction"]
            await websocket.send_text(json.dumps(self.tick_message()))
        elif kind == "inspect":
            await websocket.send_text(json.dumps(self.inspect_message(int(message["id"]))))
        elif kind == "control":
            await self.control(message.get("cmd"), message.get("value"))
        elif kind == "perf":
            _log_perf(message)

    async def control(self, cmd: str | None, value) -> None:
        if cmd == "play":
            self.running, self.error = True, None
        elif cmd == "pause":
            self.running = False
        elif cmd == "step":
            self.running = False
            self.tick()
        elif cmd == "speed":
            self.speed = float(value)
        elif cmd == "driver" and value in ("human", "random"):
            self.driver = value
        elif cmd == "switch" and self.attempt is not None and self.original is not None:
            self._show(self.original if self.history is self.attempt else self.attempt)
        elif cmd == "keep_try":
            self._end_try(keep=True)
        elif cmd == "discard_try":
            self._end_try(keep=False)
        elif cmd == "peek" and value is not None:
            self.running = False
            self.history.peek(self.world, int(value))
            self.last_feedback, self.last_events = None, []
        elif cmd == "seek" and value is not None:
            # Jump to any recorded moment and pause there. Playing on replays what happened.
            self.running = False
            try:
                self.history.seek(self.world, int(value))
            except Exception as error:
                self.error = f"{type(error).__name__}: {error}"
            self.last_feedback, self.last_events = None, []
        elif cmd == "reset":
            self.world.reset(int(value) if value is not None else self.world.seed)
            self.history = History(self.world)
            self.original = self.attempt = self.try_from = None
            self.last_feedback, self.last_events, self.error = None, [], None
            await self.broadcast(self.init_message())
        await self.broadcast(self.tick_message())


def _json_value(value):
    """Browsers can't read infinity in JSON: show "never" (e.g. a block that never softens)."""
    return "never" if isinstance(value, float) and math.isinf(value) else value


def _smooth(previous: float, latest: float) -> float:
    """A running average that follows recent values without jumping around."""
    return latest if previous == 0 else previous * 0.9 + latest * 0.1


PERF_LOG = Path(__file__).resolve().parent.parent / "logs" / "viewer-perf.jsonl"


def _log_perf(report: dict) -> None:
    """Append the browser's performance report (one JSON line every few seconds)."""
    PERF_LOG.parent.mkdir(exist_ok=True)
    report = {"time": datetime.now().isoformat(timespec="seconds"), **{k: v for k, v in report.items() if k != "type"}}
    with PERF_LOG.open("a") as f:
        f.write(json.dumps(report) + "\n")


session = Session()


@asynccontextmanager
async def lifespan(app: FastAPI):
    clock = asyncio.create_task(session.run())
    yield
    clock.cancel()
    with suppress(asyncio.CancelledError):
        await clock


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    session.clients.add(websocket)
    await websocket.send_text(json.dumps(session.init_message()))
    await websocket.send_text(json.dumps(session.tick_message()))
    try:
        while True:
            await session.handle(websocket, json.loads(await websocket.receive_text()))
    except WebSocketDisconnect:
        session.clients.discard(websocket)
