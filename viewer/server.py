"""The bridge between the Python world and the browser.

Runs one World on a clock and talks to browsers over a WebSocket at /ws.

Server -> browser:
    init     once per connection (and after a reset): world size, kinds, seed, day length
    tick     every tick: every entity (id, kind, x, y, z, warmth, tone), the body, sun, last feedback
    inspect  the full values of one entity, when asked
Browser -> server:
    action   {verb, direction}: the human's next action
    facing   {direction}: where the human is facing, so the hands' target can be highlighted
    control  {cmd, value}: play / pause / step / speed / reset / driver (human or random)
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

    # --- the clock ---

    def tick(self) -> None:
        """Advance the world one tick. If anything goes wrong (a failed check, a bug), the world
        pauses and the error is logged and shown, instead of the server crashing."""
        agent = self.human if self.driver == "human" else self.random
        started = time.perf_counter()
        try:
            result = self.world.step(agent.act(self.last_feedback))
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
        elif cmd == "reset":
            self.world.reset(int(value) if value is not None else self.world.seed)
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
