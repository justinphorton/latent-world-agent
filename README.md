# latent world

A small block world for studying agents that learn how a world works by acting in it.

The world has hidden properties (how heavy a block is, how well it holds heat, when it will
pop) and simple laws that act on them. An agent never reads those numbers. It only sees surfaces
and glow, and feels strain and warmth when it acts, so anything it learns about the world it has
to find out by poking at it. Because the world is fully simulated, everything an agent believes
can be checked against the truth.

The world is done enough to build agents on; the agents themselves are the next step.

## The world

A floating island of blocks (32 × 32 × 8), different for every seed. A frozen starter island
lives in `world/worlds/starter.npz`.

- **Materials:** soil, stone, sponge, crystal, moss and bedrock. Each block rolls its own
  values (mass, heat capacity, conductivity...) near its material's usual ones, so two stones
  are never quite the same.
- **One kind of energy, conserved.** Every block has some; warmth is energy over capacity.
  Energy only enters from the sun and leaves by radiating to the sky, and the books always
  balance.
- **Laws** (each can be switched on or off):
  - gravity: unsupported things fall
  - sun: a day and night cycle warms the tops of columns
  - flow: heat moves from warm to cool between touching things
  - pop: crystals and moss fill an inner store and pop when warm enough, heating their neighbours
  - life: moss spreads downhill when well fed, starves when cold, and burns when it pops
  - weld: hot stone goes soft and welds to whatever it touches
  - radiation: exposed faces lose heat
- **Rungs:** suggested sets of laws, from base physics (1) up to welding (4).
- On its own, the island keeps changing: crystals pop on hot afternoons and set moss patches on
  fire, and moss regrows somewhere else.
- Deterministic and seeded: the same seed and actions always give exactly the same world, and any
  moment can be snapshotted and restored.

## The agent's side

An agent controls one body. Everything it gets is listed here.

**Actions** (`world.body.ACTIONS`, 27 in all): `move`, `push`, `pull` and `hit` in four
directions (n/e/s/w), `pick_up` and `put_down` in four directions or `here`, and `wait`. Effort
costs the body energy, which goes into whatever it acts on; the body recharges in sunlight.

**Touch**, after every action: `(strain, warmth, energy)`, meaning how hard that was (0 to 1),
how warm the thing it touched was, and its own energy.

**Sight**, from `world.senses.observe()`:
- a window around the body (15 × 15 columns by default), surfaces only: buried blocks are hidden
- for each visible thing: its position, which material it looks like (a number, not a name), a
  fixed random tone (a fingerprint for recognizing it again) and its glow (warmth, in 10 levels)
- the time of day, its own position and energy

Every perception choice is a dial in `SenseSettings` (window size, whole island, glow and tone
levels, whether true ids are given), and `senses.EASY` turns them all to the easy side. The same
observation comes as a grid of numbers (`to_grid`, for neural networks) or a list of things
(`to_list`, for readable or rule-based agents).

A minimal agent loop:

```python
from world.world import World
from world.body import ACTIONS
from world.senses import observe, to_grid
import random

world = World.load("world/worlds/starter.npz")
seen = observe(world.state, world.body.entity_id, world.tick)
for _ in range(1000):
    action = random.choice(ACTIONS)          # your agent decides here, from `seen`
    result = world.step(action)
    seen = observe(world.state, world.body.entity_id, world.tick, feedback=result.feedback)
    grid = to_grid(seen)                     # (channels, x, y, z) numpy array
```

## Running it

Python 3.12+ and Node 18+.

```
python -m venv .venv
.venv\Scripts\activate            # or: source .venv/bin/activate
pip install -e ".[dev]"
cd viewer/app && npm install
```

Then, in two terminals:

```
python -m uvicorn viewer.server:app --port 8000
cd viewer/app && npm run dev       # opens on http://localhost:5173
```

In the viewer you can play the body yourself or let a random agent drive, switch between god
view and what the agent perceives, inspect any block's true values, and scrub back through
replay. The controls are behind the **controls** button at the bottom.

## Tools

- `pytest -m "not slow"`: the quick tests (`pytest` runs everything)
- `python -m world.measure`: how eventful a world is: per action, how often it changes something
  the agent can see, and which kinds of event random play ever causes
- `python -m world.measure --health 40`: run the island untouched and print one line per day
  (moss, fires, pops, welds, energy)

## License

MIT
