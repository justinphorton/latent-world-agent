// Keyboard -> actions. Each WASD key is one of the four world directions, chosen relative to the
// camera: W is whichever grid direction points most nearly "up the screen". One press, one step.

import { emit, send, store, toggleAgentView } from "./net";

// Updated every frame by the scene: the camera's forward direction on the ground (three.js x, z).
export const cameraGround = { x: -1, z: -1 };

// The camera's forward direction snapped to the nearest grid axis.
function snappedForward() {
  const { x, z } = cameraGround;
  return Math.abs(x) > Math.abs(z) ? { x: Math.sign(x), z: 0 } : { x: 0, z: Math.sign(z) };
}

// Which world direction a WASD key means right now.
function keyDirection(key: string): string | null {
  const f = snappedForward();
  const turns: Record<string, { x: number; z: number }> = {
    w: f,
    s: { x: -f.x, z: -f.z },
    d: { x: -f.z, z: f.x },
    a: { x: f.z, z: -f.x },
  };
  const v = turns[key];
  if (!v) return null;
  // three.js +x is east, three.js -z is north.
  if (v.x > 0) return "e";
  if (v.x < 0) return "w";
  return v.z < 0 ? "n" : "s";
}

function act(verb: string, direction?: string) {
  send({ type: "action", verb, direction });
}

function facing() {
  return store.tick?.body.facing ?? "n";
}

// Space is both "wait" (a tap) and "push or pull instead of step" (held while pressing WASD):
// toward anything else it pushes that way; away from where you're facing, it pulls the block
// you're facing (you step back and it comes with you). The wait is sent when Space is released,
// and only if it wasn't used. Shift+WASD turns on the spot.
const OPPOSITE: Record<string, string> = { n: "s", s: "n", e: "w", w: "e" };
let spaceHeld = false;
let spaceUsed = false;

export function installKeys() {
  const onDown = (event: KeyboardEvent) => {
    if (event.target instanceof HTMLInputElement || event.repeat) return;
    const key = event.key.toLowerCase();
    const direction = keyDirection(key);
    if (direction) {
      if (event.shiftKey) send({ type: "facing", direction }); // turn on the spot
      else if (spaceHeld) {
        if (direction === OPPOSITE[facing()]) act("pull", facing());
        else act("push", direction);
        spaceUsed = true;
      } else act("move", direction);
    } else if (key === " ") {
      spaceHeld = true;
      spaceUsed = false;
    } else if (key === "e") act("pick_up", facing());
    else if (key === "r") act("put_down", facing());
    else if (key === "f") act("hit", facing());
    else if (key === "c") act("pick_up", "here");
    else if (key === "v") act("put_down", "here");
    else if (key === "p") send({ type: "control", cmd: store.tick?.running ? "pause" : "play" });
    else if (key === "n") send({ type: "control", cmd: "step" });
    else if (key === "h") {
      store.heatView = !store.heatView;
      emit();
    } else if (key === "g") toggleAgentView();
    else return;
    event.preventDefault();
  };
  const onUp = (event: KeyboardEvent) => {
    if (event.key !== " ") return;
    if (spaceHeld && !spaceUsed) act("wait");
    spaceHeld = false;
  };
  const onBlur = () => {
    spaceHeld = false;
  };
  window.addEventListener("keydown", onDown);
  window.addEventListener("keyup", onUp);
  window.addEventListener("blur", onBlur);
  return () => {
    window.removeEventListener("keydown", onDown);
    window.removeEventListener("keyup", onUp);
    window.removeEventListener("blur", onBlur);
  };
}
