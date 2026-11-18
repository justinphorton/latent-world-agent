// The connection to the Python server, and the latest world data it sent.
// The 3D scene reads `store` directly every frame; React UI re-renders through useStore().

import { useSyncExternalStore } from "react";

export type Kind = { name: string; layer: string; color: string } | null;

export type Init = {
  shape: [number, number, number];
  kinds: Kind[];
  seed: number;
  dayLength: number;
  daylight: number; // fraction of each day the sun is up (the rest is night)
  senses: { toneLevels: number | null; glowLevels: number | null }; // how precisely the agent sees
  directions: string[];
};

export type Feedback = {
  action: { verb: string; direction: string | null };
  success: boolean;
  moved: boolean;
  strain: number;
  warmth: number | null;
  energy: number;
  spent: number;
  spent_into: string[];
  note: string;
};

export type Tick = {
  tick: number;
  sun: number;
  running: boolean;
  speed: number;
  driver: "human" | "random";
  entities: number[][]; // [id, kind, x, y, z, warmth, tone]
  body: {
    id: number;
    pos: [number, number, number];
    energy: number;
    capacity: number;
    load: number;
    facing: string;
    target: [number, number, number] | null;
  };
  feedback: Feedback | null;
  fell: number[];
  timing: ServerTiming;
  error: string | null; // set when the world hit a problem and paused itself
  // What replay can reach and where we are; tryFrom is set while trying something different.
  timeline: { start: number; head: number; now: number; tryFrom: number | null; onTry: boolean };
  agentView: number[] | null; // with agent view on: the ids of everything the agent perceives
  welds: [number, number, number, number][]; // god view: each weld as [x, y, z, axis]
};

export type Inspected = Record<string, number | string | number[]> | null;

export const store = {
  connected: false,
  init: null as Init | null,
  tick: null as Tick | null,
  inspected: null as Inspected,
  selectedId: null as number | null,
  heatView: false,
  agentView: false, // dim everything the agent can't perceive
  heatRange: [0, 1] as [number, number], // warmth range the heat view's colours stretch across
  perf: null as Perf | null,
  version: 0,
};

// How the viewer is performing, measured once a second (and sent to the server every few seconds).
export type Perf = {
  fps: number;
  frameMs: number; // average time between frames
  worstFrameMs: number; // slowest single frame in the last second
  drawCalls: number;
  triangles: number;
  parseMs: number; // time to decode one tick message
  messageKb: number;
  serverStepMs: number; // server: time to run one world tick
  serverBuildMs: number; // server: time to build one tick message
};

export type ServerTiming = { stepMs: number; buildMs: number };

// Message decode cost, averaged into the perf readout.
export const messageStats = { parseMs: 0, kb: 0 };

const listeners = new Set<() => void>();

// Agent view: the server only sends what the agent perceives while this is on.
export function toggleAgentView() {
  store.agentView = !store.agentView;
  send({ type: "control", cmd: "agent_view", value: store.agentView });
  emit();
}

export function emit() {
  store.version++;
  listeners.forEach((listener) => listener());
}

let socket: WebSocket | null = null;

export function connect() {
  socket = new WebSocket(`ws://${location.hostname}:8000/ws`);
  socket.onopen = () => {
    store.connected = true;
    send({ type: "control", cmd: "agent_view", value: store.agentView }); // keep the server in sync
    emit();
  };
  socket.onmessage = (event) => {
    const started = performance.now();
    const message = JSON.parse(event.data);
    if (message.type === "tick") {
      messageStats.parseMs = messageStats.parseMs * 0.8 + (performance.now() - started) * 0.2;
      messageStats.kb = event.data.length / 1024;
    }
    if (message.type === "init") store.init = message;
    else if (message.type === "tick") {
      store.tick = message;
      store.heatRange = heatRange(message);
    }
    else if (message.type === "inspect") store.inspected = message.entity;
    emit();
  };
  socket.onclose = () => {
    store.connected = false;
    emit();
    setTimeout(connect, 1000); // keep trying until the server is up
  };
}

// The 2nd to 98th percentile of current warmths (blocks and cover), widened to at least 0.1, so the
// heat view always uses its full colour range for the differences that actually exist right now.
function heatRange(tick: Tick): [number, number] {
  const values = tick.entities.filter((e) => e[0] !== tick.body.id).map((e) => e[5]);
  if (values.length === 0) return [0, 1];
  values.sort((a, b) => a - b);
  let lo = values[Math.floor(values.length * 0.02)];
  let hi = values[Math.min(values.length - 1, Math.floor(values.length * 0.98))];
  if (hi - lo < 0.1) {
    const mid = (hi + lo) / 2;
    lo = mid - 0.05;
    hi = mid + 0.05;
  }
  return [lo, hi];
}

export function send(message: object) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(message));
}

export function useStore<T>(select: (s: typeof store) => T): T {
  useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => store.version,
  );
  return select(store);
}
