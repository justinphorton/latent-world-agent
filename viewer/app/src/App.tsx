// The page: the 3D world, with a thin layer of controls and panels over it.

import { useEffect, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import * as THREE from "three";
import { Scene, turbo } from "./Scene";
import { Sky } from "./Sky";
import { connect, emit, send, store, toggleAgentView, useStore } from "./net";
import { installKeys } from "./keys";

export function App() {
  useEffect(() => {
    connect();
    return installKeys();
  }, []);

  const connected = useStore((s) => s.connected);
  const hasWorld = useStore((s) => !!s.init && !!s.tick);

  return (
    <div className="app">
      <Sky />
      <Canvas
        orthographic
        shadows="soft"
        dpr={[1, 1.5]}
        gl={{ antialias: true, alpha: true }}
        camera={{ position: [28, 26, 28], zoom: 21, near: 0.1, far: 500 }}
      >
        {hasWorld && <Scene />}
      </Canvas>
      <TopBar />
      <BodyPanel />
      <Inspector />
      <HeatLegend />
      <PerfReadout />
      <KeyHints />
      {!connected && (
        <div className="waiting">
          Waiting for the world server…
          <span>run: uvicorn viewer.server:app --port 8000</span>
        </div>
      )}
    </div>
  );
}

function TopBar() {
  const tick = useStore((s) => s.tick);
  const init = useStore((s) => s.init);
  const heat = useStore((s) => s.heatView);
  const agentView = useStore((s) => s.agentView);
  const [seed, setSeed] = useState("0");
  if (!tick || !init) return null;

  const day = Math.floor(tick.tick / init.dayLength) + 1;
  const phase = (tick.tick % init.dayLength) / init.dayLength;
  const control = (cmd: string, value?: unknown) => send({ type: "control", cmd, value });

  return (
    <div className="panel topbar">
      <span className="title">latent world</span>
      <button onClick={() => control(tick.running ? "pause" : "play")}>{tick.running ? "❚❚ pause" : "▶ play"}</button>
      <button onClick={() => control("step")}>step</button>
      <div className="group">
        {[0.5, 1, 4, 10].map((speed) => (
          <button key={speed} className={tick.speed === speed ? "on" : ""} onClick={() => control("speed", speed)}>
            {speed}×
          </button>
        ))}
      </div>
      <div className="group">
        {(["human", "random"] as const).map((driver) => (
          <button key={driver} className={tick.driver === driver ? "on" : ""} onClick={() => control("driver", driver)}>
            {driver === "human" ? "you" : "random agent"}
          </button>
        ))}
      </div>
      <button
        className={heat ? "on" : ""}
        onClick={() => {
          store.heatView = !store.heatView;
          emit();
        }}
      >
        heat view
      </button>
      <button className={agentView ? "on" : ""} onClick={toggleAgentView} title="dim everything the agent can't perceive (G)">
        agent view
      </button>
      <div className="group seed">
        <span>seed</span>
        <input value={seed} onChange={(e) => setSeed(e.target.value)} />
        <button onClick={() => control("reset", Number(seed) || 0)}>new island</button>
      </div>
      <div className="clock">
        <span>day {day}</span>
        <div className="dayline">
          <div className="daylight" style={{ width: `${init.daylight * 100}%` }} />
          <div className="sunmark" style={{ left: `${phase * 100}%` }}>
            {tick.sun > 0 ? "☀" : "☾"}
          </div>
        </div>
        <span className="muted">tick {tick.tick}</span>
      </div>
      {tick.error && (
        <span className="world-error" title="details in logs/viewer-errors.log">
          ⚠ paused: {tick.error}
        </span>
      )}
      <Timeline />
    </div>
  );
}

// Replay: drag back to any recorded moment (the last few days). Playing from there re-runs what
// really happened; acting while rewound starts a new future from that point.
function Timeline() {
  const timeline = useStore((s) => s.tick?.timeline);
  const dayLength = useStore((s) => s.init?.dayLength ?? 900);
  const [dragging, setDragging] = useState<number | null>(null);
  const lastPeek = useRef(0);
  if (!timeline) return null;

  const { start, head, now, tryFrom, onTry } = timeline;
  const shown = dragging ?? now;
  const live = now >= head;
  const when = (t: number) => `day ${Math.floor(t / dayLength) + 1} · tick ${t}`;
  const control = (cmd: string, value?: number) => send({ type: "control", cmd, value });
  const seek = (value: number) => {
    control("seek", value);
    setDragging(null);
  };

  return (
    <div className="timeline">
      <span className="muted">replay</span>
      <input
        type="range"
        min={start}
        max={Math.max(head, start + 1)}
        value={shown}
        onChange={(e) => {
          // While dragging, flip through the nearest snapshots (instant, in steps); the exact
          // moment loads when the slider is let go.
          const value = Number(e.target.value);
          setDragging(value);
          if (performance.now() - lastPeek.current > 100) {
            lastPeek.current = performance.now();
            send({ type: "control", cmd: "peek", value });
          }
        }}
        onPointerUp={(e) => seek(Number((e.target as HTMLInputElement).value))}
        onKeyUp={(e) => seek(Number((e.target as HTMLInputElement).value))}
      />
      <span className="num">{when(shown)}</span>
      {live ? (
        <span className="live">● live</span>
      ) : (
        <>
          <span className="muted">{head - now} ticks back</span>
          <button
            onClick={() => {
              seek(head);
              control("play");
            }}
          >
            back to live
          </button>
        </>
      )}
      {tryFrom !== null && (
        <span className="try">
          <span className={onTry ? "on" : ""}>{onTry ? `trying something different (from tick ${tryFrom})` : "original"}</span>
          <button onClick={() => control("switch")}>{onTry ? "show original" : "show try"}</button>
          <button onClick={() => control("keep_try")}>keep try</button>
          <button onClick={() => control("discard_try")}>discard try</button>
        </span>
      )}
    </div>
  );
}

function Bar({ value, color }: { value: number; color: string }) {
  return (
    <div className="bar">
      <div style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%`, background: color }} />
    </div>
  );
}

function BodyPanel() {
  const tick = useStore((s) => s.tick);
  if (!tick) return null;
  const body = tick.body;
  const fb = tick.feedback;
  const energy = body.energy / body.capacity;
  return (
    <div className="panel side body">
      <h3>body</h3>
      <div className="row">
        <span>energy</span>
        <Bar value={energy} color={energy < 0.05 ? "#e8572a" : "#f2c14b"} />
        <span className="num">{body.energy.toFixed(2)}</span>
      </div>
      <div className="row">
        <span>load</span>
        <Bar value={body.load} color="#a0b4d0" />
        <span className="num">{body.load.toFixed(2)}</span>
      </div>
      <div className="row muted">
        facing {body.facing}
      </div>
      {fb && (
        <>
          <h3>last action</h3>
          <div className={`action ${fb.success ? "ok" : "fail"}`}>
            {fb.action.verb}
            {fb.action.direction ? ` ${fb.action.direction}` : ""} {fb.success ? "✓" : "✗"}
          </div>
          {fb.note && <div className="note">“{fb.note}”</div>}
          <div className="row">
            <span>strain</span>
            <Bar value={fb.strain} color="#d47a9a" />
            <span className="num">{fb.strain.toFixed(2)}</span>
          </div>
          {fb.spent > 0 && (
            <div className="row">
              <span>spent</span>
              <span className="spent">
                {fb.spent.toFixed(3)} → {[...new Set(fb.spent_into)].join(", ")}
              </span>
            </div>
          )}
          {fb.warmth !== null && (
            <div className="row">
              <span>felt warmth</span>
              <Bar value={fb.warmth} color="#e8873a" />
              <span className="num">{fb.warmth.toFixed(2)}</span>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function Inspector() {
  const entity = useStore((s) => s.inspected);
  const selected = useStore((s) => s.selectedId);
  const tick = useStore((s) => s.tick?.tick);

  // Keep the inspected values live while something is selected.
  useEffect(() => {
    if (selected !== null) send({ type: "inspect", id: selected });
  }, [selected, tick]);

  if (!entity || selected === null) return null;
  const rows: [string, string][] = [
    ["kind", String(entity.name)],
    ["id", `#${entity.id}`],
    ["position", (entity.pos as number[]).join(", ")],
    ["warmth", Number(entity.warmth).toFixed(3)],
    ["energy", Number(entity.energy).toFixed(3)],
    ["capacity", Number(entity.capacity).toFixed(2)],
    ["conductivity", Number(entity.conductivity).toFixed(3)],
    ["mass", Number(entity.mass).toFixed(2)],
  ];
  return (
    <div className="panel side inspector">
      <h3>
        inspector <span className="muted">(god view)</span>
        <button
          className="close"
          onClick={() => {
            store.selectedId = null;
            store.inspected = null;
            emit();
          }}
        >
          ×
        </button>
      </h3>
      {rows.map(([k, v]) => (
        <div className="row kv" key={k}>
          <span>{k}</span>
          <span className="num">{v}</span>
        </div>
      ))}
    </div>
  );
}

// A small live performance readout (also logged by the server every few seconds).
function PerfReadout() {
  const perf = useStore((s) => s.perf);
  if (!perf) return null;
  const slow = perf.fps < 45;
  return (
    <div className={`panel perf ${slow ? "slow" : ""}`}>
      <span>{perf.fps} fps</span>
      <span className="muted">{perf.frameMs} ms (worst {perf.worstFrameMs})</span>
      <span className="muted">
        {perf.drawCalls} draws · {(perf.triangles / 1000).toFixed(0)}k tris
      </span>
      <span className="muted">
        server {perf.serverStepMs.toFixed(1)}+{perf.serverBuildMs.toFixed(1)} ms · msg {perf.messageKb}kb / {perf.parseMs}ms
      </span>
    </div>
  );
}

// Shows what the heat view's colours mean right now (the range stretches to fit the world).
function HeatLegend() {
  const on = useStore((s) => s.heatView);
  const [lo, hi] = useStore((s) => s.heatRange);
  if (!on) return null;
  const color = new THREE.Color();
  const stops = Array.from({ length: 9 }, (_, i) => `#${turbo(i / 8, color).getHexString(THREE.SRGBColorSpace)}`);
  return (
    <div className="panel legend">
      <span className="muted">warmth</span>
      <span className="num">{lo.toFixed(2)}</span>
      <div className="ramp" style={{ background: `linear-gradient(90deg, ${stops.join(",")})` }} />
      <span className="num">{hi.toFixed(2)}</span>
      <span className="muted">· white = very hot</span>
    </div>
  );
}

function KeyHints() {
  const hints = [
    ["WASD", "move"],
    ["Space+WASD", "push (away from facing: pull)"],
    ["Shift+WASD", "turn"],
    ["E", "pick up"],
    ["R", "put down"],
    ["F", "hit"],
    ["C", "dig under"],
    ["V", "pillar up"],
    ["Space tap", "wait"],
    ["H", "heat view"],
    ["G", "agent view"],
    ["P", "pause"],
    ["N", "step"],
    ["click", "inspect"],
    ["middle-click", "deselect"],
    ["right-drag", "rotate"],
    ["middle-drag", "pan"],
    ["scroll", "zoom to focus"],
  ];
  return (
    <div className="panel hints">
      {hints.map(([key, what]) => (
        <span key={key}>
          <kbd>{key}</kbd> {what}
        </span>
      ))}
    </div>
  );
}
