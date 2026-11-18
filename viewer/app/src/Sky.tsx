// The backdrop behind the 3D world: a clean sky gradient that follows the time of day, a warm glow
// at sunrise and sunset, and stars at night. Plain 2D (CSS), drawn behind the canvas, so it costs
// almost nothing per frame. The sun and moon themselves are drawn in 3D (Scene.tsx), so they sit
// where the light comes from. Pure decoration: none of this is part of the simulation.

import { useStore } from "./net";

type RGB = [number, number, number];

const mix = (a: RGB, b: RGB, t: number): RGB => a.map((v, i) => Math.round(v + (b[i] - v) * t)) as RGB;
const css = (c: RGB) => `rgb(${c[0]},${c[1]},${c[2]})`;

// Sky colours (top, horizon) at night, at sunrise/sunset, and at noon.
const NIGHT: [RGB, RGB] = [[18, 24, 54], [52, 58, 100]];
const DAWN: [RGB, RGB] = [[96, 120, 186], [236, 176, 140]];
const NOON: [RGB, RGB] = [[104, 158, 222], [206, 226, 244]];

function skyColors(sun: number): [RGB, RGB] {
  if (sun < 0.25) {
    const t = sun / 0.25;
    return [mix(NIGHT[0], DAWN[0], t), mix(NIGHT[1], DAWN[1], t)];
  }
  const t = Math.min(1, (sun - 0.25) / 0.6);
  return [mix(DAWN[0], NOON[0], t), mix(DAWN[1], NOON[1], t)];
}

export function Sky() {
  const sun = useStore((s) => s.tick?.sun ?? 0.6);
  const phase = useStore((s) => (s.tick && s.init ? (s.tick.tick % s.init.dayLength) / s.init.dayLength : 0.25));
  const daylight = useStore((s) => s.init?.daylight ?? 2 / 3);

  const [top, horizon] = skyColors(sun);
  const night = Math.max(0, 1 - sun * 4);
  const glow = phase < daylight ? Math.max(0, 1 - Math.abs(sun - 0.12) / 0.25) : 0; // strongest near sunrise/sunset

  return (
    <div className="sky" style={{ background: `linear-gradient(to bottom, ${css(top)} 0%, ${css(horizon)} 100%)` }}>
      <div className="stars" style={{ opacity: night }} />
      <div
        className="horizon-glow"
        style={{ background: `radial-gradient(ellipse at 50% 100%, rgba(255,160,100,${0.4 * glow}) 0%, transparent 60%)` }}
      />
    </div>
  );
}
