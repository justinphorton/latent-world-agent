// The 3D world. Draws whatever the server last sent and animates smoothly between ticks.
// Nothing here decides what happens in the world; it only shows it.

import { useEffect, useMemo, useRef } from "react";
import { useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { OrbitControls, Sparkles } from "@react-three/drei";
import * as THREE from "three";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import { emit, messageStats, send, store, useStore, type Init } from "./net";
import { cameraGround } from "./keys";

// World (x, y, z), z up  ->  three.js (x, y, z), y up, north going into the screen.
function toScene(init: Init, x: number, y: number, z: number, out: THREE.Vector3) {
  const [w, d] = init.shape;
  return out.set(x - (w - 1) / 2, z, -(y - (d - 1) / 2));
}

// Small stable per-entity variation, so no two blocks look exactly alike.
function jitter(id: number) {
  const s = Math.sin(id * 12.9898) * 43758.5453;
  return s - Math.floor(s);
}

// Heat view colours: the "turbo" colour map (dark blue -> cyan -> green -> yellow -> red), stretched
// over the warmth range that currently exists, so small differences are clearly visible.
// Anything near its cracking point (warmth > 0.85) additionally glows white, whatever the range.
export function turbo(t: number, out: THREE.Color) {
  t = Math.min(1, Math.max(0, t));
  const r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  const g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  const b = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  return out.setRGB(Math.min(1, Math.max(0, r)), Math.min(1, Math.max(0, g)), Math.min(1, Math.max(0, b)), THREE.SRGBColorSpace);
}

const WHITE = new THREE.Color("#ffffff");
function heatColor(warmth: number, out: THREE.Color) {
  const [lo, hi] = store.heatRange;
  turbo((warmth - lo) / (hi - lo), out);
  if (warmth > 0.85) out.lerp(WHITE, Math.min(1, (warmth - 0.85) * 4));
  return out;
}

const BASE_TICKS_PER_SECOND = 5; // matches viewer/server.py

const LOOK: Record<string, { roughness: number; opacity?: number }> = {
  bedrock: { roughness: 0.95 },
  soil: { roughness: 0.95 },
  stone: { roughness: 0.75 },
  sponge: { roughness: 1.0 },
  crystal: { roughness: 0.15, opacity: 0.85 },
};

const temp = {
  matrix: new THREE.Matrix4(),
  position: new THREE.Vector3(),
  target: new THREE.Vector3(),
  quaternion: new THREE.Quaternion(),
  scale: new THREE.Vector3(1, 1, 1),
  color: new THREE.Color(),
  euler: new THREE.Euler(),
};

export function Scene() {
  const init = store.init;
  if (!init) return null;
  return (
    <>
      <Lighting />
      <Entities init={init} />
      <Body init={init} />
      <Markers init={init} />
      <Underside init={init} />
      <Fireflies init={init} />
      <PerfMeter />
      <OrbitControls
        makeDefault
        target={[0, 2, 0]}
        maxPolarAngle={Math.PI / 2.15}
        enableDamping
        enableZoom={false} // zooming is handled smoothly by ZoomToFocus
        // Left click is for selecting; right drag rotates, middle drag pans.
        mouseButtons={{ LEFT: -1 as THREE.MOUSE, MIDDLE: THREE.MOUSE.PAN, RIGHT: THREE.MOUSE.ROTATE }}
      />
      <CameraTracker />
      <ZoomToFocus init={init} />
    </>
  );
}

// ---------------------------------------------------------------------------
// Blocks and moss: one instanced mesh per kind. Buried blocks (enclosed on all six sides) are
// never drawn, and per-frame work is limited to things that are actually moving.
// ---------------------------------------------------------------------------

// What the viewer currently shows for one entity (its animated position) and where that is in
// the instanced meshes.
type Shown = { pos: THREE.Vector3; target: THREE.Vector3; grow: number };
type Slot = { entry: Shown; mesh: THREE.InstancedMesh; index: number; cover: boolean; j: number; shadowIndex: number };

const MOSS_DRY = new THREE.Color("#9a8a4a");
const RED_HOT = new THREE.Color("#ffb070");

// Whether anything is animating this frame; shadows only need redrawing when something moves.
const animation = { entities: true, body: true };

function Entities({ init }: { init: Init }) {
  const capacity = init.shape[0] * init.shape[1] * init.shape[2];
  // One instanced mesh per kind, plus one extra for soil whose top is open to the sky, which gets
  // a Minecraft-style topsoil shell (its own top face, and a band wrapping down its sides).
  const soilColor = init.kinds.find((k) => k?.name === "soil")?.color ?? "#8a5a3b";
  const looks = useMemo(() => [...init.kinds, { name: "soil-surface", layer: "block", color: soilColor }], [init, soilColor]);
  const SURFACE = init.kinds.length;
  const SOIL = init.kinds.findIndex((k) => k?.name === "soil");
  const meshes = useRef<(THREE.InstancedMesh | null)[]>([]);
  const ids = useRef<number[][]>(looks.map(() => []));
  const shown = useRef(new Map<number, Shown>());

  // Buried blocks aren't drawn in full. Instead each becomes a plain, untextured full-size cube in a
  // darkened version of its colour, so every crack between the rounded blocks ends on something solid.
  const core = useRef<THREE.InstancedMesh | null>(null);
  const coreGeometry = useMemo(() => new THREE.BoxGeometry(1, 1, 1), []);
  const coreMaterial = useMemo(() => new THREE.MeshLambertMaterial(), []);

  // Shadows are cast by invisible plain boxes (12 triangles each) standing in for the rounded blocks
  // (~300 each): the shadows look the same, and the shadow pass becomes ~25x cheaper.
  const shadowCaster = useRef<THREE.InstancedMesh | null>(null);
  const shadowGeometry = useMemo(() => new THREE.BoxGeometry(0.97, 0.97, 0.97), []);
  const shadowMaterial = useMemo(() => new THREE.MeshBasicMaterial({ colorWrite: false, depthWrite: false }), []);

  // Soft, chunky, well-rounded blocks with a small gap between them: the stylized look.
  const blockGeometry = useMemo(() => new RoundedBoxGeometry(0.97, 0.97, 0.97, 2, 0.09), []);
  const mossGeometry = useMemo(() => new THREE.IcosahedronGeometry(0.42, 0), []);

  const baseColors = useMemo(() => looks.map((k) => new THREE.Color(k?.color ?? "#888")), [looks]);

  // One material per look; looks with a special top face get a material per box face instead.
  // Box faces are ordered +x, -x, +y (top), -y, +z, -z.
  const materials = useMemo(
    () =>
      looks.map((kind) => {
        if (!kind || kind.layer === "creature") return null;
        const look = LOOK[kind.name] ?? { roughness: 0.95 };
        // Plain, smooth colour: each block's tint comes from its instance colour.
        const make = (shade = 1) =>
          new THREE.MeshStandardMaterial({
            color: new THREE.Color(shade, shade, shade),
            roughness: look.roughness,
            metalness: kind.name === "crystal" ? 0.1 : 0,
            transparent: look.opacity !== undefined,
            opacity: look.opacity ?? 1,
            flatShading: kind.layer === "cover",
          });
        if (kind.name === "soil-surface") {
          // Exposed soil: the same block, with a darker brown top face as its outer layer.
          const side = make();
          return [side, side, make(0.62), side, side, side];
        }
        return make();
      }),
    [looks],
  );

  // Work is split so a quiet frame costs almost nothing:
  //  - when a new tick arrives (a few times a second), every visible entity's slot and colour is
  //    rewritten, and anything not yet at its new position is put on the `moving` list;
  //  - every frame, only the entities on the moving list are animated.
  const built = useRef<{ tick: unknown; heat: boolean }>({ tick: null, heat: false });
  const moving = useRef(new Map<number, Slot>());

  const placeMatrix = (slot: Slot) => {
    const { entry, cover, j } = slot;
    temp.position.copy(entry.pos);
    if (cover) {
      // Moss: a low, lumpy tuft lying on the block below, a little different every time.
      temp.position.y -= 0.42;
      temp.euler.set(0, j * Math.PI * 2, 0);
      temp.quaternion.setFromEuler(temp.euler);
      const s = (0.95 + j * 0.25) * entry.grow;
      temp.scale.set(s * 1.15, s * 0.32, s * 1.15);
    } else {
      temp.quaternion.identity();
      temp.scale.setScalar(entry.grow);
    }
    temp.matrix.compose(temp.position, temp.quaternion, temp.scale);
    slot.mesh.setMatrixAt(slot.index, temp.matrix);
    if (slot.shadowIndex >= 0) shadowCaster.current?.setMatrixAt(slot.shadowIndex, temp.matrix);
  };

  useFrame((_, delta) => {
    const tick = store.tick;
    if (!tick) return;
    const touched = new Set<THREE.InstancedMesh>();

    if (built.current.tick !== tick || built.current.heat !== store.heatView) {
      built.current = { tick, heat: store.heatView };
      const { hidden, isBlock } = buriedBlocks(init, tick.entities);
      const counts = looks.map(() => 0);
      const seen = new Set<number>();
      moving.current.clear();
      let cores = 0;
      let casters = 0;

      for (const [id, kindIndex, x, y, z, warmth, tone] of tick.entities) {
        const kind = init.kinds[kindIndex];
        if (!kind || kind.layer === "creature") continue;
        if (hidden.has(id)) {
          if (core.current) {
            toScene(init, x, y, z, temp.position);
            temp.matrix.makeTranslation(temp.position.x, temp.position.y, temp.position.z);
            core.current.setMatrixAt(cores, temp.matrix);
            core.current.setColorAt(cores, temp.color.copy(baseColors[kindIndex]).multiplyScalar(0.35));
            cores++;
          }
          continue;
        }
        // Soil with no block on top is drawn with the topsoil shell.
        const lookIndex = kindIndex === SOIL && !isBlock(x, y, z + 1) ? SURFACE : kindIndex;
        const mesh = meshes.current[lookIndex];
        if (!mesh) continue;
        seen.add(id);

        let entry = shown.current.get(id);
        if (!entry) {
          // Moss grows in; blocks just appear (they're usually being uncovered, not created).
          entry = { pos: toScene(init, x, y, z, new THREE.Vector3()), target: new THREE.Vector3(), grow: kind.layer === "cover" ? 0 : 1 };
          shown.current.set(id, entry);
        }
        toScene(init, x, y, z, entry.target);
        const cover = kind.layer === "cover";
        const slot: Slot = { entry, mesh, index: counts[lookIndex]++, cover, j: jitter(id), shadowIndex: cover ? -1 : casters++ };
        placeMatrix(slot);
        if (entry.pos.distanceToSquared(entry.target) > 1e-6 || entry.grow < 1) moving.current.set(id, slot);

        // Colour: the kind's colour with its tone, or the heat ramp.
        if (store.heatView) heatColor(warmth, temp.color);
        else {
          // Tone comes from the simulation: an imperfect hint at the block's hidden properties.
          temp.color.copy(baseColors[lookIndex]).multiplyScalar(1 + tone * 0.28);
          if (kind.layer === "cover") temp.color.lerp(MOSS_DRY, Math.max(0, 0.3 - warmth));
          if (warmth > 0.75) temp.color.lerp(RED_HOT, Math.min(1, (warmth - 0.75) * 3));
        }
        mesh.setColorAt(slot.index, temp.color);
        ids.current[lookIndex][slot.index] = id;
      }

      if (core.current) {
        core.current.count = cores;
        core.current.instanceMatrix.needsUpdate = true;
        if (core.current.instanceColor) core.current.instanceColor.needsUpdate = true;
      }
      if (shadowCaster.current) {
        shadowCaster.current.count = casters;
        touched.add(shadowCaster.current);
      }
      meshes.current.forEach((mesh, lookIndex) => {
        if (!mesh) return;
        mesh.count = counts[lookIndex];
        if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
        touched.add(mesh);
      });
      if (shown.current.size > seen.size + 200) {
        for (const id of shown.current.keys()) if (!seen.has(id)) shown.current.delete(id);
      }
    }

    // Animate only what is moving or growing.
    const follow = 1 - Math.exp(-delta * 14);
    for (const [id, slot] of moving.current) {
      const { entry } = slot;
      entry.pos.lerp(entry.target, follow);
      entry.grow = Math.min(1, entry.grow + delta * 4);
      if (entry.pos.distanceToSquared(entry.target) < 1e-6 && entry.grow >= 1) {
        entry.pos.copy(entry.target);
        moving.current.delete(id);
      }
      placeMatrix(slot);
      touched.add(slot.mesh);
      if (slot.shadowIndex >= 0 && shadowCaster.current) touched.add(shadowCaster.current);
    }
    for (const mesh of touched) mesh.instanceMatrix.needsUpdate = true;
    animation.entities = moving.current.size > 0 || touched.size > 0;
  });

  // Only a plain left click selects: not middle or right, and not the end of a camera drag.
  const inspect = (lookIndex: number) => (event: ThreeEvent<MouseEvent>) => {
    if (event.nativeEvent.button !== 0 || event.delta > 4) return;
    event.stopPropagation();
    if (event.instanceId === undefined) return;
    const id = ids.current[lookIndex][event.instanceId];
    store.selectedId = id;
    send({ type: "inspect", id });
  };

  return (
    <>
      <instancedMesh
        ref={(mesh) => {
          core.current = mesh;
          if (mesh && !mesh.instanceColor) mesh.setColorAt(0, new THREE.Color());
        }}
        args={[coreGeometry, coreMaterial, capacity]}
        frustumCulled={false}
      />
      <instancedMesh
        ref={(mesh) => {
          shadowCaster.current = mesh;
        }}
        args={[shadowGeometry, shadowMaterial, capacity]}
        castShadow
        frustumCulled={false}
        raycast={() => null} // invisible: never the target of a click
      />
      {looks.map((kind, lookIndex) => {
        const material = materials[lookIndex];
        if (!kind || !material) return null;
        return (
          <instancedMesh
            key={kind.name}
            ref={(mesh) => {
              meshes.current[lookIndex] = mesh;
              if (mesh && !mesh.instanceColor) mesh.setColorAt(0, new THREE.Color());
            }}
            args={[kind.layer === "cover" ? mossGeometry : blockGeometry, material, capacity]}
            receiveShadow
            onClick={inspect(lookIndex)}
            frustumCulled={false}
          />
        );
      })}
    </>
  );
}

// Ids of blocks buried two layers deep: enclosed on all six sides by blocks that are themselves
// enclosed (the floor of the world counts as covered from below). Blocks just under the surface
// are still drawn, because the rounded edges leave small grooves you can see a little way into.
// Also returns isBlock(x, y, z), so callers can ask what's next to a block.
function buriedBlocks(init: Init, entities: number[][]) {
  const [w, d] = init.shape;
  const key = (x: number, y: number, z: number) => (z * d + y) * w + x;
  const solid = new Set<number>();
  for (const [, kind, x, y, z] of entities) if (init.kinds[kind]?.layer === "block") solid.add(key(x, y, z));
  const isBlock = (x: number, y: number, z: number) => x >= 0 && y >= 0 && x < w && y < d && solid.has(key(x, y, z));

  const FACES = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  const allAround = (x: number, y: number, z: number, filled: (x: number, y: number, z: number) => boolean) =>
    FACES.every(([dx, dy, dz]) => filled(x + dx, y + dy, z + dz));
  const inside = (x: number, y: number) => x >= 0 && y >= 0 && x < w && y < d;

  // First pass: blocks enclosed by blocks.
  const enclosed = new Set<number>();
  for (const [, kind, x, y, z] of entities) {
    if (init.kinds[kind]?.layer !== "block") continue;
    if (allAround(x, y, z, (a, b, c) => inside(a, b) && (c < 0 || solid.has(key(a, b, c))))) enclosed.add(key(x, y, z));
  }
  // Second pass: enclosed blocks whose neighbours are all enclosed too.
  const ids = new Set<number>();
  for (const [id, kind, x, y, z] of entities) {
    if (init.kinds[kind]?.layer !== "block" || !enclosed.has(key(x, y, z))) continue;
    if (allAround(x, y, z, (a, b, c) => c < 0 || enclosed.has(key(a, b, c)))) ids.add(id);
  }
  return { hidden: ids, isBlock };
}

// ---------------------------------------------------------------------------
// The body: a small round creature that looks the way it is facing.
// ---------------------------------------------------------------------------

const FACING_ANGLE: Record<string, number> = { e: 0, n: 90, w: 180, s: 270 };

function Body({ init }: { init: Init }) {
  const group = useRef<THREE.Group>(null);
  const shown = useRef<THREE.Vector3 | null>(null);
  const heading = useRef(0);
  const bounce = useRef(0);

  useFrame((state, delta) => {
    const body = store.tick?.body;
    if (!body || !group.current) return;
    toScene(init, ...body.pos, temp.target);
    temp.target.y -= 0.5;
    if (!shown.current) shown.current = temp.target.clone();
    const distance = shown.current.distanceTo(temp.target);
    shown.current.lerp(temp.target, 1 - Math.exp(-delta * 20));
    if (distance > 0.05) bounce.current = Math.min(1, bounce.current + delta * 6);
    else bounce.current = Math.max(0, bounce.current - delta * 4);
    animation.body = distance > 0.005 || bounce.current > 0;

    // Face the direction: world angle -> rotation about three.js y.
    const want = ((FACING_ANGLE[body.facing] ?? 90) * Math.PI) / 180;
    let diff = want - heading.current;
    diff = Math.atan2(Math.sin(diff), Math.cos(diff));
    heading.current += diff * (1 - Math.exp(-delta * 12));

    const t = state.clock.elapsedTime;
    const hop = Math.abs(Math.sin(t * 14)) * 0.12 * bounce.current;
    const breathe = 1 + Math.sin(t * 2) * 0.02;
    group.current.position.set(shown.current.x, shown.current.y + hop, shown.current.z);
    group.current.rotation.y = heading.current;
    group.current.scale.set(breathe, 1 / breathe, breathe);
  });

  return (
    <group ref={group}>
      <Critter />
    </group>
  );
}

// A small, cute blob creature: a soft squashed body, glossy dot eyes with a sparkle, blush marks
// and two little feet. It faces +x in its own space (the body group turns it to the facing
// direction). Its colour brightens, and glows a little, the more energy it has.
const CRITTER_COLOR = new THREE.Color("#ff9f80");

function Critter() {
  const skin = useRef<THREE.MeshStandardMaterial>(null);
  const eyes = useRef<(THREE.Object3D | null)[]>([]);
  const brightness = useRef<number | null>(null);

  useFrame((state, delta) => {
    const body = store.tick?.body;
    if (body && skin.current) {
      // 0 when empty, 1 at half capacity or more; eased so the change is gentle.
      const want = THREE.MathUtils.clamp(body.energy / body.capacity / 0.5, 0, 1);
      brightness.current = brightness.current === null ? want : brightness.current + (want - brightness.current) * Math.min(1, delta * 2);
      const b = brightness.current;
      skin.current.color.copy(CRITTER_COLOR).multiplyScalar(0.62 + 0.45 * b);
      skin.current.emissiveIntensity = 0.04 + 0.3 * b;
    }
    const blink = state.clock.elapsedTime % 3.7 < 0.12 ? 0.12 : 1;
    eyes.current.forEach((eye) => eye?.scale.set(1, blink, 1));
  });

  return (
    <group>
      {/* body: a slightly squashed ball */}
      <mesh position={[0, 0.31, 0]} scale={[1, 0.86, 1]} castShadow>
        <sphereGeometry args={[0.36, 32, 24]} />
        <meshStandardMaterial ref={skin} color={CRITTER_COLOR} emissive={CRITTER_COLOR} emissiveIntensity={0.2} roughness={0.45} />
      </mesh>
      {/* eyes: glossy dark ovals with a small white sparkle */}
      {[0.11, -0.11].map((side, i) => (
        <group key={side} ref={(g) => (eyes.current[i] = g)} position={[0.315, 0.38, side]}>
          <mesh scale={[0.5, 1.25, 0.9]}>
            <sphereGeometry args={[0.05, 14, 10]} />
            <meshStandardMaterial color="#1a1220" roughness={0.15} />
          </mesh>
          <mesh position={[0.022, 0.022, -0.012]}>
            <sphereGeometry args={[0.013, 8, 6]} />
            <meshBasicMaterial color="#ffffff" />
          </mesh>
        </group>
      ))}
      {/* blush */}
      {[0.2, -0.2].map((side) => (
        <mesh key={side} position={[0.285, 0.3, side]} rotation={[0, side > 0 ? -0.55 : 0.55, 0]} scale={[0.3, 0.5, 1]}>
          <sphereGeometry args={[0.05, 12, 8]} />
          <meshStandardMaterial color="#ff5f73" roughness={0.6} transparent opacity={0.5} />
        </mesh>
      ))}
      {/* two little feet */}
      {[0.13, -0.13].map((side) => (
        <mesh key={side} position={[0.1, 0.035, side]} scale={[1.25, 0.5, 1]} castShadow>
          <sphereGeometry args={[0.075, 12, 8]} />
          <meshStandardMaterial color="#e8805f" roughness={0.6} />
        </mesh>
      ))}
    </group>
  );
}

// ---------------------------------------------------------------------------
// Outlines: where the hands would act, and what is selected for inspection.
// ---------------------------------------------------------------------------

function Markers({ init }: { init: Init }) {
  const hand = useRef<THREE.LineSegments>(null);
  const selected = useRef<THREE.LineSegments>(null);
  const edges = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(1.04, 1.04, 1.04)), []);

  useFrame((state) => {
    const tick = store.tick;
    if (hand.current) {
      const target = tick?.body.target;
      hand.current.visible = !!target;
      if (target) toScene(init, ...target, hand.current.position);
      const material = hand.current.material as THREE.LineBasicMaterial;
      material.opacity = 0.55 + Math.sin(state.clock.elapsedTime * 5) * 0.3;
    }
    if (selected.current) {
      const row = tick?.entities.find((e) => e[0] === store.selectedId);
      selected.current.visible = !!row;
      if (row) toScene(init, row[2], row[3], row[4], selected.current.position);
    }
  });

  return (
    <>
      <lineSegments ref={hand} geometry={edges}>
        <lineBasicMaterial color="#fff6c8" transparent />
      </lineSegments>
      <lineSegments ref={selected} geometry={edges}>
        <lineBasicMaterial color="#7ee0ff" />
      </lineSegments>
    </>
  );
}

// ---------------------------------------------------------------------------
// The rock hanging under the island. Decoration only (not part of the simulation): it follows the
// island's real footprint, hangs deepest in the middle, and darkens with depth.
// ---------------------------------------------------------------------------

const ROCK_TOP = new THREE.Color("#6b5a52");
const ROCK_BOTTOM = new THREE.Color("#2a2433");

function Underside({ init }: { init: Init }) {
  const mesh = useRef<THREE.InstancedMesh>(null);
  const material = useMemo(() => new THREE.MeshLambertMaterial(), []);
  const seed = useStore((s) => s.init?.seed);

  const blocks = useMemo(() => {
    // The island's footprint: every column that has a floor.
    const floor = new Set<string>();
    for (const [, kind, x, y, z] of store.tick?.entities ?? []) {
      if (z === 0 && init.kinds[kind]?.name === "bedrock") floor.add(`${x},${y}`);
    }
    const [w, d] = init.shape;
    const cx = (w - 1) / 2;
    const cy = (d - 1) / 2;
    const radius = Math.min(w, d) / 2;
    const depths = new Map<string, number>();
    for (const key of floor) {
      const [x, y] = key.split(",").map(Number);
      const edge = 1 - Math.hypot(x - cx, y - cy) / radius; // 1 in the middle, 0 at the rim
      depths.set(key, Math.max(1, Math.round(Math.pow(Math.max(edge, 0), 0.8) * 9 + jitter(x * 97 + y) * 2.5)));
    }
    const out: { x: number; y: number; k: number; depth: number }[] = [];
    for (const [key, depth] of depths) {
      const [x, y] = key.split(",").map(Number);
      for (let k = 1; k <= depth; k++) out.push({ x, y, k, depth });
    }
    return out;
  }, [init, seed]);

  // Plain full-size cubes: no rounded edges means no gaps to see through, and only 12 triangles each.
  const geometry = useMemo(() => new THREE.BoxGeometry(1, 1, 1), []);

  useFrame(() => {
    if (!mesh.current || mesh.current.userData.built === blocks) return;
    blocks.forEach(({ x, y, k }, i) => {
      toScene(init, x, y, -k, temp.position);
      temp.scale.set(1, 1, 1);
      temp.matrix.compose(temp.position, temp.quaternion.identity(), temp.scale);
      mesh.current!.setMatrixAt(i, temp.matrix);
      temp.color.copy(ROCK_TOP).lerp(ROCK_BOTTOM, Math.min(1, k / 7)).multiplyScalar(0.9 + jitter(x * 13 + y * 7 + k) * 0.2);
      mesh.current!.setColorAt(i, temp.color);
    });
    mesh.current.count = blocks.length;
    mesh.current.instanceMatrix.needsUpdate = true;
    if (mesh.current.instanceColor) mesh.current.instanceColor.needsUpdate = true;
    mesh.current.userData.built = blocks;
  });

  return (
    <instancedMesh ref={mesh} args={[geometry, material, Math.max(1, blocks.length)]} key={blocks.length} frustumCulled={false} />
  );
}

// ---------------------------------------------------------------------------
// Light. The simulation's sun has no direction (it warms the top of every column equally), so the
// light comes from one fixed, mostly-overhead angle and only its brightness and colour follow the
// day: warm and bright at noon, orange at the edges of the day, blue moonlight at night. Shadows
// never move, and the same single light casts them day and night.
// ---------------------------------------------------------------------------

const LIGHT_DIRECTION = new THREE.Vector3(10, 30, 7); // overhead-ish, slightly from one side
const MOON = new THREE.Color("#9fb4ff");
const DAWN = new THREE.Color("#ff9a5a");
const NOON = new THREE.Color("#fff4e0");
const NIGHT_SKY = new THREE.Color("#9fb4ff");
const DAY_SKY = new THREE.Color("#cfe3ff");

function Lighting() {
  const light = useRef<THREE.DirectionalLight>(null);
  const sky = useRef<THREE.HemisphereLight>(null);
  const clock = useRef<number | null>(null); // a smoothly running copy of the world's tick
  const { gl } = useThree();
  const lastShadowTick = useRef<unknown>(null);

  useFrame((_, delta) => {
    const tick = store.tick;
    const init = store.init;
    if (!tick || !init || !light.current || !sky.current) return;

    // The light never moves, so shadows only need redrawing when something in the world moved.
    gl.shadowMap.autoUpdate = false;
    if (animation.entities || animation.body || lastShadowTick.current !== tick) {
      gl.shadowMap.needsUpdate = true;
      lastShadowTick.current = tick;
    }

    // The server ticks a few times a second; brightness changes continuously in between, at the
    // world's current speed, gently pulled back if it drifts from the real tick.
    if (clock.current === null || Math.abs(clock.current - tick.tick) > 3) clock.current = tick.tick;
    if (tick.running) clock.current += delta * BASE_TICKS_PER_SECOND * tick.speed;
    clock.current += (tick.tick - clock.current) * Math.min(1, delta * 2);
    const phase = (((clock.current % init.dayLength) + init.dayLength) % init.dayLength) / init.dayLength;
    // same curve as the simulation's sun_level: a hump over the daylight part, then dark
    const sun = phase < init.daylight ? Math.sin((Math.PI * phase) / init.daylight) : 0;

    // Day: warm sunlight, orange near sunrise/sunset. Night: the same light as blue moonlight.
    const day = Math.min(1, sun * 4); // 0 at night, 1 once the sun is properly up
    light.current.intensity = THREE.MathUtils.lerp(1.1, 2.8 * Math.pow(sun, 0.5), day);
    temp.color.copy(DAWN).lerp(NOON, Math.pow(sun, 0.5));
    light.current.color.copy(MOON).lerp(temp.color, day);

    sky.current.intensity = 0.8 + 0.4 * sun;
    sky.current.color.copy(NIGHT_SKY).lerp(DAY_SKY, Math.min(1, sun * 2.5));
  });

  return (
    <>
      <hemisphereLight ref={sky} args={["#cfe3ff", "#3b2f3a", 0.8]} />
      <directionalLight
        ref={light}
        position={LIGHT_DIRECTION}
        castShadow
        shadow-mapSize={[2048, 2048]}
        shadow-camera-left={-24}
        shadow-camera-right={24}
        shadow-camera-top={24}
        shadow-camera-bottom={-24}
        shadow-camera-near={1}
        shadow-camera-far={90}
        // Small offsets: big ones detach shadows from where things touch, so blocks look like they float.
        shadow-bias={-0.0002}
        shadow-normalBias={0.008}
        shadow-radius={4}
      />
    </>
  );
}

// Measures how the viewer performs: refreshed on screen once a second, and reported to the server
// every 5 seconds, which logs it (logs/viewer-perf.jsonl) so it can be checked without the browser.
function PerfMeter() {
  const { gl } = useThree();
  const window_ = useRef({ frames: 0, time: 0, worst: 0, sinceReport: 0 });

  useFrame((_, delta) => {
    const w = window_.current;
    w.frames++;
    w.time += delta;
    w.worst = Math.max(w.worst, delta);
    if (w.time < 1) return;

    const timing = store.tick?.timing;
    store.perf = {
      fps: Math.round(w.frames / w.time),
      frameMs: +((w.time / w.frames) * 1000).toFixed(1),
      worstFrameMs: +(w.worst * 1000).toFixed(1),
      drawCalls: gl.info.render.calls,
      triangles: gl.info.render.triangles,
      parseMs: +messageStats.parseMs.toFixed(2),
      messageKb: +messageStats.kb.toFixed(1),
      serverStepMs: timing?.stepMs ?? 0,
      serverBuildMs: timing?.buildMs ?? 0,
    };
    emit();
    w.sinceReport += w.time;
    if (w.sinceReport >= 5) {
      send({ type: "perf", ...store.perf, pixelRatio: gl.getPixelRatio(), width: gl.domElement.width, height: gl.domElement.height });
      w.sinceReport = 0;
    }
    w.frames = 0;
    w.time = 0;
    w.worst = 0;
  });
  return null;
}

// Drifting specks of light at night. Pure decoration: they are not part of the simulation.
function Fireflies({ init }: { init: Init }) {
  const night = useStore((s) => (s.tick?.sun ?? 1) < 0.08);
  if (!night) return null;
  const width = init.shape[0] * 0.8;
  return <Sparkles count={70} scale={[width, 3, width]} position={[0, 4, 0]} size={5} speed={0.25} color="#ffe28a" opacity={0.9} />;
}

// Smooth zooming. Scrolling in also brings a focus to the middle of the view: the selected entity if
// there is one, otherwise the body. The zoom happens around the focus (so it never pushes the focus
// away on screen). If the focus is outside a calm zone around the middle, a separate ease also pulls
// it to the centre; inside the zone the view doesn't slide at all. Scrolling out zooms around the
// middle. Middle-drag panning cancels the centering.
const MIN_ZOOM = 6;
const MAX_ZOOM = 140;
const CENTER_ZONE = 0.3; // focus within this share of the half-screen from the middle: no sliding

function ZoomToFocus({ init }: { init: Init }) {
  const { gl, camera, controls } = useThree();
  const zoomGoal = useRef<number | null>(null); // zoom still to reach
  const focus = useRef<THREE.Vector3 | null>(null); // the point being zoomed toward
  const centering = useRef(false); // whether the view is currently sliding the focus to the middle

  // How far a point is from the middle of the screen: 0 = centre, 1 = edge.
  const screenDistance = (point: THREE.Vector3) => {
    const ndc = point.clone().project(camera);
    return Math.max(Math.abs(ndc.x), Math.abs(ndc.y));
  };

  useEffect(() => {
    const onWheel = (event: WheelEvent) => {
      event.preventDefault();
      const tick = store.tick;
      if (!tick) return;
      const before = zoomGoal.current ?? camera.zoom;
      const after = THREE.MathUtils.clamp(before * Math.exp(-event.deltaY * 0.0015), MIN_ZOOM, MAX_ZOOM);
      zoomGoal.current = after;
      if (after <= before) return; // zooming out: around the middle of the view
      const row = store.selectedId !== null ? tick.entities.find((e) => e[0] === store.selectedId) : undefined;
      const point = new THREE.Vector3();
      if (row) toScene(init, row[2], row[3], row[4], point);
      else toScene(init, ...tick.body.pos, point);
      focus.current = point;
      // Only slide the view if the focus is outside the calm zone around the middle of the screen.
      if (!centering.current && screenDistance(point) > CENTER_ZONE) centering.current = true;
    };
    // Middle button: stop the browser's autoscroll (so middle-drag only pans), and treat a middle
    // click that didn't drag as "clear the selection".
    let middleDownAt: { x: number; y: number } | null = null;
    const onMouseDown = (event: MouseEvent) => {
      if (event.button !== 1) return;
      event.preventDefault();
      middleDownAt = { x: event.clientX, y: event.clientY };
      centering.current = false; // you're taking the camera: stop centering
    };
    const onMouseUp = (event: MouseEvent) => {
      if (event.button !== 1 || !middleDownAt) return;
      const moved = Math.hypot(event.clientX - middleDownAt.x, event.clientY - middleDownAt.y);
      middleDownAt = null;
      if (moved < 4 && store.selectedId !== null) {
        store.selectedId = null;
        store.inspected = null;
        emit();
      }
    };
    gl.domElement.addEventListener("wheel", onWheel, { passive: false });
    gl.domElement.addEventListener("mousedown", onMouseDown);
    gl.domElement.addEventListener("mouseup", onMouseUp);
    return () => {
      gl.domElement.removeEventListener("wheel", onWheel);
      gl.domElement.removeEventListener("mousedown", onMouseDown);
      gl.domElement.removeEventListener("mouseup", onMouseUp);
    };
  }, [gl, camera, controls, init]);

  useFrame((_, delta) => {
    const orbit = controls as unknown as { target: THREE.Vector3 } | null;
    if (!orbit) return;

    if (zoomGoal.current !== null) {
      const before = camera.zoom;
      camera.zoom += (zoomGoal.current - camera.zoom) * (1 - Math.exp(-delta * 10));
      if (Math.abs(zoomGoal.current - camera.zoom) < 1e-3) {
        camera.zoom = zoomGoal.current;
        zoomGoal.current = null;
      }
      camera.updateProjectionMatrix();

      // Zoom *around* the focus, not around the middle, so zooming alone never pushes the focus
      // outward on screen (that outward-then-back wobble is what looked like jitter). Keeping a
      // point P still while zooming means moving the view centre to P + (centre - P) * before/after.
      if (focus.current && camera.zoom > before) {
        const step = focus.current.clone().sub(orbit.target).multiplyScalar(1 - before / camera.zoom);
        orbit.target.add(step);
        camera.position.add(step);
      }
    }

    if (focus.current && centering.current) {
      // Ease the view's centre onto the focus; move the camera with it so the angle is unchanged.
      const remaining = focus.current.clone().sub(orbit.target);
      const step = remaining.clone().multiplyScalar(1 - Math.exp(-delta * 6));
      orbit.target.add(step);
      camera.position.add(step);
      if (remaining.lengthSq() < 1e-4) centering.current = false;
    }
    if (zoomGoal.current === null && !centering.current) focus.current = null;
  });
  return null;
}

// Keeps keys.ts informed of which way the camera faces, so W always means "up the screen".
function CameraTracker() {
  const { camera, controls } = useThree();
  useFrame(() => {
    const target = (controls as unknown as { target?: THREE.Vector3 } | null)?.target ?? new THREE.Vector3();
    const x = target.x - camera.position.x;
    const z = target.z - camera.position.z;
    const length = Math.hypot(x, z) || 1;
    cameraGround.x = x / length;
    cameraGround.z = z / length;
  });
  return null;
}
