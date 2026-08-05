/**
 * The X/Y patch map runtime — a direct mirror of the lab's `patchmap.py`.
 *
 * Any change here must be mirrored there (and vice versa), because the build
 * VALIDATES the surface by rendering exactly what these functions produce. If
 * the two drift, the artifact stops describing what the user hears — the
 * failure mode that made a dial sweep painful before it was caught.
 *
 * The blend is deliberately not a smooth average. Weights are inverse distance
 * raised to a high power, which makes the surface a SOFT VORONOI: across most
 * of its area one anchor dominates so completely that you are standing on that
 * validated patch, and the blending happens in narrow bands at the boundaries
 * between neighbours. Measured over a 40-anchor map, `sharpness` 12 leaves
 * about two thirds of the surface reproducing a checked patch exactly.
 */

export interface MapPoint {
  preset_id: string;
  name: string;
  fxp_path: string;
  /** Position in the unit square. */
  xy: [number, number];
}

/**
 * One WORLD for a role: a start patch plus the map of everything reachable
 * through it.
 *
 * The lens supplies the oscillators, filter types and routing that every point
 * on this map is heard through, and the runtime can never write those — so a
 * map is not "the instrument", it is one structure's worth of it. Exhaust a
 * map and the way onward is a different lens, not more dragging.
 */
export interface MapLens {
  lens: { preset_id: string; name: string; category: string };
  param_names: string[];
  points: MapPoint[];
  /** Absolute parameter values at each point, in `param_names` order. */
  snapshots: number[][];
  sharpness: number;
  neighbours: number;
  snap: number;
}

export interface MapRole {
  role: string;
  track_id?: string;
  lenses: MapLens[];
  declined: boolean;
}

export interface MapGraph {
  version: string;
  roles: Record<string, MapRole>;
  quality?: Record<string, unknown>;
}

/**
 * The lens at `index`, WRAPPING.
 *
 * Wrapping rather than clamping because the index is shared across roles that
 * have different numbers of viable lenses, and because a layered group spreads
 * its members across the list — with six members and three lenses, clamping
 * would pile members 3-5 onto the last one while wrapping keeps them spread.
 */
export function lensAt(role: MapRole, index: number): MapLens | null {
  const n = role.lenses?.length ?? 0;
  if (n === 0) return null;
  return role.lenses[((Math.floor(index) % n) + n) % n];
}

const EPS = 1e-9;

/**
 * Indices and weights of the anchors contributing at `xy`.
 *
 * Exported for tests: the snap behaviour and the convexity of the weights are
 * the two properties the whole design rests on.
 */
export function blendWeights(
  points: ReadonlyArray<readonly [number, number]>,
  xy: readonly [number, number],
  neighbours = 4,
  sharpness = 20,
  snap = 0.9,
): { idx: number[]; w: number[] } {
  if (points.length === 0) return { idx: [], w: [] };

  const d = points.map((p, i) => ({
    i,
    d: Math.hypot(p[0] - xy[0], p[1] - xy[1]),
  }));
  d.sort((a, b) => (a.d === b.d ? a.i - b.i : a.d - b.d));
  const near = d.slice(0, Math.min(neighbours, d.length));

  // standing exactly on an anchor: it IS that patch, no arithmetic
  if (near[0].d < EPS) {
    return { idx: near.map((n) => n.i), w: near.map((_, k) => (k === 0 ? 1 : 0)) };
  }

  const raw = near.map((n) => 1 / Math.pow(n.d, sharpness));
  const total = raw.reduce((a, b) => a + b, 0);
  const w = raw.map((v) => v / total);

  // one anchor dominates — let it win outright, so most of the map
  // reproduces a validated patch EXACTLY rather than to within rounding
  if (w[0] >= snap) {
    return { idx: near.map((n) => n.i), w: near.map((_, k) => (k === 0 ? 1 : 0)) };
  }
  return { idx: near.map((n) => n.i), w };
}

/**
 * How certain a position is — the dominance of its nearest anchor, 0..1.
 *
 * 1 means "this IS a validated patch"; lower means the sound is a hybrid of
 * several, which is still inside the convex hull of checked configurations but
 * has not itself been rendered. Used to sit uncertain layers back in the mix.
 */
export function confidenceAt(
  role: Pick<MapLens, 'points' | 'sharpness' | 'neighbours' | 'snap'>,
  xy: readonly [number, number],
): number {
  const pts = role.points.map((p) => p.xy as readonly [number, number]);
  const { w } = blendWeights(pts, xy, role.neighbours, role.sharpness, role.snap);
  return w.length ? w[0] : 1;
}

/**
 * The map point nearest `xy` — where a gesture resolves when it is released.
 */
export function nearestPoint(
  role: Pick<MapLens, 'points'>,
  xy: readonly [number, number],
): readonly [number, number] | null {
  let best: readonly [number, number] | null = null;
  let bd = Infinity;
  for (const p of role.points) {
    const d = Math.hypot(p.xy[0] - xy[0], p.xy[1] - xy[1]);
    if (d < bd) { bd = d; best = p.xy as readonly [number, number]; }
  }
  return best;
}

/** The parameter vector at a point on the map. */
export function paramsAtXY(
  role: Pick<MapLens, 'points' | 'snapshots' | 'sharpness' | 'neighbours' | 'snap'>,
  xy: readonly [number, number],
): number[] {
  const pts = role.points.map((p) => p.xy as readonly [number, number]);
  const { idx, w } = blendWeights(
    pts, xy, role.neighbours, role.sharpness, role.snap,
  );
  if (idx.length === 0) return [];
  const width = role.snapshots[idx[0]]?.length ?? 0;
  const out = new Array<number>(width).fill(0);
  idx.forEach((anchor, k) => {
    const snap = role.snapshots[anchor];
    if (!snap || w[k] === 0) return;
    for (let j = 0; j < width; j++) out[j] += (snap[j] ?? 0) * w[k];
  });
  return out;
}

/**
 * Indices of every parameter the map moves ANYWHERE.
 *
 * Written at every position including where the delta is zero — see the
 * matching note on the tour's `movedParamIndices`. A parameter that is not
 * sent keeps whatever the previous position left it at, which silently makes
 * the control path-dependent.
 */
export function movedParamIndices(snapshots: readonly number[][]): number[] {
  const start = snapshots[0] ?? [];
  const out: number[] = [];
  start.forEach((_, i) => {
    if (snapshots.some((s) => Math.abs((s[i] ?? 0) - (start[i] ?? 0)) > 1e-5)) {
      out.push(i);
    }
  });
  return out;
}
