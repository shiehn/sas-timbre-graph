/**
 * The X/Y map runtime, and its agreement with the lab.
 *
 * The build validates the surface by RENDERING what `patchmap.py` produces. If
 * this TypeScript drifts from that Python by even a little, the artifact stops
 * describing what the user hears — and every safety claim made about it
 * becomes a claim about a different sound. So the cross-language vectors are
 * not a nicety; they are the thing that keeps the gate honest.
 */

import {
  blendWeights,
  confidenceAt,
  movedParamIndices,
  nearestPoint,
  paramsAtXY,
} from '../src/patchmap';
import vectors from './patchmap-vectors.json';

type XY = readonly [number, number];

const POINTS = vectors.points as unknown as XY[];
const ROLE = {
  points: POINTS.map((xy, i) => ({
    preset_id: `p${i}`, name: `p${i}`, fxp_path: `p${i}.fxp`, xy: xy as [number, number],
  })),
  snapshots: vectors.snapshots as number[][],
  sharpness: 12,
  neighbours: 4,
  snap: 0.9,
};

describe('the map runtime agrees with the lab that validated it', () => {
  it.each(vectors.cases.map((c, i) => [i, c] as const))(
    'matches python at case %i',
    (_i, c) => {
      const got = paramsAtXY(ROLE, c.xy as unknown as XY);
      expect(got).toHaveLength(c.params.length);
      got.forEach((v, j) => expect(v).toBeCloseTo(c.params[j], 9));
    },
  );
});

describe('standing on a patch gives you that patch', () => {
  it('reproduces an anchor exactly at its own coordinates', () => {
    POINTS.forEach((xy, i) => {
      const { idx, w } = blendWeights(POINTS, xy);
      expect(idx[0]).toBe(i);
      expect(w[0]).toBe(1);
      expect(paramsAtXY(ROLE, xy)).toEqual(ROLE.snapshots[i]);
    });
  });

  it('snaps outright across most of the surface, not just at the points', () => {
    // the "cheat": the user must not be able to wander far from a real patch
    let exact = 0;
    const N = 30;
    for (let a = 0; a < N; a++) {
      for (let b = 0; b < N; b++) {
        const { w } = blendWeights(POINTS, [a / (N - 1), b / (N - 1)]);
        if (w[0] === 1) exact++;
      }
    }
    expect(exact / (N * N)).toBeGreaterThan(0.5);
  });
});

describe('the blend can never invent a sound', () => {
  it('weights are non-negative and sum to one everywhere', () => {
    for (let a = 0; a <= 10; a++) {
      for (let b = 0; b <= 10; b++) {
        const { w } = blendWeights(POINTS, [a / 10, b / 10]);
        expect(Math.min(...w)).toBeGreaterThanOrEqual(0);
        expect(w.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 9);
      }
    }
  });

  it('never extrapolates past the anchors it blends', () => {
    for (let a = 0; a <= 8; a++) {
      for (let b = 0; b <= 8; b++) {
        const xy: XY = [a / 8, b / 8];
        const { idx } = blendWeights(POINTS, xy);
        const got = paramsAtXY(ROLE, xy);
        got.forEach((v, j) => {
          const near = idx.map((i) => ROLE.snapshots[i][j]);
          expect(v).toBeGreaterThanOrEqual(Math.min(...near) - 1e-9);
          expect(v).toBeLessThanOrEqual(Math.max(...near) + 1e-9);
        });
      }
    }
  });

  it('sharpness tightens the surface monotonically', () => {
    const dominance = (s: number): number => {
      let t = 0;
      for (let a = 0; a <= 12; a++)
        for (let b = 0; b <= 12; b++)
          t += blendWeights(POINTS, [a / 12, b / 12], 4, s)[ 'w' ][0];
      return t / (13 * 13);
    };
    expect(dominance(2)).toBeLessThan(dominance(12));
    expect(dominance(12)).toBeLessThan(dominance(24));
  });
});

describe('degenerate maps do not explode', () => {
  it('an empty map yields nothing', () => {
    expect(blendWeights([], [0.5, 0.5])).toEqual({ idx: [], w: [] });
    expect(paramsAtXY({ ...ROLE, points: [], snapshots: [] }, [0.5, 0.5])).toEqual([]);
  });

  it('a one-point map is that point everywhere', () => {
    const solo = {
      ...ROLE,
      points: [ROLE.points[0]],
      snapshots: [ROLE.snapshots[0]],
    };
    for (const xy of [[0, 0], [1, 1], [0.4, 0.6]] as XY[]) {
      expect(paramsAtXY(solo, xy)).toEqual(ROLE.snapshots[0]);
    }
  });
});

describe('movedParamIndices', () => {
  it('keeps a parameter that moves anywhere, including where it reads zero', () => {
    expect(movedParamIndices([[0.2, 0.5], [0.5, 0.9], [0.8, 0.5]])).toEqual([0, 1]);
  });

  it('drops a parameter the map never touches', () => {
    expect(movedParamIndices([[0.2, 0.5], [0.5, 0.5], [0.8, 0.5]])).toEqual([0]);
  });
});

describe('confidence — how certain a position is', () => {
  const LENS = {
    points: POINTS.map((xy, i) => ({
      preset_id: `p${i}`, name: `p${i}`, fxp_path: `p${i}.fxp`, xy: xy as [number, number],
    })),
    sharpness: 12, neighbours: 4, snap: 0.9,
  };

  it('is total when standing on a checked patch', () => {
    for (const p of LENS.points) expect(confidenceAt(LENS, p.xy)).toBe(1);
  });

  it('falls where several anchors share the position', () => {
    const a = LENS.points[0].xy;
    const b = LENS.points[1].xy;
    const mid: [number, number] = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    expect(confidenceAt(LENS, mid)).toBeLessThan(1);
    expect(confidenceAt(LENS, mid)).toBeGreaterThan(0);
  });

  it('never leaves 0..1, anywhere on the surface', () => {
    for (let a = 0; a <= 10; a++) {
      for (let b = 0; b <= 10; b++) {
        const c = confidenceAt(LENS, [a / 10, b / 10]);
        expect(c).toBeGreaterThanOrEqual(0);
        expect(c).toBeLessThanOrEqual(1);
      }
    }
  });
});

describe('nearestPoint — where a released gesture lands', () => {
  const LENS = {
    points: POINTS.map((xy, i) => ({
      preset_id: `p${i}`, name: `p${i}`, fxp_path: `p${i}.fxp`, xy: xy as [number, number],
    })),
  };

  it('returns the point you are already on', () => {
    for (const p of LENS.points) expect(nearestPoint(LENS, p.xy)).toEqual(p.xy);
  });

  it('always lands on a CHECKED point, never in between', () => {
    for (let a = 0; a <= 8; a++) {
      for (let b = 0; b <= 8; b++) {
        const got = nearestPoint(LENS, [a / 8, b / 8]);
        expect(LENS.points.some((p) => p.xy[0] === got![0] && p.xy[1] === got![1])).toBe(true);
      }
    }
  });

  it('has nothing to land on when the map is empty', () => {
    expect(nearestPoint({ points: [] }, [0.5, 0.5])).toBeNull();
  });
});

describe('the confidence duck agrees with the lab', () => {
  /**
   * The panel plays low-confidence positions quieter, and the lab's safety gate
   * measures `rms * that factor` to judge what the listener actually hears. If
   * the two laws drift, the gate certifies a loudness nobody is exposed to —
   * or misses one they are. Same class of bug as the blend vectors above.
   */
  const { levelForConfidence } = require('../TimbreGraphPanel');

  it('matches the python law at every sampled confidence', () => {
    for (const c of vectors.level_cases as Array<{ c: number; level: number }>) {
      expect(levelForConfidence(c.c)).toBeCloseTo(c.level, 9);
    }
  });

  it('uses the same floor the lab was told about', () => {
    expect(levelForConfidence(0)).toBeCloseTo(vectors.confidence_floor as number, 9);
  });

  it('leaves a certain position at full level', () => {
    expect(levelForConfidence(1)).toBe(1);
  });

  it('clamps nonsense rather than amplifying', () => {
    expect(levelForConfidence(5)).toBe(1);
    expect(levelForConfidence(-2)).toBeCloseTo(vectors.confidence_floor as number, 9);
  });
});
