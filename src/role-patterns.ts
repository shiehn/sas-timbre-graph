/**
 * Deterministic per-role MIDI patterns — what "Generate" writes.
 *
 * No LLM anywhere: this plugin explores SOUND, so its MIDI exists only to make
 * the six synths audible while the dial morphs them. The patterns are the
 * training lab's role probes (training/src/timbre_graph_lab/probes.py) ported
 * note-for-note, so the panel plays each patch the same way it was measured —
 * what you hear morphing is what the lab verified.
 *
 * Notes are in quarter-note beats; each pattern spans 4 beats and is tiled
 * across the scene. Tiling respects the scene meter via beats-per-bar.
 */

import type { PluginMidiNote } from '@signalsandsorcery/plugin-sdk';

export type TimbreRole = 'kick' | 'snare' | 'hat' | 'bass' | 'pad' | 'lead';

export const TIMBRE_ROLES: readonly TimbreRole[] = [
  'kick', 'snare', 'hat', 'bass', 'pad', 'lead',
];

/**
 * Canonical app role token per timbre role — from the platform's
 * instrument-classification taxonomy (plural percussion tokens, singular
 * 'lead'/'bass'). Stamped on tracks at creation so the deterministic
 * generation derives each track's pattern from its role with no typed prompt.
 */
export const APP_ROLE_TOKENS: Record<TimbreRole, string> = {
  kick: 'kicks',
  snare: 'snares',
  hat: 'hats',
  bass: 'bass',
  pad: 'pads',
  lead: 'lead',
};

/** (pitch, onsetBeat, lengthBeats, velocity) — mirrors probes.py `_events`. */
type N = [number, number, number, number];

const PATTERNS: Record<TimbreRole, N[]> = {
  // four-on-the-floor with a velocity dip and one syncopated extra
  kick: [
    [36, 0.0, 0.15, 118], [36, 1.0, 0.15, 96], [36, 2.0, 0.15, 118],
    [36, 2.75, 0.15, 80], [36, 3.0, 0.15, 110],
  ],
  // backbeat pair + ghosts
  snare: [
    [50, 1.0, 0.12, 116], [50, 1.75, 0.12, 62], [50, 3.0, 0.12, 118],
    [50, 3.75, 0.12, 90],
  ],
  // eighths with accents, longer final hit exposes decay
  hat: [
    [66, 0.0, 0.1, 110], [66, 0.5, 0.1, 72], [66, 1.0, 0.1, 110],
    [66, 1.5, 0.1, 72], [66, 2.0, 0.1, 110], [66, 2.5, 0.1, 72],
    [66, 3.0, 0.1, 110], [66, 3.5, 0.4, 100],
  ],
  // sustained, staccato, repeated, octave
  bass: [
    [36, 0.0, 0.9, 108], [36, 1.0, 0.2, 108], [36, 1.5, 0.2, 92],
    [43, 2.0, 0.9, 108], [48, 3.0, 0.9, 100],
  ],
  // two chords with overlapping release
  pad: [
    [48, 0.0, 2.1, 96], [52, 0.0, 2.1, 96], [55, 0.0, 2.1, 96],
    [53, 2.0, 2.0, 96], [57, 2.0, 2.0, 96], [60, 2.0, 2.0, 96],
  ],
  // phrase spanning ~octave+fifth, mixed articulation
  lead: [
    [60, 0.0, 0.45, 110], [64, 0.5, 0.45, 100], [67, 1.0, 0.9, 112],
    [72, 2.0, 0.2, 104], [67, 2.5, 0.2, 92], [79, 3.0, 0.9, 116],
  ],
};

const PATTERN_BEATS = 4;

/** Map an app track role (plural tokens: 'kicks', 'pads', …) to a timbre role. */
export function toTimbreRole(role: string | undefined): TimbreRole | null {
  const r = (role ?? '').toLowerCase();
  if (r.startsWith('kick')) return 'kick';
  if (r.startsWith('snare')) return 'snare';
  if (r.startsWith('hat') || r.startsWith('hihat') || r.startsWith('hi-hat')) return 'hat';
  if (r.startsWith('bass')) return 'bass';
  if (r.startsWith('pad') || r.startsWith('chord')) return 'pad';
  if (r.startsWith('lead')) return 'lead';
  return null;
}

/**
 * The role's probe pattern tiled across a scene.
 *
 * Tiles restart at each 4-beat boundary and are clipped to the scene length,
 * so odd meters get a truncated final tile rather than notes past the loop.
 */
export function tiledPattern(
  role: TimbreRole,
  bars: number,
  beatsPerBar: number,
): PluginMidiNote[] {
  const totalBeats = Math.max(bars, 1) * Math.max(beatsPerBar, 1);
  const out: PluginMidiNote[] = [];
  for (let offset = 0; offset < totalBeats; offset += PATTERN_BEATS) {
    for (const [pitch, onset, len, velocity] of PATTERNS[role]) {
      const start = offset + onset;
      if (start >= totalBeats) continue;
      out.push({
        pitch,
        startBeat: start,
        durationBeats: Math.min(len, totalBeats - start),
        velocity,
      });
    }
  }
  return out;
}


// ---------------------------------------------------------------------------
// Seeded variation — Generate produces a DIFFERENT, role-true pattern each
// time. Deterministic per seed (testable); the adapter passes a fresh seed
// per click. No LLM: variation is structural (subdivision, ghosts, contour,
// voicing), never role-identity (a kick stays a kick at C1).
// ---------------------------------------------------------------------------

/** mulberry32 — tiny deterministic PRNG. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const pick = <T,>(r: () => number, xs: readonly T[]): T =>
  xs[Math.floor(r() * xs.length)];

/** One 4-beat bar of a role, varied by the seed. */
function variedBar(role: TimbreRole, r: () => number): N[] {
  switch (role) {
    case 'kick': {
      const out: N[] = [];
      for (let b = 0; b < 4; b++) {
        if (b === 2 && r() < 0.2) continue;               // occasional dropped 3
        out.push([36, b, 0.15, 100 + Math.floor(r() * 22)]);
      }
      if (r() < 0.6) out.push([36, pick(r, [1.5, 2.5, 2.75, 3.5]), 0.15, 70 + Math.floor(r() * 25)]);
      return out;
    }
    case 'snare': {
      const out: N[] = [[50, 1, 0.12, 110 + Math.floor(r() * 12)],
                        [50, 3, 0.12, 110 + Math.floor(r() * 12)]];
      const ghosts = Math.floor(r() * 3);
      for (let i = 0; i < ghosts; i++) {
        out.push([50, pick(r, [0.5, 1.75, 2.25, 2.5, 3.75]), 0.1, 45 + Math.floor(r() * 30)]);
      }
      return out;
    }
    case 'hat': {
      const step = pick(r, [0.5, 0.5, 0.25]);             // 8ths twice as likely
      const out: N[] = [];
      for (let t = 0; t < 4 - 1e-9; t += step) {
        if (r() < 0.12) continue;                          // holes breathe
        const accent = Math.abs(t % 1) < 1e-9;
        out.push([66, t, step < 0.5 ? 0.08 : 0.1, (accent ? 100 : 64) + Math.floor(r() * 16)]);
      }
      if (out.length > 0 && r() < 0.5) out[out.length - 1][2] = 0.4; // open-ish last
      return out;
    }
    case 'bass': {
      const degrees = pick(r, [[0, 0, 7, 12], [0, 12, 0, 7], [0, 7, 5, 0], [0, 0, 0, 12], [0, 3, 5, 7]] as const);
      const rhythm = pick(r, [[0, 1, 2, 3], [0, 1.5, 2, 3.5], [0, 0.75, 2, 2.75]] as const);
      return rhythm.map((t, i) => [36 + degrees[i % degrees.length], t,
        pick(r, [0.2, 0.4, 0.9]), 92 + Math.floor(r() * 20)] as N);
    }
    case 'pad': {
      const root = pick(r, [48, 48, 45, 50]);
      const chord = pick(r, [[0, 4, 7], [0, 3, 7], [0, 5, 7], [0, 4, 7, 11]] as const);
      const change = r() < 0.5 ? 2 : 4;                    // move mid-bar or hold
      const out: N[] = [];
      for (let t = 0; t < 4; t += change) {
        const shift = t > 0 ? pick(r, [0, 2, 5, -2]) : 0;
        for (const iv of chord) out.push([root + shift + iv, t, change + 0.1, 90 + Math.floor(r() * 12)]);
      }
      return out;
    }
    case 'lead': {
      const pool = [60, 62, 64, 67, 69, 72, 74, 76, 79];
      const rhythm = pick(r, [[0, 0.5, 1, 2, 2.5, 3], [0, 1, 1.5, 2, 3], [0, 0.75, 1.5, 2.25, 3]] as const);
      let idx = Math.floor(r() * pool.length);
      return rhythm.map((t, i) => {
        idx = Math.max(0, Math.min(pool.length - 1, idx + pick(r, [-2, -1, 1, 2, i === rhythm.length - 1 ? 3 : 1])));
        return [pool[idx], t, pick(r, [0.3, 0.45, 0.9]), 96 + Math.floor(r() * 24)] as N;
      });
    }
  }
}

/** A seeded, varied pattern tiled across the scene (fresh bar every 4 beats). */
export function variedPattern(
  role: TimbreRole,
  bars: number,
  beatsPerBar: number,
  seed: number,
): PluginMidiNote[] {
  const totalBeats = Math.max(bars, 1) * Math.max(beatsPerBar, 1);
  const r = rng(seed);
  const out: PluginMidiNote[] = [];
  for (let offset = 0; offset < totalBeats; offset += PATTERN_BEATS) {
    for (const [pitch, onset, len, velocity] of variedBar(role, r)) {
      const start = offset + onset;
      if (start >= totalBeats) continue;
      out.push({
        pitch,
        startBeat: start,
        durationBeats: Math.min(len, totalBeats - start),
        velocity: Math.max(1, Math.min(127, velocity)),
      });
    }
  }
  return out;
}
