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
