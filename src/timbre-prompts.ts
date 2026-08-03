/**
 * Prompts for the six roles — derived, never typed.
 *
 * The panel has no prompt box: a role IS the request. "kicks" means write a
 * kick part, and the user's creative control is the morph dial, not a text
 * field. So each role supplies its own user prompt and its own system prompt,
 * and generation otherwise goes through exactly the same machinery as every
 * other panel (`host.generateWithLLM` + scene harmony context).
 *
 * PITCH IS PART OF THE SOUND HERE. These are six Surge synths, not samplers.
 * The lab measured every patch at a specific pitch, and the morph's verified
 * behaviour belongs to the patch AS PLAYED THERE — a kick patch is tuned at
 * MIDI 36 and stops being a kick two octaves up. So the percussion roles pin
 * pitch and let the model write only rhythm and dynamics, while the pitched
 * roles follow the scene's chords inside the register the patch was measured
 * in.
 */

import type { PluginMidiNote } from '@signalsandsorcery/plugin-sdk';
import type { TimbreRole } from './role-patterns';

/** The measured pitch for the unpitched roles — see PATTERNS in role-patterns. */
export const FIXED_PITCH: Partial<Record<TimbreRole, number>> = {
  kick: 36,
  snare: 50,
  hat: 66,
};

/** Register each pitched patch was measured in: [lowest, highest] MIDI. */
export const REGISTER: Partial<Record<TimbreRole, [number, number]>> = {
  bass: [28, 50],
  pad: [45, 64],
  lead: [55, 84],
};

/** The request a role makes on the user's behalf. */
export const ROLE_REQUEST: Record<TimbreRole, string> = {
  kick: 'kick drum — the low anchor of the groove',
  snare: 'snare — backbeat and ghost notes',
  hat: 'hi-hat — the subdivision that carries the momentum',
  bass: 'bass line — locked to the kick, following the chord roots',
  pad: 'sustained chord pad voicing the progression',
  lead: 'lead melody over the progression',
};

export function isPitched(role: TimbreRole): boolean {
  return REGISTER[role] !== undefined;
}

/**
 * The user-side prompt for a role. Also written to `tracks.prompt` at
 * creation, so sibling panels reading generation context see this track's
 * intent the same way they see a hand-typed one.
 */
export function roleUserPrompt(role: TimbreRole): string {
  return ROLE_REQUEST[role];
}

export function buildTimbreSystemPrompt(role: TimbreRole, meter = '4/4'): string {
  const pitched = isPitched(role);
  const fixed = FIXED_PITCH[role];
  const register = REGISTER[role];

  const pitchRule = pitched
    ? [
        `PITCH: this patch was voiced for MIDI ${register![0]}-${register![1]}. Stay inside`,
        `that range — outside it the patch stops sounding like a ${role}.`,
        `Follow the chord progression: every sustained note must belong to the`,
        `chord under it, and land chord changes on their beat.`,
      ].join(' ')
    : [
        `PITCH: emit EVERY note at MIDI pitch ${fixed} exactly. This is a tuned`,
        `percussion patch, not a sampler — it only sounds correct at that pitch.`,
        `Write the RHYTHM and the VELOCITIES; the pitch is fixed.`,
      ].join(' ');

  return [
    `You write one ${role} part for an electronic production.`,
    ``,
    pitchRule,
    ``,
    `TIMING: positions and durations are in quarter-note beats from clip start`,
    `(0 = first beat). The scene is in ${meter}. Never write past the clip end`,
    `given in the context block.`,
    ``,
    `VELOCITY: 1-127. Use dynamics — accents, ghost notes — rather than one`,
    `flat value; a part with a single velocity sounds like a machine.`,
    ``,
    `Return ONLY JSON, no prose:`,
    `{"notes":[{"pitch":<int>,"startBeat":<number>,"durationBeats":<number>,"velocity":<int>}]}`,
  ].join('\n');
}

/**
 * Deterministic guard rails applied to whatever the model returned: drop
 * notes outside the clip, trim overhang, pin percussion pitch, and fold
 * pitched notes into the measured register.
 *
 * The morph's verified behaviour is a property of the patch as the lab played
 * it, so this is not cosmetic tidying — a lead an octave too high or a kick
 * transposed off 36 is a different instrument than the one measured.
 */
export function constrainNotes(
  notes: PluginMidiNote[],
  role: TimbreRole,
  totalBeats: number,
): PluginMidiNote[] {
  const fixed = FIXED_PITCH[role];
  const register = REGISTER[role];

  // Keep only what fits inside the clip, trimming overhang.
  const inClip: PluginMidiNote[] = [];
  for (const n of notes) {
    if (n.startBeat >= totalBeats || n.startBeat < 0) continue;
    const durationBeats = Math.min(n.durationBeats, totalBeats - n.startBeat);
    if (durationBeats <= 0) continue;
    inClip.push({ ...n, durationBeats });
  }

  if (fixed !== undefined) {
    return inClip
      .map((n) => ({ ...n, pitch: fixed }))
      .sort((a, b) => a.startBeat - b.startBeat);
  }
  if (!register || inClip.length === 0) {
    return inClip.sort((a, b) => a.startBeat - b.startBeat);
  }

  /*
   * Transpose the PHRASE by whole octaves, not each note independently.
   * Per-note folding inverts intervals — a rising minor third from 60 to 63
   * became 48 then 39, a descending sixth — which destroys the melodic shape
   * the model was asked for. One shared octave shift moves the part into the
   * patch's register while every interval survives intact.
   */
  const [lo, hi] = register;
  const mean = inClip.reduce((a, n) => a + n.pitch, 0) / inClip.length;
  const shift = Math.round(((lo + hi) / 2 - mean) / 12) * 12;

  const out = inClip.map((n) => {
    let pitch = n.pitch + shift;
    // A phrase wider than the register still has stragglers; pull only those
    // in, by octaves, so at worst a single outlier changes octave.
    while (pitch < lo) pitch += 12;
    while (pitch > hi) pitch -= 12;
    if (pitch < lo) pitch = lo;
    return { ...n, pitch };
  });

  return out.sort((a, b) => a.startBeat - b.startBeat);
}
