/**
 * GeneratorPanelAdapter for the Timbre Graph family.
 *
 * Everything shared rides panel-core: track rows, mixer strip, sound drawer +
 * history, shuffle cycling, event wiring, render skeleton. This adapter
 * supplies only what genuinely differs here:
 *
 * - generation is DETERMINISTIC — no LLM. "Generate" writes the role's probe
 *   pattern (the same MIDI the training lab measured the patch with), so the
 *   panel is audible the moment tracks exist and the dial has something to
 *   morph.
 * - sounds are Surge states via the SDK's shared Surge sound adapter, and 🎲
 *   is the host's preset shuffle — both reused, not reimplemented.
 */

import { createElement } from 'react';
import type {
  GeneratorPanelAdapter,
  GenerationServices,
  GeneratorTrackState,
  LLMNoteResponse,
  PluginHost,
  PluginTrackHandle,
  TrackCreatedContext,
} from '@signalsandsorcery/plugin-sdk';
import { createSurgeSoundAdapter } from '@signalsandsorcery/plugin-sdk';
import {
  APP_ROLE_TOKENS,
  TIMBRE_ROLES,
  tiledPattern,
  toTimbreRole,
} from './role-patterns';
import {
  TIMBRE_GROUP_META_KEY,
  timbreGroupIsComplete,
  timbreGroupSpec,
} from './timbre-group-meta';
import { TimbreGroupRow } from './TimbreGroupRow';
import type { TimbreGroupMeta } from './timbre-group-meta';

const ACCENT = '#2DD4BF'; // teal — distinct from synth violet / bass amber

export function createTimbreGraphAdapter(
  host: PluginHost,
): GeneratorPanelAdapter<TimbreGroupMeta> {
  return {
    identity: {
      familyKey: 'timbre-graph',
      familyLabel: 'Timbre Graph',
      trackNamePrefix: 'timbre',
      // The unit of creation here is the GROUP — one click makes six tracks.
      addTrackLabel: 'Add Graph',
      logTag: 'TimbreGraphPanel',
      accentColor: ACCENT,
      transitionAccentColor: ACCENT,
      placeholderAccentColor: ACCENT,
      // six fixed roles + headroom for copies/experiments (host caps at 16)
      maxTracks: 12,
      // deterministic pattern write — near-instant; keeps the bar honest
      estimatedGenerationMs: 800,
    },
    features: {
      // Role-derived deterministic generation: no prompt, no LLM, no auth
      // gate. Without this the core's generate handler silently no-ops on
      // promptless rows (observed live: Generate All did nothing).
      promptlessGeneration: true,
      instrumentPicker: false,
      bulkComposePlaceholders: false,
      exportMidi: true,
      transitionDesigner: false,
      importTracks: false,
    },

    createTrackOptions() {
      return { loadSynth: true, synthName: 'Surge XT' };
    },

    async applyPortedTrackSound(_handle: PluginTrackHandle): Promise<void> {
      // importTracks is off; nothing to do.
    },

    /**
     * "Add" means ADD A GROUP here. The core's add button creates ONE track
     * and hands it to this hook — that track becomes the group's kick
     * anchor, and the remaining five roles are created and stamped alongside
     * it. The core reloads tracks afterwards, so the whole six-row group
     * appears at once. Failures are non-fatal per the contract: whatever was
     * created lands as plain rows and delete works normally.
     */
    async onTrackCreated(
      handle: PluginTrackHandle,
      ctx: TrackCreatedContext,
    ): Promise<void> {
      const groupId = handle.dbId;
      await host.setTrackRole(handle.id, APP_ROLE_TOKENS.kick);
      await host.setSceneData(
        ctx.activeSceneId,
        ctx.trackDataKey(handle.dbId, TIMBRE_GROUP_META_KEY),
        { groupId, memberIndex: 0, role: 'kick' },
      );
      for (let i = 1; i < TIMBRE_ROLES.length; i++) {
        const role = TIMBRE_ROLES[i];
        const sibling = await host.createTrack({
          name: `timbre-${role}`,
          role: APP_ROLE_TOKENS[role],
          loadSynth: true,
          synthName: 'Surge XT',
        });
        await host.setSceneData(
          ctx.activeSceneId,
          ctx.trackDataKey(sibling.dbId, TIMBRE_GROUP_META_KEY),
          { groupId, memberIndex: i, role },
        );
      }
    },

    buildSystemPrompt(): string {
      // No LLM path in this family; required by the contract, never sent.
      return 'unused';
    },

    parseNotesResponse(): LLMNoteResponse | null {
      return null;
    },

    sound: createSurgeSoundAdapter(host),

    shuffle: {
      async shuffle(track: GeneratorTrackState, excludeNames: string[]) {
        const result = await host.shufflePreset(track.handle.id, excludeNames);
        return { appliedName: result.presetName ?? 'preset' };
      },
      isExhaustedError(err: unknown): boolean {
        return err instanceof Error && /no presets? available/i.test(err.message);
      },
    },

    groupExtensions: [
      {
        ...timbreGroupSpec,
        isComplete: timbreGroupIsComplete,
        renderGroup: (group, ctx) =>
          createElement(TimbreGroupRow, { group, ctx }),
      },
    ],

    generation: {
      /**
       * Deterministic: write the role's probe pattern across the scene.
       * The dial changes SOUND; this MIDI only makes the sound audible, and
       * matches how the lab measured each patch.
       */
      async generate(
        track: GeneratorTrackState,
        services: GenerationServices,
      ): Promise<void> {
        const role = toTimbreRole(track.role) ?? toTimbreRole(track.handle.role);
        if (!role) {
          throw new Error(
            `Track "${track.handle.name}" has no timbre role — expected one of kick/snare/hat/bass/pad/lead`,
          );
        }
        const mc = await services.host.getMusicalContext();
        const bars = mc.bars ?? 4;
        const bpm = mc.bpm ?? 120;
        const [num, den] = (mc.timeSignature ?? '4/4')
          .split('/')
          .map((v: string) => parseInt(v, 10));
        const beatsPerBar =
          Number.isFinite(num) && Number.isFinite(den) && den > 0
            ? (num * 4) / den
            : 4;
        const totalBeats = bars * beatsPerBar;
        const endTime = (totalBeats * 60) / bpm;

        // Progress steps so the row's bar reflects real phases rather than
        // only estimate pacing: context read -> pattern write -> done.
        services.updateTrack(track.handle.id, { generationProgress: 0.3 });
        await services.host.writeMidiClip(track.handle.id, {
          startTime: 0,
          endTime,
          tempo: bpm,
          notes: tiledPattern(role, bars, beatsPerBar),
        });
        services.updateTrack(track.handle.id, {
          hasMidi: true,
          generationProgress: 1,
        });
      },
    },
  };
}
