/**
 * GeneratorPanelAdapter for the Timbre Graph family.
 *
 * Everything shared rides panel-core: track rows, mixer strip, sound drawer +
 * history, shuffle cycling, event wiring, render skeleton. This adapter
 * supplies only what genuinely differs here:
 *
 * - generation goes through the SAME machinery as every other panel
 *   (`host.generateWithLLM` with the scene's key/chords/siblings), but the
 *   PROMPT IS DERIVED FROM THE ROLE rather than typed: a role is the request.
 *   The training lab's probe pattern stays as the offline fallback, so a
 *   failed or unreachable model still leaves audible, role-true MIDI.
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
import {
  createSurgeSoundAdapter,
  formatConcurrentTracks,
  formatMusicalContext,
  parseLLMNoteResponse,
  panelClipEndSeconds,
  panelMeter,
  panelQuarterNotesPerBar,
} from '@signalsandsorcery/plugin-sdk';
import {
  APP_ROLE_TOKENS,
  TIMBRE_ROLES,
  toTimbreRole,
  variedPattern,
} from './role-patterns';
import {
  buildTimbreSystemPrompt,
  constrainNotes,
  isPitched,
  roleUserPrompt,
} from './timbre-prompts';
import {
  TIMBRE_GROUP_META_KEY,
  timbreGroupIsComplete,
  timbreGroupSpec,
} from './timbre-group-meta';
import { TimbreGroupRow } from './TimbreGroupRow';
import type { TimbreGroupMeta } from './timbre-group-meta';
import { BUNDLED_GRAPH } from './bundled-graph';

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
      // one LLM call per role, same order as the sibling panels
      estimatedGenerationMs: 15000,
    },
    features: {
      // The ROLE is the prompt, so rows carry no prompt text. Without this
      // the core's generate handler gates on a non-empty prompt and silently
      // no-ops (observed live: Generate All did nothing). Generation itself
      // still goes through the LLM — this only waives the text box.
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

      /**
       * Restore the shipped graph's measured anchor patch for a role, so the
       * group starts on the exact sounds the morph was verified against.
       * Non-fatal: a machine without that library patch keeps the default
       * Surge sound and the morph still applies as relative moves.
       */
      const applyAnchor = async (trackId: string, role: string): Promise<void> => {
        const fxp = BUNDLED_GRAPH.roles[role]?.fxp_path;
        if (!fxp || typeof host.applySurgeFxpPreset !== 'function') return;
        try {
          await host.applySurgeFxpPreset(trackId, fxp);
        } catch {
          /* library patch absent on this machine — keep the default patch */
        }
      };

      await host.setTrackRole(handle.id, APP_ROLE_TOKENS.kick);
      await host.setSceneData(
        ctx.activeSceneId,
        ctx.trackDataKey(handle.dbId, TIMBRE_GROUP_META_KEY),
        { groupId, memberIndex: 0, role: 'kick' },
      );
      await applyAnchor(handle.id, 'kick');

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
        await applyAnchor(sibling.id, role);
      }
    },

    /**
     * Used by the transition path (the main generate() derives its own
     * per-role prompt). No role is supplied here, so this covers the family.
     */
    buildSystemPrompt(_validRoles: readonly string[], timeSignature?: string): string {
      return buildTimbreSystemPrompt('lead', timeSignature ?? '4/4');
    },

    parseNotesResponse(content: string): LLMNoteResponse | null {
      return parseLLMNoteResponse(content);
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
       * The same generation machinery every other panel uses — scene harmony
       * context, `host.generateWithLLM`, deterministic validation — with the
       * prompt DERIVED from the track's role instead of typed by the user.
       *
       * A role is the request: "kicks" means write a kick part. What the user
       * shapes here is timbre (the dial), so a prompt box would be a second,
       * redundant control. But the notes still have to fit the song, which
       * means the model needs the key, the chords and the siblings — exactly
       * what the bass and pad panels send.
       *
       * The offline probe pattern remains as the fallback: if the model is
       * unreachable or returns nothing usable, the role still gets audible,
       * role-true MIDI rather than an empty track.
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
        const bars = mc.bars > 0 ? mc.bars : 4;
        const bpm = mc.bpm > 0 ? mc.bpm : 120;
        const meter = panelMeter(mc);
        const beatsPerBar = panelQuarterNotesPerBar(mc);
        const totalBeats = bars * beatsPerBar;
        const endTime = panelClipEndSeconds(mc);

        // Sibling context: the other five roles of this group are being
        // written in the same pass, so what matters is the rest of the scene.
        let concurrentBlock = '';
        try {
          const genCtx = await services.host.getGenerationContext(track.handle.id);
          concurrentBlock = formatConcurrentTracks(genCtx);
        } catch {
          /* sibling context is best-effort, never a gate */
        }

        const userPrompt = [
          concurrentBlock,
          concurrentBlock ? '' : null,
          // Percussion has no harmony to follow; chords would be noise.
          formatMusicalContext(mc, { includeChords: isPitched(role) }),
          '',
          `Write the ${roleUserPrompt(role)}.`,
          `The clip is ${totalBeats} quarter-note beats long. Output the JSON.`,
        ]
          .filter((l) => l !== null)
          .join('\n');

        let notes = variedPattern(
          role, bars, beatsPerBar, Math.floor(Math.random() * 0xffffffff),
        );
        try {
          const llm = await services.host.generateWithLLM({
            system: buildTimbreSystemPrompt(role, meter),
            user: userPrompt,
            responseFormat: 'json',
          });
          const parsed = parseLLMNoteResponse(llm.content);
          const constrained = parsed
            ? constrainNotes(parsed.notes, role, totalBeats)
            : [];
          if (constrained.length > 0) {
            notes = constrained;
          } else {
            console.warn(
              `[TimbreGraphPanel] ${role}: model returned no usable notes — using the probe pattern`,
            );
          }
        } catch (err) {
          // An offline or failing model must not leave a silent track.
          console.warn(
            `[TimbreGraphPanel] ${role}: generation failed (${
              err instanceof Error ? err.message : String(err)
            }) — using the probe pattern`,
          );
        }

        await services.host.writeMidiClip(track.handle.id, {
          startTime: 0,
          endTime,
          tempo: bpm,
          notes,
        });
        services.updateTrack(track.handle.id, { hasMidi: true });
      },
    },
  };
}
