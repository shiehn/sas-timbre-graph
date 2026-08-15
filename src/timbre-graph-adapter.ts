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
 *   A failed model FAILS — it does not quietly substitute the training lab's
 *   probe pattern, which only ever made a broken generation sound like a
 *   working one.
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
  parseTrackGroups,
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
} from './role-patterns';
import {
  buildTimbreSystemPrompt,
  constrainNotes,
  isPitched,
  roleUserPrompt,
} from './timbre-prompts';
import {
  TIMBRE_GROUP_CONFIG_KEY,
  TIMBRE_GROUP_META_KEY,
  asTimbreGroupConfig,
  timbreGroupIsComplete,
  timbreGroupSpec,
} from './timbre-group-meta';
import type { TimbreGroupMode } from './timbre-group-meta';
import type { TimbreRole } from './role-patterns';
import { TimbreGroupRow } from './TimbreGroupRow';
import type { TimbreGroupMeta } from './timbre-group-meta';
import { BUNDLED_GRAPH } from './bundled-graph';

const ACCENT = '#2DD4BF'; // teal — distinct from synth violet / bass amber

/**
 * Arm the host-managed brickwall limiter on a track. EAR SAFETY, not tone.
 *
 * The dial writes ~80 continuous parameters at once, and some combinations
 * put a Surge filter into self-oscillation. Two of them "started screaming"
 * in live play and hurt the user (2026-08-03). The anchors are
 * loudness-normalized at build time so this should almost never engage — it
 * is the last line, for the space BETWEEN anchors that no build gate can
 * enumerate.
 *
 * SDK 3.0.0 removed the built-in FX surface; the limiter is now armed by
 * INTENT (`applyManagedFxPreset(id, 'safety-limiter')`) and the host resolves
 * the actual preset internally — no more cross-repo preset-index contract to
 * drift. The host-side apply is idempotent, so callers re-arm freely.
 *
 * Every failure path warns and continues: a missing limiter must never block
 * track creation or adoption, but it must be visible in the log.
 */
export async function armSafetyLimiter(
  host: PluginHost,
  trackId: string,
): Promise<void> {
  if (typeof host.applyManagedFxPreset !== 'function') {
    console.warn(
      `[TimbreGraphPanel] host cannot arm the safety limiter on ${trackId} ` +
        `(needs SDK 3.0.0 applyManagedFxPreset) — the dial can still reach ` +
        `painful levels`,
    );
    return;
  }
  try {
    await host.applyManagedFxPreset(trackId, 'safety-limiter');
  } catch (err) {
    console.warn(
      `[TimbreGraphPanel] could not arm the safety limiter on ${trackId} — ` +
        `the dial can still reach painful levels`,
      err,
    );
  }
}

/**
 * One re-arm pass over the tracks a discovery/adoption cycle handed back.
 *
 * Arming only at creation is not enough: the host STRIPS all built-in FX —
 * this limiter included — from the project file on every load, so a reopened
 * project came back unprotected. The panel therefore re-arms on every
 * adoption pass (see the `core.tracks` effect in TimbreGraphPanel).
 *
 * `armedHandles` is NOT dedup-for-correctness (the host-side apply is
 * idempotent); it is a PASS detector. panel-core's loadTracks rebuilds every
 * row around fresh handle objects from `getPluginTracks()` — on mount, on
 * scene change, on `onEngineReady` after a project load, and after agent
 * mutations — while in-place row patches (generation progress ticks,
 * mute/volume changes) spread the row but keep `t.handle` by reference. So
 * keying on handle identity fires exactly once per track per discovery pass
 * and never on render/state churn. Fire-and-forget: failures warn inside
 * armSafetyLimiter.
 */
export function rearmSafetyLimiters(
  host: PluginHost,
  tracks: ReadonlyArray<{ handle: { id: string } }>,
  armedHandles: WeakSet<object>,
): void {
  for (const t of tracks) {
    if (armedHandles.has(t.handle)) continue;
    armedHandles.add(t.handle);
    void armSafetyLimiter(host, t.handle.id);
  }
}

/**
 * Per-group mode, cached so the row can render synchronously.
 *
 * The config lives in scene data (the durable copy); this map is only what the
 * last read or write put there, so a fresh open shows `ensemble` until the
 * panel reads back — which is the safe default, since it is what the tracks'
 * roles already say.
 */
const groupConfigCache = new Map<
  string,
  { mode: TimbreGroupMode; layerRole: TimbreRole }
>();

function readGroupConfig(groupId: string): {
  mode: TimbreGroupMode;
  layerRole: TimbreRole;
} {
  return groupConfigCache.get(groupId) ?? { mode: 'ensemble', layerRole: 'bass' };
}

/**
 * Restore every group's mode from scene data.
 *
 * Without this the cache is write-only and the durable copy is never read, so
 * reopening a project showed a layered group as `ensemble`: the pad still drove
 * each track through its own lens (that comes from the members' own meta), but
 * the dropdown lied and Generate would write six competing parts instead of
 * one. Called from the panel's scene-data effect, which already has the blob.
 */
export function primeGroupConfigs(sceneData: Record<string, unknown>): void {
  for (const [key, value] of Object.entries(sceneData ?? {})) {
    if (!key.endsWith(`:${TIMBRE_GROUP_CONFIG_KEY}`)) continue;
    const cfg = asTimbreGroupConfig(value);
    if (!cfg) continue;
    // key shape: group:<groupId>:timbreGroupConfig
    const parts = key.split(':');
    const groupId = parts.length >= 3 ? parts[1] : null;
    if (groupId) {
      groupConfigCache.set(groupId, { mode: cfg.mode, layerRole: cfg.role });
    }
  }
}

/**
 * Switch a group between ensemble and layered.
 *
 * Layered means every member plays the SAME role and the same part, differing
 * only in which lens they are heard through — so the switch rewrites each
 * member's role token and stamps a distinct `lensIndex`. Ensemble restores the
 * canonical one-role-per-member layout. MIDI is untouched either way: the user
 * presses Generate when they want parts.
 */
/**
 * Write the same clip to every other member of a LAYERED group.
 *
 * No-op for an ensemble group (the members play different roles and want
 * different parts) and best-effort throughout: a sibling that will not accept
 * the clip leaves a quiet layer, not a failed generation.
 */
async function copyPartToLayers(
  services: GenerationServices,
  sourceEngineId: string,
  clip: { startTime: number; endTime: number; tempo: number; notes: LLMNoteResponse['notes'] },
): Promise<void> {
  try {
    const sceneId = services.activeSceneId;
    if (!sceneId) return;
    const handles = await services.host.getPluginTracks();
    const source = handles.find((h) => h.id === sourceEngineId);
    if (!source) return;

    const sceneData = await services.host.getAllSceneData(sceneId);
    const groups = parseTrackGroups<TimbreGroupMeta>(sceneData, timbreGroupSpec);
    const group = groups.find((g) =>
      g.members.some((m) => m.dbId === source.dbId),
    );
    if (!group) return;
    if (readGroupConfig(group.groupId).mode !== 'layered') return;

    const byDbId = new Map(handles.map((h) => [h.dbId, h.id]));
    for (const m of group.members) {
      const id = byDbId.get(m.dbId);
      if (!id || id === sourceEngineId) continue;
      try {
        await services.host.writeMidiClip(id, clip);
        services.updateTrack(id, { hasMidi: true });
      } catch (err) {
        console.warn(`[TimbreGraphPanel] layer ${id} did not take the part`, err);
      }
    }
  } catch (err) {
    console.warn('[TimbreGraphPanel] could not copy the part across layers', err);
  }
}

async function applyGroupMode(
  host: PluginHost,
  group: { groupId: string; members: Array<{ dbId: string; track: { handle: { id: string } } }> },
  sceneId: string | null,
  mode: TimbreGroupMode,
  layerRole: TimbreRole,
): Promise<void> {
  groupConfigCache.set(group.groupId, { mode, layerRole });
  if (!sceneId) return;
  try {
    await host.setSceneData(sceneId, `group:${group.groupId}:${TIMBRE_GROUP_CONFIG_KEY}`,
      { mode, role: layerRole });
    for (let i = 0; i < group.members.length; i++) {
      const m = group.members[i];
      const role: TimbreRole = mode === 'layered'
        ? layerRole
        : TIMBRE_ROLES[Math.min(i, TIMBRE_ROLES.length - 1)];
      await host.setTrackRole(m.track.handle.id, APP_ROLE_TOKENS[role]);
      await host.setSceneData(
        sceneId,
        `track:${m.dbId}:${TIMBRE_GROUP_META_KEY}`,
        // in layered mode the lens is what distinguishes the members; in
        // ensemble they differ by role and all sit in world 0
        { groupId: group.groupId, memberIndex: i, role,
          lensIndex: mode === 'layered' ? i : 0 },
      );
    }
  } catch (err) {
    console.warn('[TimbreGraphPanel] could not switch group mode', err);
  }
}

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

      // Every newborn gets the ear-safety brickwall (armSafetyLimiter has
      // the full why). Creation-time arming is only half the story: the host
      // strips built-in FX on project load, so the panel ALSO re-arms on
      // every adoption pass (rearmSafetyLimiters).

      /**
       * Restore the shipped map's ORIGIN patch for a role, so the group
       * begins on the exact sound the tour was validated against — point 0
       * is also the structural lens every position is heard through.
       * Non-fatal: a machine without that library patch keeps the default
       * Surge sound and the map still applies as relative moves.
       */
      const applyAnchor = async (trackId: string, role: string): Promise<void> => {
        const fxp = BUNDLED_GRAPH.roles[role]?.lenses?.[0]?.points?.[0]?.fxp_path;
        // Silence here is expensive: without the start patch the dial applies
        // deltas measured on one sound to a completely different one, which
        // still moves parameters but never reaches the configurations the
        // tour promises — it reads as "the morph is weak" with nothing in the
        // log to explain why. So every miss is reported.
        if (!fxp) {
          console.warn(`[TimbreGraphPanel] ${role}: map has no origin patch`);
          return;
        }
        if (typeof host.applySurgeFxpPreset !== 'function') {
          console.warn(
            `[TimbreGraphPanel] ${role}: host cannot apply .fxp presets ` +
              `(needs SDK 2.58.0) — the dial will ride the default patch`,
          );
          return;
        }
        try {
          await host.applySurgeFxpPreset(trackId, fxp);
        } catch (err) {
          console.warn(
            `[TimbreGraphPanel] ${role}: start patch ${fxp} did not load — ` +
              `the dial will ride whatever sound this track has`,
            err,
          );
        }
      };

      await host.setTrackRole(handle.id, APP_ROLE_TOKENS.kick);
      await host.setSceneData(
        ctx.activeSceneId,
        ctx.trackDataKey(handle.dbId, TIMBRE_GROUP_META_KEY),
        { groupId, memberIndex: 0, role: 'kick' },
      );
      await applyAnchor(handle.id, 'kick');
      await armSafetyLimiter(host, handle.id);

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
        await armSafetyLimiter(host, sibling.id);
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
          createElement(TimbreGroupRow, {
            group,
            ctx,
            ...readGroupConfig(group.groupId),
            onModeChange: (mode: TimbreGroupMode, layerRole: TimbreRole) => {
              void applyGroupMode(
                host, group, ctx.services.activeSceneId ?? null, mode, layerRole,
              );
            },
          }),
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
       * Throws on failure and writes nothing. Silence with a visible error is
       * the honest outcome; audible filler that pretends to be a generation
       * is not.
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

        // No fallback pattern. There used to be one — the training lab's
        // probe pattern, pre-seeded so an offline model "never left a silent
        // track". In practice it made every failure INAUDIBLE as a failure:
        // the model fell over, six tracks instantly played probe MIDI, and
        // nothing anywhere said the LLM had not run. A failed generation must
        // read as a failed generation, exactly as it does on bass and pad, so
        // let the error propagate to the core's handler (toast + row error).
        // Fresh per press, so repeated Generates on the lead roll a different
        // arpeggio rate instead of returning the same feel every time.
        const shapeSeed = Math.floor(Math.random() * 0xffffffff);
        const llm = await services.host.generateWithLLM({
          system: buildTimbreSystemPrompt(role, meter, shapeSeed),
          user: userPrompt,
          responseFormat: 'json',
        });
        const parsed = parseLLMNoteResponse(llm.content);
        const notes = parsed ? constrainNotes(parsed.notes, role, totalBeats) : [];
        if (notes.length === 0) {
          throw new Error(
            `The model returned no usable notes for ${role}. Nothing was written — press Generate again.`,
          );
        }

        await services.host.writeMidiClip(track.handle.id, {
          startTime: 0,
          endTime,
          tempo: bpm,
          notes,
        });

        /*
         * LAYERED groups play ONE part on every layer.
         *
         * Six copies of a role are a stack, not an arrangement — six
         * independently generated bass lines would fight rather than thicken,
         * and would cost six LLM calls to produce the mess. So the part is
         * written once here and copied to the siblings; what differs between
         * them is the structure they are heard through, which is the whole
         * point of the mode.
         */
        await copyPartToLayers(services, track.handle.id, {
          startTime: 0, endTime, tempo: bpm, notes,
        });

        // Success patch. The core clears `isGenerating` on the ERROR path
        // only — on success that is the adapter's job (same contract bass,
        // pad, ensemble and arp follow). Without this the row's progress bar
        // never resolves, which is exactly how this shipped.
        services.updateTrack(track.handle.id, (t) => ({
          ...t,
          isGenerating: false,
          error: null,
          hasMidi: true,
          generationProgress: 0,
          editNotes: notes,
          editBars: bars,
          editBpm: bpm,
          editBeatsPerBar: beatsPerBar,
        }));
      },
    },
  };
}
