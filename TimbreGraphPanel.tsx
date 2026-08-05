/**
 * Timbre Graph panel — six coupled Surge XT tracks and one tour dial.
 *
 * Built on the SDK's panel-core: `useGeneratorPanelCore` + `GeneratorPanelShell`
 * own the track rows, mixer strip, sound drawer/history, shuffle, add-track
 * button and render phases — all shared with the other generator families.
 * This file contributes only what is unique here: the TOUR SECTION rendered
 * in the shell's `beforeRows` slot.
 *
 * The dial replays a precomputed tour (built by the training lab): a sequence
 * of ~20 screened, render-validated parameter configurations per synth, each
 * taken from a real factory patch. Runtime is linear interpolation between
 * the two snapshots either side of the dial — no model, no solver, no
 * latency. Per-track link toggles freeze a synth without touching the rest.
 *
 * Why a tour rather than an axis (docs/TRAINING.md C15): the previous dial
 * solved for the SMALLEST parameter change that moved one perceptual axis,
 * so from 0 to 100 the patch never stopped being itself. Anchors travel
 * between configurations instead, and the artifact carries their absolute
 * values — the panel's only job is to interpolate and send the difference.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type {
  GeneratorPanelAdapter,
  PluginHost,
  PluginUIProps,
} from '@signalsandsorcery/plugin-sdk';
import {
  GeneratorPanelShell,
  parseTrackGroups,
  useGeneratorPanelCore,
} from '@signalsandsorcery/plugin-sdk';
import { createTimbreGraphAdapter } from './src/timbre-graph-adapter';
import {
  APP_ROLE_TOKENS,
  TIMBRE_ROLES,
  toTimbreRole,
  type TimbreRole,
} from './src/role-patterns';
import {
  TIMBRE_GROUP_META_KEY,
  timbreGroupSpec,
  type TimbreGroupMeta,
} from './src/timbre-group-meta';
import { BUNDLED_GRAPH } from './src/bundled-graph';
import {
  confidenceAt,
  lensAt,
  movedParamIndices,
  nearestPoint,
  paramsAtXY,
  type MapGraph,
} from './src/patchmap';
import { XYPad } from './src/XYPad';

const ACCENT = '#2DD4BF'; // teal, matching the panel accent

const ROLE_LABELS: Record<TimbreRole, string> = {
  kick: 'Kick',
  snare: 'Snare',
  hat: 'Hat',
  bass: 'Bass',
  pad: 'Chord Pad',
  lead: 'Lead',
};

/**
 * Project-data key. Deliberately NOT the old 'timbre-graph.morph': a stored
 * single-anchor graph has no meaning here, and a fresh key makes every
 * existing project fall back to the shipped tour with no migration code.
 */
const GRAPH_KEY = 'timbre-graph.map';
const LINKS_KEY = 'timbre-graph.links';
const MAP_VERSION = 'map-graph-v2';
/** Where the pad starts, and what "home" means: point 0, the lens. */
const ORIGIN: [number, number] = [0, 0];
/**
 * Quietest a position plays when it is deep in hybrid territory, as a fraction
 * of full (0.15 ≈ -16 dB).
 *
 * This is the RUNTIME half of ear safety, and it is the half that actually
 * generalises. The build can only ever sample a continuous surface: pruning a
 * 9x9 grid still left a handful of random positions out of 640 measuring 4-11 dB
 * hot, because a blend of two individually-safe filter settings can resonate
 * where neither does. Sampling harder is a losing game.
 *
 * But those positions are precisely the LOW-CONFIDENCE ones — the hybrids. So
 * the rule "the less sure we are of a sound, the quieter we play it" converts
 * the residual risk from a scream into a quiet hybrid, without any sampling
 * claim at all. At sharpness 20 roughly three quarters of the surface has
 * confidence 1.0 and is therefore untouched by this.
 *
 * Was 0.75 (-2.5 dB), too shallow to cancel an 11 dB lurch; then 0.25.
 */
const CONFIDENCE_FLOOR = 0.15;

/** The tour section rendered above the standard track rows. Exported for tests. */
/**
 * How loud a position plays, given how certain it is. Exported so the lab's
 * safety gate can measure what the LISTENER hears rather than what the synth
 * emits — the two differ by exactly this factor.
 */
export function levelForConfidence(confidence: number): number {
  const c = Math.min(1, Math.max(0, confidence));
  return CONFIDENCE_FLOOR + (1 - CONFIDENCE_FLOOR) * c;
}

export function MorphSection({
  host,
  activeSceneId,
  resolveTrackIds,
  onTracksChanged,
}: {
  host: PluginHost;
  activeSceneId: string | null;
  /**
   * Role -> engine ids of the LIVE tracks to drive, read at apply time.
   * The graph's stamped track_id is only a fallback: groups get removed and
   * re-added, and a dial faithfully writing to deleted tracks is silent
   * (observed live — stamped 1235-1255 vs live 1464-1484).
   */
  resolveTrackIds: (role: string) => Array<{ id: string; lensIndex: number }>;
  onTracksChanged: () => void;
}) {
  // The plugin SHIPS with a measured tour; a project-stored one (from an
  // import) overrides it. The dial is therefore live on first open — the
  // training lab is for re-training, not a user prerequisite.
  const [graph, setGraph] = useState<MapGraph | null>(BUNDLED_GRAPH);
  const [imported, setImported] = useState(false);
  const [pos, setPos] = useState<[number, number]>(ORIGIN);
  const [hovered, setHovered] = useState<number | null>(null);
  /**
   * Which WORLD every layer is exploring.
   *
   * A lens is a structure the runtime cannot write — oscillator and filter
   * types, routing — so its map is a hard ceiling on what dragging can reach.
   * Changing it is the difference between hunting inside one sound and
   * starting somewhere else entirely. Shared across the six roles so the
   * ensemble moves together; clamped per role, since roles yield different
   * numbers of viable lenses.
   */
  const [world, setWorld] = useState(0);
  const [linked, setLinked] = useState<Record<string, boolean>>({});
  const [status, setStatus] = useState('');
  const [importProgress, setImportProgress] = useState<number | null>(null);
  const applying = useRef(false);
  const pending = useRef<[number, number] | null>(null);

  /**
   * The dots drawn on the pad.
   *
   * Taken from the role with the richest map: every role shares the same unit
   * square, so one set of coordinates addresses all six at once, and the
   * densest one gives the truest picture of where things are. The puck's
   * position means the same thing to every layer.
   */
  const padPoints = useMemo(() => {
    const lenses = Object.values(graph?.roles ?? {})
      .map((r) => lensAt(r, world))
      .filter((l): l is NonNullable<typeof l> => l !== null);
    const best = lenses.reduce(
      (a, b) => (b.points.length > (a?.points.length ?? 0) ? b : a),
      lenses[0],
    );
    return (best?.points ?? []).map((p) => ({ xy: p.xy, name: p.name }));
  }, [graph, world]);

  /** How many worlds are on offer — the most any role has. */
  const worldCount = useMemo(
    () => Math.max(1, ...Object.values(graph?.roles ?? {}).map((r) => r.lenses?.length ?? 0)),
    [graph],
  );

  /** Per role, the parameters the map touches anywhere. Computed once. */
  const movedByRole = useMemo(() => {
    const out: Record<string, number[]> = {};
    for (const [role, t] of Object.entries(graph?.roles ?? {})) {
      const l = lensAt(t, world);
      if (l) out[role] = movedParamIndices(l.snapshots);
    }
    return out;
  }, [graph, world]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const raw = await host.getProjectData?.<unknown>(GRAPH_KEY);
        if (!cancelled && raw) {
          const stored = typeof raw === 'string' ? JSON.parse(raw) : raw;
          if (stored && (stored as MapGraph).version === MAP_VERSION
            && (stored as MapGraph).roles) {
            setGraph(stored as MapGraph);
            setImported(true);
          }
        }
        const links = await host.getProjectData?.<unknown>(LINKS_KEY);
        if (!cancelled && links) {
          setLinked(
            typeof links === 'string'
              ? JSON.parse(links)
              : (links as Record<string, boolean>),
          );
        }
      } catch {
        /* empty state renders the import affordance */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [host]);

  /**
   * Import a morph-graph file and stand the instrument up: one owned track
   * per role, Surge loaded, the measured anchor preset restored from its
   * .fxp, and the engine track id stamped so every later parameter write
   * addresses the track by id (never by name). The role's probe MIDI comes
   * from the standard per-row Generate button (deterministic — see adapter).
   */
  const importGraphFile = useCallback(
    async (file: File) => {
      setStatus('Importing…');
      try {
        const text = await new Promise<string>((resolve, reject) => {
          const r = new FileReader();
          r.onload = () => resolve(String(r.result));
          r.onerror = () => reject(r.error ?? new Error('could not read file'));
          r.readAsText(file);
        });
        const parsed = JSON.parse(text) as MapGraph;
        if (parsed?.version !== MAP_VERSION || !parsed?.roles) {
          setStatus(`Not a patch map (need version ${MAP_VERSION}).`);
          return;
        }
        const roleList = Object.keys(parsed.roles);
        let groupId: string | null = null;
        for (let i = 0; i < roleList.length; i++) {
          const role = roleList[i];
          const t = parsed.roles[role];
          setStatus(`Setting up ${role}… (${i + 1}/${roleList.length})`);
          setImportProgress((i + 0.2) / roleList.length);
          const timbre = toTimbreRole(role);
          const handle = await host.createTrack({
            name: `timbre-${role}`,
            role: timbre ? APP_ROLE_TOKENS[timbre] : role,
            loadSynth: true,
            synthName: 'Surge XT',
          });
          groupId = groupId ?? handle.dbId;
          if (activeSceneId) {
            // same group seam the Add button uses — the six render as ONE group
            await host.setSceneData(
              activeSceneId,
              `track:${handle.dbId}:${TIMBRE_GROUP_META_KEY}`,
              { groupId, memberIndex: i, role: timbre ?? 'lead' },
            );
          }
          // anchor 0 is the tour's structural lens: every later position is
          // heard through this preset's oscillators and routing
          const startFxp = t.lenses?.[0]?.points?.[0]?.fxp_path;
          if (startFxp && typeof host.applySurgeFxpPreset === 'function') {
            try {
              await host.applySurgeFxpPreset(handle.id, startFxp);
            } catch (err) {
              // Missing on this machine: track keeps a default patch;
              // snapshots remain meaningful relative moves.
              console.warn('[TimbreGraph] preset restore failed', role, err);
            }
          }
          t.track_id = handle.id;
          setImportProgress((i + 1) / roleList.length);
        }
        await host.setProjectData?.(GRAPH_KEY, JSON.stringify(parsed));
        setGraph(parsed);
        setPos(ORIGIN);
        setStatus('');
        setImportProgress(null);
        onTracksChanged();
      } catch (err) {
        setStatus(err instanceof Error ? err.message : 'Import failed');
        setImportProgress(null);
      }
    },
    [host, activeSceneId, onTracksChanged],
  );

  /** Drop an imported graph and fall back to the shipped one. */
  const revertToBundled = useCallback(async () => {
    await host.setProjectData?.(GRAPH_KEY, '');
    setGraph(BUNDLED_GRAPH);
    setImported(false);
    setPos(ORIGIN);
  }, [host]);

  /** Coalesced apply: while a write is in flight the newest position wins. */
  const apply = useCallback(
    async (value: [number, number]) => {
      if (!graph) return;
      if (applying.current) {
        pending.current = value;
        return;
      }
      applying.current = true;
      // Per-role isolation. A single try around the whole loop meant one
      // role's failure aborted every role after it — and `lead` is last in
      // role order, so ANY earlier error made it look permanently dead.
      const failures: string[] = [];
      try {
        // At the start of the tour every delta is zero, so nothing is SENT
        // even though the tracks are right there. Reporting that as "no
        // tracks" would be a lie the user hits every time they press reset,
        // so the status keys off whether targets EXIST, not whether the dial
        // had anything to say.
        let hadTargets = false;
        for (const role of Object.keys(graph.roles)) {
          if (linked[role] === false) continue;
          const entry = graph.roles[role];
          if (!entry || entry.declined) continue;

          const live = resolveTrackIds(role);
          const targets = live.length > 0
            ? live
            : entry.track_id
              ? [{ id: entry.track_id, lensIndex: 0 }]
              : [];
          if (targets.length === 0) continue;
          hadTargets = true;

          /*
           * Grouped BY WORLD, not by role.
           *
           * A layered group is six tracks of one role that differ only in the
           * lens they are heard through, so one set of parameters per role
           * would make them six identical voices — the exact opposite of the
           * mode. Grouping keeps the arithmetic once per distinct world while
           * letting each member land somewhere structurally different.
           */
          const byWorld = new Map<number, string[]>();
          for (const tgt of targets) {
            const w = tgt.lensIndex + world;
            byWorld.set(w, [...(byWorld.get(w) ?? []), tgt.id]);
          }

          for (const [w, ids] of byWorld) {
            const t = lensAt(entry, w);
            if (!t) continue;
            // The move from this world's ORIGIN to the puck, applied RELATIVE
            // to each track's live parameters — so the map rides on whatever
            // sound the track currently has instead of snapping to absolutes.
            //
            // Deltas go out RAW: both ends are render-validated configurations,
            // every value is already in [0,1], and the host clamps to each
            // parameter's declared range, so scaling here could only stop the
            // puck reaching the patches the map promises.
            const values = paramsAtXY(t, value);
            const start = paramsAtXY(t, ORIGIN);

            // EVERY parameter this world moves, at every position — a zero
            // delta is a real instruction ("return this one to base"), not a
            // no-op to skip. See movedParamIndices.
            const params: Record<string, number> = {};
            for (const i of movedParamIndices(t.snapshots)) {
              params[t.param_names[i]] = (values[i] ?? 0) - (start[i] ?? 0);
            }
            if (Object.keys(params).length === 0) continue;
            if (typeof host.setSynthParameters !== 'function') {
              setStatus('This host predates SDK 2.57.0 — parameter writes unavailable.');
              return;
            }
            /*
             * Sit uncertain layers back in the mix.
             *
             * A position where one anchor dominates IS a checked patch; a
             * hybrid is inside the hull of checked configurations but has not
             * itself been rendered. Weighting level by that confidence makes
             * the stack lean on what is known without ever silencing the
             * exploration — and it is free, because the blend already computes
             * the number. Deliberately a narrow range: this is a lean, not a
             * duck, and it must not fight the panel's own mixer.
             */
            const level = levelForConfidence(confidenceAt(t, value));

            for (const id of ids) {
              try {
                await host.setSynthParameters(id, params, 0, { relative: true });
                if (typeof host.setTrackVolume === 'function') {
                  await host.setTrackVolume(id, level);
                }
              } catch (err) {
                // Log per failure so it is diagnosable in renderer-logs, and
                // keep going: the other layers must still move.
                failures.push(role);
                console.error(
                  `[TimbreGraph] map write failed role=${role} track=${id}`,
                  err,
                );
              }
            }
          }
        }
        if (!hadTargets) {
          setStatus('No live tracks to drive — Add Graph first, then move the dial.');
        } else if (failures.length > 0) {
          setStatus(`failed on: ${[...new Set(failures)].join(', ')}`);
        } else {
          setStatus('');
        }
      } catch (err) {
        setStatus(err instanceof Error ? err.message : 'could not apply parameters');
      } finally {
        applying.current = false;
        if (pending.current !== null) {
          const next = pending.current;
          pending.current = null;
          void apply(next);
        }
      }
    },
    [graph, host, linked, resolveTrackIds, movedByRole, world],
  );

  const onMove = useCallback(
    (value: [number, number]) => {
      setPos(value);
      void apply(value);
    },
    [apply],
  );

  /**
   * Let go and the puck settles onto the nearest checked point.
   *
   * Dragging stays continuous and exploratory; the resolve happens at the one
   * moment the user has stopped and is listening, which is also the moment a
   * guarantee is worth most. Uses the richest role's map — the same one the
   * dots are drawn from — so what you see is what you snap to.
   */
  const onSettle = useCallback(
    (value: [number, number]) => {
      const roles = Object.values(graph?.roles ?? {});
      const best = roles
        .map((r) => lensAt(r, world))
        .filter((l): l is NonNullable<typeof l> => l !== null)
        .reduce((a, b) => (b.points.length > (a?.points.length ?? 0) ? b : a),
                null as ReturnType<typeof lensAt>);
      const snap = best ? nearestPoint(best, value) : null;
      if (!snap) return;
      const at: [number, number] = [snap[0], snap[1]];
      setPos(at);
      void apply(at);
    },
    [graph, world, apply],
  );

  const toggleLink = useCallback(
    (role: string) => {
      setLinked((prev) => {
        const next = { ...prev, [role]: prev[role] === false };
        void host.setProjectData?.(LINKS_KEY, JSON.stringify(next));
        return next;
      });
    },
    [host],
  );

  if (!graph) {
    return (
      <div
        data-testid="timbre-graph-morph-empty"
        style={{ padding: '10px 12px', fontSize: 12, lineHeight: 1.5 }}
      >
        <div style={{ opacity: 0.8, marginBottom: 8 }}>
          <b>Add Graph</b> creates the six-track group (kick, snare, hat, bass,
          pad, lead) on the shipped start patches — then <b>Generate All</b>{' '}
          to hear it, and sweep the dial to tour all six together.
        </div>
        <label
          style={{
            display: 'inline-block', padding: '5px 10px', borderRadius: 4,
            background: 'rgba(127,127,127,0.18)', cursor: 'pointer',
          }}
        >
          Import tour graph…
          <input
            type="file"
            accept=".json,application/json"
            style={{ display: 'none' }}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void importGraphFile(f);
            }}
          />
        </label>
        {importProgress !== null && (
          <div
            data-testid="timbre-import-progress"
            style={{
              marginTop: 8, height: 4, borderRadius: 2,
              background: 'rgba(127,127,127,0.2)', overflow: 'hidden',
            }}
          >
            <div
              style={{
                width: `${Math.round(importProgress * 100)}%`, height: '100%',
                background: '#2DD4BF', transition: 'width 200ms',
              }}
            />
          </div>
        )}
        {status && (
          <div style={{ marginTop: 8, color: '#d66', fontSize: 11 }}>{status}</div>
        )}
      </div>
    );
  }

  return (
    <div
      data-testid="timbre-graph-morph-section"
      style={{ padding: '10px 12px', fontSize: 12 }}
    >
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 8 }}>
        <span style={{ fontWeight: 600 }}>Map</span>
        {worldCount > 1 && (
          <button
            type="button"
            data-testid="timbre-next-world"
            onClick={() => {
              const next = (world + 1) % worldCount;
              setWorld(next);
              // land on the new world's origin: the old coordinates mean
              // nothing here, and its start patch is the honest place to begin
              setPos(ORIGIN);
              void apply(ORIGIN);
            }}
            title="Different starting patches — a different set of sounds to explore"
            style={{
              font: 'inherit', fontSize: 11, padding: '2px 8px', borderRadius: 10,
              border: `1px solid ${ACCENT}`, background: 'transparent',
              cursor: 'pointer', opacity: 0.85,
            }}
          >
            world {world + 1}/{worldCount} ↻
          </button>
        )}
        <span style={{ flex: 1 }} />
        {imported && (
          <button
            type="button"
            onClick={() => void revertToBundled()}
            title="Discard the imported map and use the one shipped with the plugin"
            style={{ font: 'inherit', background: 'none', border: 'none',
                     cursor: 'pointer', opacity: 0.55, fontSize: 11 }}
          >
            use shipped
          </button>
        )}
        <label
          title="Advanced: load a map produced by the training lab"
          style={{ opacity: 0.5, fontSize: 11, cursor: 'pointer' }}
        >
          import…
          <input
            type="file"
            accept=".json,application/json"
            style={{ display: 'none' }}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) void importGraphFile(f);
            }}
          />
        </label>
      </div>

      <div style={{ display: 'flex', gap: 10, alignItems: 'flex-start' }}>
        <XYPad
          points={padPoints}
          value={pos}
          onChange={onMove}
          onRelease={onSettle}
          hovered={hovered}
          onHover={setHovered}
          accent={ACCENT}
        />
        <div style={{ flex: 1, minWidth: 0 }}>
          {/* The one label on the surface, and only while pointing at it:
              the map stays a place to explore, not a preset list to read. */}
          <div style={{ minHeight: 16, opacity: 0.6, fontSize: 11 }}>
            {hovered !== null ? padPoints[hovered]?.name : ''}
          </div>
          <button
            type="button"
            onClick={() => onMove(ORIGIN)}
            style={{ font: 'inherit', fontSize: 11, marginTop: 6, padding: '3px 8px',
                     borderRadius: 4, border: '1px solid rgba(127,127,127,0.35)',
                     background: 'transparent', cursor: 'pointer' }}
          >
            home
          </button>
        </div>
      </div>

      {/* link chips: freeze a synth without touching the rest */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
        {TIMBRE_ROLES.map((role) => {
          const t = graph.roles[role];
          if (!t) return null;
          const isLinked = linked[role] !== false;
          // "holding": this role has no tour to travel, so it stays put and
          // says so rather than inventing a change.
          const holding = t.declined || (lensAt(t, world)?.points.length ?? 0) < 2;
          return (
            <button
              key={role}
              type="button"
              onClick={() => toggleLink(role)}
              title={
                isLinked
                  ? 'Linked — follows the dial. Click to freeze.'
                  : 'Frozen. Click to relink.'
              }
              style={{
                font: 'inherit', fontSize: 11, padding: '2px 8px',
                borderRadius: 10, border: '1px solid rgba(127,127,127,0.35)',
                cursor: 'pointer',
                background: isLinked ? 'rgba(45,212,191,0.15)' : 'transparent',
                opacity: isLinked ? 1 : 0.5,
              }}
            >
              {ROLE_LABELS[toTimbreRole(role) ?? 'lead'] ?? role}
              {!isLinked ? ' ⛓' : holding ? ' ·' : ''}
            </button>
          );
        })}
      </div>

      {status && (
        <div style={{ marginTop: 8, color: '#d66', fontSize: 11 }}>{status}</div>
      )}
    </div>
  );
}

export function TimbreGraphPanel(props: PluginUIProps) {
  const adapter = useMemo(() => createTimbreGraphAdapter(props.host), [props.host]);
  // Cast mirrors the bass panel: the core's option type is the unknown-meta
  // erasure of the family-typed adapter.
  const core = useGeneratorPanelCore({
    ui: props,
    adapter: adapter as GeneratorPanelAdapter,
  });

  // Live track list behind a ref so the resolver (and therefore the slots
  // memo) stays referentially stable while always reading current tracks.
  const tracksRef = useRef(core.tracks);
  tracksRef.current = core.tracks;
  /**
   * Engine ids of the tracks the dial may drive for a role.
   *
   * Scoped to members of a timbre GROUP, never "any owned track whose role
   * matches". panel-core's adoptSceneTracks claims unowned tracks of this
   * plugin's generator type, so a pre-existing synth/lead track in the scene
   * becomes visible here — and the dial was writing Surge morph values into
   * other panels' tracks (observed live on track 1083).
   */
  const groupMemberIdsRef = useRef<Set<string>>(new Set());
  /** engine id -> the member's own lens index (layered groups spread these). */
  const memberLensRef = useRef<Map<string, number>>(new Map());
  /**
   * Engine ids to drive for a role, each with the WORLD it explores.
   *
   * A layered group puts every member on the same role, so the id alone is not
   * enough to know what to write — the members differ only by lens. Carrying
   * that here keeps the apply loop honest for both modes: an ensemble member
   * simply reports lens 0 and follows the global world offset.
   */
  const resolveTrackIds = useCallback(
    (role: string): Array<{ id: string; lensIndex: number }> => {
      const want = toTimbreRole(role);
      const members = groupMemberIdsRef.current;
      return tracksRef.current
        .filter((t) => members.has(t.handle.id))
        .filter((t) => toTimbreRole(t.role) === want
          || toTimbreRole(t.handle.role) === want)
        .map((t) => ({
          id: t.handle.id,
          lensIndex: memberLensRef.current.get(t.handle.id) ?? 0,
        }));
    },
    [],
  );

  // Group membership comes from the scene-data meta the core has resolved.
  // Kept in a ref so the resolver stays referentially stable.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const sceneId = props.activeSceneId;
      if (!sceneId) {
        groupMemberIdsRef.current = new Set();
        memberLensRef.current = new Map();
        return;
      }
      try {
        const sceneData = await props.host.getAllSceneData(sceneId);
        const groups = parseTrackGroups<TimbreGroupMeta>(sceneData, timbreGroupSpec);
        const lensByDbId = new Map<string, number>();
        for (const g of groups) {
          for (const m of g.members) {
            // fall back to member index: in a layered group that spreads the
            // members across worlds, which is the whole point of the mode
            lensByDbId.set(m.dbId, m.meta.lensIndex ?? m.meta.memberIndex);
          }
        }
        if (cancelled) return;
        const live = tracksRef.current.filter((t) => lensByDbId.has(t.handle.dbId));
        groupMemberIdsRef.current = new Set(live.map((t) => t.handle.id));
        memberLensRef.current = new Map(
          live.map((t) => [t.handle.id, lensByDbId.get(t.handle.dbId) ?? 0]),
        );
      } catch {
        groupMemberIdsRef.current = new Set();
        memberLensRef.current = new Map();
      }
    })();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.activeSceneId, props.host, core.tracks]);

  const slots = useMemo(
    () => ({
      beforeRows: (
        <MorphSection
          host={props.host}
          activeSceneId={props.activeSceneId}
          resolveTrackIds={resolveTrackIds}
          onTracksChanged={() => void core.loadTracks()}
        />
      ),
    }),
    // core.loadTracks is referentially stable per panel-core's contract
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [props.host, props.activeSceneId],
  );

  return <GeneratorPanelShell core={core} slots={slots} />;
}
