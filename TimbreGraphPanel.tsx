/**
 * Timbre Graph panel — six coupled Surge XT tracks and one morph dial.
 *
 * Built on the SDK's panel-core: `useGeneratorPanelCore` + `GeneratorPanelShell`
 * own the track rows, mixer strip, sound drawer/history, shuffle, add-track
 * button and render phases — all shared with the other generator families.
 * This file contributes only what is unique here: the MORPH SECTION rendered
 * in the shell's `beforeRows` slot.
 *
 * The dial replays a precomputed morph graph (built by the training lab): a
 * render-verified parameter snapshot per synth per control position. Runtime
 * is linear interpolation between two snapshots — no model, no solver, no
 * latency. Per-track link toggles freeze a synth without touching the rest.
 *
 * Findings that shaped this UI (docs/TRAINING.md C13): expressiveness is
 * asymmetric and role-specific, so a track that cannot follow the current
 * direction shows "holding" instead of faking motion.
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

const ROLE_LABELS: Record<TimbreRole, string> = {
  kick: 'Kick',
  snare: 'Snare',
  hat: 'Hat',
  bass: 'Bass',
  pad: 'Chord Pad',
  lead: 'Lead',
};

/** Shape of the artifact written by `tglab morph`. */
export interface MorphGraph {
  version: string;
  axis: { name: string; vector: number[] };
  control_points: number[];
  roles: Record<
    string,
    {
      role: string;
      preset_id: string;
      /** Engine track id, stamped at import. Identity is never a name. */
      track_id?: string;
      /** Path to the anchor's .fxp, restored at import. */
      fxp_path?: string;
      name: string;
      param_names: string[];
      /** Measured per-parameter audibility (Jacobian row norm); 0 = inert. */
      sensitivity?: number[];
      baseline: number[];
      snapshots: number[][];
      cosine: number[];
      declined: boolean;
    }
  >;
  quality?: Record<string, unknown>;
}

const GRAPH_KEY = 'timbre-graph.morph';
const LINKS_KEY = 'timbre-graph.links';

/**
 * Linear interpolation between the two verified snapshots either side of
 * `control`. This is the entire runtime of the instrument.
 */
export function paramsAt(
  controlPoints: readonly number[],
  snapshots: readonly number[][],
  control: number,
): number[] {
  if (controlPoints.length === 0) return [];
  const lo = controlPoints[0];
  const hi = controlPoints[controlPoints.length - 1];
  const c = Math.min(Math.max(control, lo), hi);
  let j = 1;
  while (j < controlPoints.length - 1 && controlPoints[j] < c) j += 1;
  const x0 = controlPoints[j - 1];
  const x1 = controlPoints[j];
  const w = x1 === x0 ? 0 : (c - x0) / (x1 - x0);
  const a = snapshots[j - 1] ?? [];
  const b = snapshots[j] ?? a;
  return a.map((v, i) => v * (1 - w) + (b[i] ?? v) * w);
}

/** Which dial directions this track can actually follow. */
export function reachableDirections(
  controlPoints: readonly number[],
  snapshots: readonly number[][],
  baseline: readonly number[],
): { negative: boolean; positive: boolean } {
  const moves = (i: number): boolean =>
    (snapshots[i] ?? []).some((v, k) => Math.abs(v - (baseline[k] ?? v)) > 1e-6);
  let negative = false;
  let positive = false;
  controlPoints.forEach((c, i) => {
    if (c < 0 && moves(i)) negative = true;
    if (c > 0 && moves(i)) positive = true;
  });
  return { negative, positive };
}

/** The morph section rendered above the standard track rows. Exported for tests. */
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
  resolveTrackIds: (role: string) => string[];
  onTracksChanged: () => void;
}) {
  // The plugin SHIPS with a measured graph; a project-stored one (from an
  // import) overrides it. The dial is therefore live on first open — the
  // training lab is for re-training, not a user prerequisite.
  const [graph, setGraph] = useState<MorphGraph | null>(
    BUNDLED_GRAPH as unknown as MorphGraph,
  );
  const [imported, setImported] = useState(false);
  const [control, setControl] = useState(0);
  /**
   * Gesture strength as a target PERCEPTUAL effect, in standardized
   * descriptor units (1.0 ~ one corpus standard deviation of timbre change).
   *
   * Scaling by parameter movement was measurably backwards. Damped
   * least-squares prefers small moves, so a sensitive control needs only a
   * tiny delta while an inert one needs a large one — meaning the LARGEST
   * deltas land on the LEAST audible parameters. Normalising to peak delta
   * therefore scaled each gesture to its most inaudible component: the lead's
   * budget went to `a_width` (measured audibility 0.46) while
   * `a_filter_1_cutoff` (10.95, i.e. 24x more audible) barely moved, and the
   * lead never audibly changed.
   *
   * Using the shipped per-parameter sensitivity, the gesture is scaled by its
   * predicted audible effect instead, so every role moves by a comparable
   * amount of PERCEIVED change.
   */
  const [strength, setStrength] = useState(4);
  const [linked, setLinked] = useState<Record<string, boolean>>({});
  const [status, setStatus] = useState('');
  const [importProgress, setImportProgress] = useState<number | null>(null);
  const applying = useRef(false);
  const pending = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const raw = await host.getProjectData?.<unknown>(GRAPH_KEY);
        if (!cancelled && raw) {
          const stored = typeof raw === 'string' ? JSON.parse(raw) : raw;
          if (stored && (stored as MorphGraph).roles) {
            setGraph(stored as MorphGraph);
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
        const parsed = JSON.parse(text) as MorphGraph;
        if (!parsed?.roles || !parsed?.control_points?.length) {
          setStatus('Not a morph graph: missing roles/control_points.');
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
          if (t.fxp_path && typeof host.applySurgeFxpPreset === 'function') {
            try {
              await host.applySurgeFxpPreset(handle.id, t.fxp_path);
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
        setControl(0);
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
    setGraph(BUNDLED_GRAPH as unknown as MorphGraph);
    setImported(false);
    setControl(0);
  }, [host]);

  /** Coalesced apply: while a write is in flight the newest position wins. */
  const apply = useCallback(
    async (value: number) => {
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
        let wrote = 0;
        for (const role of Object.keys(graph.roles)) {
          if (linked[role] === false) continue;
          const t = graph.roles[role];
          if (!t || t.declined) continue;
          const live = resolveTrackIds(role);
          const targets = live.length > 0 ? live : t.track_id ? [t.track_id] : [];
          if (targets.length === 0) continue;
          // DELTAS from the dial centre, scaled by depth, applied RELATIVE to
          // each track's live parameters — the morph rides on whatever sound
          // the track currently has instead of snapping it to the artifact's
          // absolute operating point. Only params the graph actually moves
          // are sent, so untouched controls stay untouched.
          const values = paramsAt(graph.control_points, t.snapshots, value);
          const centre = paramsAt(graph.control_points, t.snapshots, 0);
          const raw = values.map((v, i) => v - centre[i]);
          const span = Math.max(
            Math.abs(graph.control_points[graph.control_points.length - 1]),
            1e-9,
          );
          const dialFrac = Math.min(1, Math.abs(value) / span);

          // Predicted audible effect of the gesture: |delta ⊙ sensitivity|.
          // Scaling by this (rather than by delta size) puts the budget where
          // the sound actually lives. Falls back to delta magnitude for a
          // legacy artifact that carries no sensitivity.
          const sens = t.sensitivity;
          const effect = Math.sqrt(
            raw.reduce((acc, d, i) => {
              const w = sens ? (sens[i] ?? 0) : 1;
              return acc + (d * w) ** 2;
            }, 0),
          );
          const fallbackPeak = Math.max(...raw.map(Math.abs));
          const gain = sens && effect > 1e-9
            ? (strength * dialFrac) / effect
            : fallbackPeak > 1e-9
              ? (0.5 * dialFrac) / fallbackPeak
              : 0;

          const params: Record<string, number> = {};
          t.param_names.forEach((name, i) => {
            // Per-parameter ceiling: a gain sized for perceptual effect can
            // ask an inert control for an absurd move, which just rails it.
            const delta = Math.max(-0.6, Math.min(0.6, raw[i] * gain));
            if (Math.abs(delta) > 1e-5) params[name] = delta;
          });
          if (Object.keys(params).length === 0) continue;
          if (typeof host.setSynthParameters !== 'function') {
            setStatus('This host predates SDK 2.57.0 — parameter writes unavailable.');
            return;
          }
          for (const id of targets) {
            try {
              await host.setSynthParameters(id, params, 0, { relative: true });
              wrote += 1;
            } catch (err) {
              // Log per failure so it is diagnosable in renderer-logs, and
              // keep going: the other five synths must still morph.
              failures.push(role);
              console.error(
                `[TimbreGraph] morph write failed role=${role} track=${id}`,
                err,
              );
            }
          }
        }
        if (wrote === 0) {
          setStatus('No live tracks to drive — Add Graph first, then move the dial.');
        } else if (failures.length > 0) {
          setStatus(`morph failed on: ${[...new Set(failures)].join(', ')}`);
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
    [graph, host, linked, resolveTrackIds, strength],
  );

  const onDial = useCallback(
    (value: number) => {
      setControl(value);
      void apply(value);
    },
    [apply],
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
          pad, lead) on the shipped anchor patches — then <b>Generate All</b>{' '}
          to hear it, and sweep the dial to morph all six together.
        </div>
        <label
          style={{
            display: 'inline-block', padding: '5px 10px', borderRadius: 4,
            background: 'rgba(127,127,127,0.18)', cursor: 'pointer',
          }}
        >
          Import morph graph…
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

  const lo = graph.control_points[0];
  const hi = graph.control_points[graph.control_points.length - 1];

  return (
    <div
      data-testid="timbre-graph-morph-section"
      style={{ padding: '10px 12px', fontSize: 12 }}
    >
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 8 }}>
        <span style={{ fontWeight: 600 }}>Morph</span>
        <span style={{ opacity: 0.6 }}>axis: {graph.axis.name}</span>
        <span style={{ flex: 1 }} />
        <label style={{ display: 'flex', alignItems: 'center', gap: 4, opacity: 0.75 }}>
          depth
          <select
            aria-label="morph depth"
            value={strength}
            onChange={(e) => setStrength(Number(e.target.value))}
            style={{ font: 'inherit', fontSize: 11 }}
          >
            <option value={1}>verified</option>
            <option value={2}>subtle</option>
            <option value={4}>strong</option>
            <option value={8}>extreme</option>
            <option value={16}>max</option>
          </select>
        </label>
        {imported && (
          <button
            type="button"
            onClick={() => void revertToBundled()}
            title="Discard the imported graph and use the one shipped with the plugin"
            style={{ font: 'inherit', background: 'none', border: 'none',
                     cursor: 'pointer', opacity: 0.55, fontSize: 11 }}
          >
            use shipped
          </button>
        )}
        <label
          title="Advanced: load a graph produced by the training lab"
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

      <input
        aria-label="morph"
        type="range"
        min={lo}
        max={hi}
        step={(hi - lo) / 200}
        value={control}
        onChange={(e) => onDial(Number(e.target.value))}
        style={{ width: '100%' }}
      />
      <div style={{ display: 'flex', justifyContent: 'space-between', opacity: 0.55 }}>
        <span>−</span>
        <button
          type="button"
          onClick={() => onDial(0)}
          style={{ font: 'inherit', background: 'none', border: 'none',
                   cursor: 'pointer', opacity: 0.7 }}
        >
          reset
        </button>
        <span>+</span>
      </div>

      {/* link chips: freeze a synth without touching the rest */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
        {TIMBRE_ROLES.map((role) => {
          const t = graph.roles[role];
          if (!t) return null;
          const dirs = reachableDirections(
            graph.control_points, t.snapshots, t.baseline,
          );
          const isLinked = linked[role] !== false;
          const holding =
            t.declined ||
            (control < 0 && !dirs.negative) ||
            (control > 0 && !dirs.positive);
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
  const resolveTrackIds = useCallback((role: string): string[] => {
    const want = toTimbreRole(role);
    const members = groupMemberIdsRef.current;
    return tracksRef.current
      .filter((t) => members.has(t.handle.id))
      .filter((t) => toTimbreRole(t.role) === want
        || toTimbreRole(t.handle.role) === want)
      .map((t) => t.handle.id);
  }, []);

  // Group membership comes from the scene-data meta the core has resolved.
  // Kept in a ref so the resolver stays referentially stable.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const sceneId = props.activeSceneId;
      if (!sceneId) {
        groupMemberIdsRef.current = new Set();
        return;
      }
      try {
        const sceneData = await props.host.getAllSceneData(sceneId);
        const groups = parseTrackGroups<TimbreGroupMeta>(sceneData, timbreGroupSpec);
        const memberDbIds = new Set(
          groups.flatMap((g) => g.members.map((m) => m.dbId)),
        );
        if (cancelled) return;
        groupMemberIdsRef.current = new Set(
          tracksRef.current
            .filter((t) => memberDbIds.has(t.handle.dbId))
            .map((t) => t.handle.id),
        );
      } catch {
        groupMemberIdsRef.current = new Set();
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
