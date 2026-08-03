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
  useGeneratorPanelCore,
} from '@signalsandsorcery/plugin-sdk';
import { createTimbreGraphAdapter } from './src/timbre-graph-adapter';
import {
  APP_ROLE_TOKENS,
  TIMBRE_ROLES,
  toTimbreRole,
  type TimbreRole,
} from './src/role-patterns';
import { TIMBRE_GROUP_META_KEY } from './src/timbre-group-meta';

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
  onTracksChanged,
}: {
  host: PluginHost;
  activeSceneId: string | null;
  onTracksChanged: () => void;
}) {
  const [graph, setGraph] = useState<MorphGraph | null>(null);
  const [control, setControl] = useState(0);
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
          setGraph(typeof raw === 'string' ? JSON.parse(raw) : (raw as MorphGraph));
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

  const clearGraph = useCallback(async () => {
    await host.setProjectData?.(GRAPH_KEY, '');
    setGraph(null);
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
      try {
        for (const role of Object.keys(graph.roles)) {
          if (linked[role] === false) continue;
          const t = graph.roles[role];
          if (!t || t.declined || !t.track_id) continue;
          const values = paramsAt(graph.control_points, t.snapshots, value);
          const params: Record<string, number> = {};
          t.param_names.forEach((name, i) => {
            params[name] = values[i];
          });
          if (typeof host.setSynthParameters !== 'function') {
            setStatus('This host predates SDK 2.57.0 — parameter writes unavailable.');
            return;
          }
          await host.setSynthParameters(t.track_id, params);
        }
        setStatus('');
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
    [graph, host, linked],
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
          No morph graph loaded. Build one with the training lab
          (<code>tglab probe</code>, then <code>tglab morph --axis softer</code>),
          then import it — that creates the six tracks with their measured
          anchor presets. Press each track&apos;s Generate to hear it.
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
        <button
          type="button"
          onClick={() => void clearGraph()}
          style={{ font: 'inherit', background: 'none', border: 'none',
                   cursor: 'pointer', opacity: 0.55, fontSize: 11 }}
        >
          unload
        </button>
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

  const slots = useMemo(
    () => ({
      beforeRows: (
        <MorphSection
          host={props.host}
          activeSceneId={props.activeSceneId}
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
