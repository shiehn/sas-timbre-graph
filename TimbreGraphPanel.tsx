/**
 * Timbre Graph panel — six coupled Surge XT tracks and one morph dial.
 *
 * The dial is deliberately unlabeled at the point of use: turning it walks a
 * precomputed morph graph, and every synth follows in whatever parameter
 * vocabulary it actually has. The graph is built offline by the training lab
 * (`tglab morph`) and is a plain list of render-verified parameter snapshots,
 * so this component only ever INTERPOLATES between two of them — no solver, no
 * model, no inference, and therefore no latency.
 *
 * Findings that shaped this UI (docs/TRAINING.md C13):
 *   - Roles differ in what they can express. A kick cannot get "brighter" at
 *     all (0.000 reach in its high bands), so a track that cannot follow the
 *     current axis is shown as holding rather than hidden or faked.
 *   - Expressiveness is asymmetric: some patches move one way only, so the
 *     dial marks the dead side instead of pretending it works.
 *   - Unlink is trivially cheap here: stop applying that track's row.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { PluginUIProps } from '@signalsandsorcery/plugin-sdk';

/**
 * Snapshots are applied through `host.setSynthParameters` (SDK 2.57.0), which
 * writes several synth parameters BY NAME in one call. It stays optional in the
 * SDK surface, so an older host degrades to a clear message rather than a
 * crash — and rather than a half-applied patch, which would correspond to no
 * verified control position at all.
 */

const ROLES = ['kick', 'snare', 'hat', 'bass', 'pad', 'lead'] as const;
type Role = (typeof ROLES)[number];

const ROLE_LABELS: Record<Role, string> = {
  kick: 'Kick',
  snare: 'Snare',
  hat: 'Hat',
  bass: 'Bass',
  pad: 'Chord Pad',
  lead: 'Lead',
};

/** Shape of the artifact written by `tglab morph`. */
interface MorphGraph {
  version: string;
  axis: { name: string; vector: number[] };
  control_points: number[];
  roles: Record<
    string,
    {
      role: string;
      preset_id: string;
      /** Engine/DB track id, stamped at setup time. Never a display name. */
      track_id?: string;
      /** Path to the anchor's .fxp, restored at setup via applySurgeFxpPreset. */
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

export function TimbreGraphPanel({ host }: PluginUIProps) {
  const [graph, setGraph] = useState<MorphGraph | null>(null);
  const [control, setControl] = useState(0);
  const [linked, setLinked] = useState<Record<Role, boolean>>(
    () => Object.fromEntries(ROLES.map((r) => [r, true])) as Record<Role, boolean>,
  );
  const [status, setStatus] = useState<string>('');
  const applying = useRef(false);
  const pending = useRef<number | null>(null);

  // ── load the graph + saved link state ────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const raw = await host?.getProjectData?.<unknown>(GRAPH_KEY);
        if (!cancelled && raw) setGraph(typeof raw === 'string' ? JSON.parse(raw) : (raw as MorphGraph));
        const savedLinks = await host?.getProjectData?.<unknown>(LINKS_KEY);
        if (!cancelled && savedLinks) {
          const parsed = typeof savedLinks === 'string' ? JSON.parse(savedLinks) : savedLinks;
          setLinked((prev) => ({ ...prev, ...(parsed as Record<Role, boolean>) }));
        }
      } catch {
        /* no graph yet — the panel explains how to build one */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [host]);

  const tracks = useMemo(() => {
    if (!graph) return [];
    return ROLES.map((role) => {
      const t = graph.roles[role];
      if (!t) return { role, present: false as const };
      const dirs = reachableDirections(graph.control_points, t.snapshots, t.baseline);
      return { role, present: true as const, track: t, dirs };
    });
  }, [graph]);

  /**
   * Push the current dial position to the engine. Coalesced: while a write is
   * in flight the newest position is remembered and sent afterwards, so
   * dragging never queues a backlog of stale parameter writes.
   */
  const apply = useCallback(
    async (value: number) => {
      if (!graph || !host) return;
      if (applying.current) {
        pending.current = value;
        return;
      }
      applying.current = true;
      try {
        for (const role of ROLES) {
          if (!linked[role]) continue;
          const t = graph.roles[role];
          if (!t || t.declined) continue;
          const values = paramsAt(graph.control_points, t.snapshots, value);
          const params: Record<string, number> = {};
          t.param_names.forEach((name, i) => {
            params[name] = values[i];
          });
          if (typeof host.setSynthParameters !== 'function') {
            setStatus('This host predates SDK 2.57.0 — synth parameter writes unavailable.');
            return;
          }
          await host.setSynthParameters(t.track_id ?? role, params);
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

  /**
   * Import a morph-graph file and stand the instrument up: one track per
   * role, Surge XT loaded, the anchor preset restored from its .fxp, and the
   * engine track id stamped into the graph so every later parameter write
   * addresses the track by id (never by name). Idempotent per import — it
   * always creates a fresh set of tracks it owns.
   */
  const importGraphFile = useCallback(
    async (file: File) => {
      if (!host) return;
      setStatus('Importing…');
      try {
        const text = await new Promise<string>((resolve, reject) => {
          // FileReader rather than File.text(): identical result, but present
          // in every runtime this panel meets (File.text is missing in some).
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
        const roleIds = Object.keys(parsed.roles);
        for (const role of roleIds) {
          const t = parsed.roles[role];
          setStatus(`Setting up ${role}…`);
          const handle = await host.createTrack({
            name: `TG ${ROLE_LABELS[(role as Role)] ?? role}`,
            role,
          });
          await host.loadSynthPlugin(handle.id, 'Surge XT');
          if (t.fxp_path && typeof host.applySurgeFxpPreset === 'function') {
            try {
              await host.applySurgeFxpPreset(handle.id, t.fxp_path);
            } catch (err) {
              // Missing file on this machine: the track still exists with a
              // default patch; snapshots remain meaningful relative moves.
              console.warn('[TimbreGraph] preset restore failed', role, err);
            }
          }
          t.track_id = handle.id;
        }
        await host.setProjectData?.(GRAPH_KEY, JSON.stringify(parsed));
        setGraph(parsed);
        setControl(0);
        setStatus('');
      } catch (err) {
        setStatus(err instanceof Error ? err.message : 'Import failed');
      }
    },
    [host],
  );

  const clearGraph = useCallback(async () => {
    await host?.setProjectData?.(GRAPH_KEY, '');
    setGraph(null);
    setControl(0);
  }, [host]);

  const toggleLink = useCallback(
    (role: Role) => {
      setLinked((prev) => {
        const next = { ...prev, [role]: !prev[role] };
        void host?.setProjectData?.(LINKS_KEY, JSON.stringify(next));
        return next;
      });
    },
    [host],
  );

  if (!graph) {
    return (
      <div style={{ padding: 12, fontSize: 12, lineHeight: 1.5 }}>
        <div style={{ fontWeight: 600, marginBottom: 6 }}>Timbre Graph</div>
        <div style={{ opacity: 0.8, marginBottom: 10 }}>
          No morph graph loaded yet. Build one with the training lab:
          <pre style={{ marginTop: 8, fontSize: 11, opacity: 0.9 }}>
            cd sas-timbre-graph/training{'\n'}
            tglab probe{'\n'}
            tglab morph --axis softer
          </pre>
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
        <div style={{ marginTop: 6, opacity: 0.6, fontSize: 11 }}>
          e.g. training/workspace/morph-softer.json
        </div>
        {status && (
          <div style={{ marginTop: 8, color: '#d66', fontSize: 11 }}>{status}</div>
        )}
      </div>
    );
  }

  const lo = graph.control_points[0];
  const hi = graph.control_points[graph.control_points.length - 1];

  return (
    <div style={{ padding: 12, fontSize: 12 }} data-testid="timbre-graph-panel">
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 10 }}>
        <span style={{ fontWeight: 600 }}>Timbre Graph</span>
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

      {/* the dial */}
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
      <div style={{ display: 'flex', justifyContent: 'space-between', opacity: 0.55, marginBottom: 10 }}>
        <span>−</span>
        <button
          type="button"
          onClick={() => onDial(0)}
          style={{ font: 'inherit', background: 'none', border: 'none', cursor: 'pointer', opacity: 0.7 }}
        >
          reset
        </button>
        <span>+</span>
      </div>

      {/* six always-visible tracks */}
      <div style={{ display: 'grid', gap: 4 }}>
        {tracks.map((row) => {
          const holding =
            row.present &&
            (row.track.declined ||
              (control < 0 && !row.dirs.negative) ||
              (control > 0 && !row.dirs.positive));
          return (
            <div
              key={row.role}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '3px 6px',
                borderRadius: 4,
                opacity: row.present && linked[row.role] ? 1 : 0.5,
                background: 'rgba(127,127,127,0.08)',
              }}
            >
              <button
                type="button"
                aria-label={`${linked[row.role] ? 'unlink' : 'link'} ${row.role}`}
                onClick={() => toggleLink(row.role)}
                title={linked[row.role] ? 'Linked — follows the dial' : 'Unlinked — frozen'}
                style={{
                  font: 'inherit',
                  cursor: 'pointer',
                  border: 'none',
                  background: 'none',
                  width: 18,
                }}
              >
                {linked[row.role] ? '🔗' : '⛓️‍💥'}
              </button>
              <span style={{ width: 74 }}>{ROLE_LABELS[row.role]}</span>
              <span style={{ flex: 1, opacity: 0.7 }}>
                {row.present ? row.track.name : '—'}
              </span>
              <span style={{ opacity: 0.6, fontSize: 11 }}>
                {!row.present
                  ? 'absent'
                  : !linked[row.role]
                    ? 'frozen'
                    : holding
                      ? 'holding'
                      : 'following'}
              </span>
            </div>
          );
        })}
      </div>

      {status && (
        <div style={{ marginTop: 8, color: '#d66', fontSize: 11 }}>{status}</div>
      )}
    </div>
  );
}
