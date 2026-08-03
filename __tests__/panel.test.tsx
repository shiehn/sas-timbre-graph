import { createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react-dom/test-utils';
import TimbreGraphPlugin, { timbreGraphManifest } from '../index';
import { MorphSection, TimbreGraphPanel, paramsAt, reachableDirections } from '../TimbreGraphPanel';
import pluginJson from '../plugin.json';
import { createTimbreGraphAdapter } from '../src/timbre-graph-adapter';

describe('TimbreGraphPlugin registration surface', () => {
  it('keeps class metadata in sync with plugin.json', () => {
    const p = new TimbreGraphPlugin();
    expect(p.id).toBe(pluginJson.id);
    expect(p.displayName).toBe(pluginJson.displayName);
    expect(p.version).toBe(pluginJson.version);
    expect(p.generatorType).toBe(pluginJson.generatorType);
  });

  it('exports the manifest for sas-app registration', () => {
    // src/plugins/index.ts imports { Plugin, manifest } from the package root
    expect(timbreGraphManifest).toBe(pluginJson);
    expect(timbreGraphManifest.id).toBe('@signalsandsorcery/timbre-graph');
    expect(timbreGraphManifest.main).toBe('dist/index.js');
  });

  it('exposes the panel component', () => {
    expect(new TimbreGraphPlugin().getUIComponent()).toBe(TimbreGraphPanel);
  });
});

describe('paramsAt — the entire runtime of the instrument', () => {
  const controls = [-1, -0.5, 0, 0.5, 1];
  const snaps = [
    [0.0, 1.0],
    [0.25, 0.75],
    [0.5, 0.5],
    [0.75, 0.25],
    [1.0, 0.0],
  ];

  it('is exact at every stored control point', () => {
    controls.forEach((c, i) => {
      expect(paramsAt(controls, snaps, c)).toEqual(snaps[i]);
    });
  });

  it('interpolates linearly between neighbours', () => {
    expect(paramsAt(controls, snaps, 0.25)).toEqual([0.625, 0.375]);
  });

  it('clamps outside the dial range instead of extrapolating', () => {
    expect(paramsAt(controls, snaps, -99)).toEqual(snaps[0]);
    expect(paramsAt(controls, snaps, 99)).toEqual(snaps[4]);
  });

  it('never leaves the raw [0,1] parameter range', () => {
    for (let c = -1.5; c <= 1.5; c += 0.05) {
      paramsAt(controls, snaps, c).forEach((v) => {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
      });
    }
  });

  it('handles an empty graph without throwing', () => {
    expect(paramsAt([], [], 0.5)).toEqual([]);
  });
});

describe('reachableDirections — asymmetric expressiveness', () => {
  const controls = [-1, 0, 1];
  const baseline = [0.5, 0.5];

  it('detects a one-sided track (measured: snare/hat get softer, not harder)', () => {
    const snaps = [
      [0.5, 0.5], // negative side holds
      [0.5, 0.5],
      [0.7, 0.4], // positive side moves
    ];
    expect(reachableDirections(controls, snaps, baseline)).toEqual({
      negative: false,
      positive: true,
    });
  });

  it('detects a two-sided track', () => {
    const snaps = [
      [0.3, 0.6],
      [0.5, 0.5],
      [0.7, 0.4],
    ];
    expect(reachableDirections(controls, snaps, baseline)).toEqual({
      negative: true,
      positive: true,
    });
  });

  it('reports a fully declined track', () => {
    const snaps = [baseline, baseline, baseline];
    expect(reachableDirections(controls, snaps, baseline)).toEqual({
      negative: false,
      positive: false,
    });
  });
});

describe('TimbreGraphPanel rendering', () => {
  function render(el: React.ReactElement) {
    const container = document.createElement('div');
    document.body.appendChild(container);
    const root = createRoot(container);
    act(() => root.render(el));
    return {
      container,
      cleanup: () => {
        act(() => root.unmount());
        container.remove();
      },
    };
  }

  it('ships with a graph: the dial is live with no import', () => {
    const { container, cleanup } = render(
      createElement(MorphSection, { host: {}, activeSceneId: 'scene-1', resolveTrackIds: () => [], onTracksChanged: () => {} } as never),
    );
    // the shipped graph means a dial, not a "go run a Python CLI" message
    expect(container.querySelector('input[type="range"]')).not.toBeNull();
    expect(container.textContent).toContain('axis:');
    expect(container.textContent).not.toContain('tglab');
    cleanup();
  });

  it('lists all six roles once a graph is present', async () => {
    const graph = {
      version: 'morph-graph-v1',
      axis: { name: 'softer', vector: [] },
      control_points: [-1, 0, 1],
      roles: Object.fromEntries(
        ['kick', 'snare', 'hat', 'bass', 'pad', 'lead'].map((role) => [
          role,
          {
            role,
            preset_id: role,
            name: `${role} patch`,
            param_names: ['a', 'b'],
            baseline: [0.5, 0.5],
            snapshots: [
              [0.4, 0.6],
              [0.5, 0.5],
              [0.6, 0.4],
            ],
            cosine: [0.7, 1, 0.7],
            declined: false,
          },
        ]),
      ),
    };
    const setCalls: Array<[string, Record<string, number>]> = [];
    const host = {
      getProjectData: async (k: string) => (k.endsWith('morph') ? graph : null),
      setProjectData: async () => {},
      setSynthParameters: async (role: string, params: Record<string, number>) => {
        setCalls.push([role, params]);
      },
    };

    const { container, cleanup } = render(
      createElement(MorphSection, { host, activeSceneId: 'scene-1', resolveTrackIds: () => [], onTracksChanged: () => {} } as never),
    );
    await act(async () => {
      await Promise.resolve();
    });
    for (const label of ['Kick', 'Snare', 'Hat', 'Bass', 'Chord Pad', 'Lead']) {
      expect(container.textContent).toContain(label);
    }
    expect(container.textContent).toContain('softer');
    cleanup();
  });
});

describe('TimbreGraphPanel import flow', () => {
  function render(el: React.ReactElement) {
    const container = document.createElement('div');
    document.body.appendChild(container);
    const root = createRoot(container);
    act(() => root.render(el));
    return {
      container,
      cleanup: () => {
        act(() => root.unmount());
        container.remove();
      },
    };
  }

  it('keeps import as a secondary affordance for re-training', () => {
    const { container, cleanup } = render(
      createElement(MorphSection, { host: {}, activeSceneId: 'scene-1', resolveTrackIds: () => [], onTracksChanged: () => {} } as never),
    );
    expect(container.textContent).toContain('import');
    expect(container.querySelector('input[type="file"]')).not.toBeNull();
    cleanup();
  });

  it('creates a track per role, loads Surge, restores the fxp, stamps ids', async () => {
    const calls: string[] = [];
    let savedGraph: string | null = null;
    const host = {
      getProjectData: async () => null,
      setProjectData: async (_k: string, v: string) => {
        savedGraph = v;
      },
      createTrack: async ({ name, role, loadSynth, synthName }: {
        name: string; role: string; loadSynth?: boolean; synthName?: string;
      }) => {
        calls.push(`create:${role}:${loadSynth ? synthName : 'nosynth'}`);
        return { id: `engine-${role}`, name, dbId: `db-${role}` };
      },
      setSceneData: async (sceneId: string, key: string, value: unknown) => {
        const v = value as { groupId: string; memberIndex: number; role: string };
        calls.push(`meta:${sceneId}:${key}:${v.memberIndex}:${v.role}`);
      },
      applySurgeFxpPreset: async (trackId: string, fxp: string) => {
        calls.push(`fxp:${trackId}:${fxp.split('/').pop()}`);
      },
    };
    const graph = {
      version: 'morph-graph-v1',
      axis: { name: 'softer', vector: [] },
      control_points: [-1, 0, 1],
      roles: {
        kick: {
          role: 'kick', preset_id: 'k', name: 'Kick 909ish',
          fxp_path: '/fake/Kick 909ish.fxp',
          param_names: ['a'], baseline: [0.5],
          snapshots: [[0.4], [0.5], [0.6]], cosine: [1, 1, 1], declined: false,
        },
      },
    };
    const file = new File([JSON.stringify(graph)], 'morph-softer.json', {
      type: 'application/json',
    });

    const { container, cleanup } = render(
      createElement(MorphSection, { host, activeSceneId: 'scene-1', resolveTrackIds: () => [], onTracksChanged: () => {} } as never),
    );
    await act(async () => Promise.resolve());
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    Object.defineProperty(input, 'files', { value: [file] });
    await act(async () => {
      input.dispatchEvent(new Event('change', { bubbles: true }));
      // FileReader + the async import chain need a couple of ticks
      await new Promise((r) => setTimeout(r, 0));
      await Promise.resolve();
    });

    expect(calls).toEqual([
      'create:kicks:Surge XT',                       // canonical app role token
      'meta:scene-1:track:db-kicks:timbreGroup:0:kick', // group seam stamped
      'fxp:engine-kicks:Kick 909ish.fxp',
    ]);
    expect(savedGraph).not.toBeNull();
    const stamped = JSON.parse(savedGraph as unknown as string);
    expect(stamped.roles.kick.track_id).toBe('engine-kicks');
    cleanup();
  });

  it('survives a missing fxp on this machine (track still created)', async () => {
    const calls: string[] = [];
    const host = {
      getProjectData: async () => null,
      setProjectData: async () => {},
      createTrack: async ({ role }: { role: string }) => ({
        id: `engine-${role}`, name: role, dbId: role,
      }),
      setSceneData: async () => {},
      applySurgeFxpPreset: async () => {
        calls.push('fxp-attempted');
        throw new Error('FILE_NOT_FOUND');
      },
    };
    const graph = {
      version: 'morph-graph-v1',
      axis: { name: 'softer', vector: [] },
      control_points: [-1, 0, 1],
      roles: {
        pad: {
          role: 'pad', preset_id: 'p', name: 'Gone',
          fxp_path: '/missing.fxp',
          param_names: ['a'], baseline: [0.5],
          snapshots: [[0.4], [0.5], [0.6]], cosine: [1, 1, 1], declined: false,
        },
      },
    };
    const file = new File([JSON.stringify(graph)], 'g.json');
    const { container, cleanup } = render(
      createElement(MorphSection, { host, activeSceneId: 'scene-1', resolveTrackIds: () => [], onTracksChanged: () => {} } as never),
    );
    await act(async () => Promise.resolve());
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    Object.defineProperty(input, 'files', { value: [file] });
    await act(async () => {
      input.dispatchEvent(new Event('change', { bubbles: true }));
      await new Promise((r) => setTimeout(r, 0));
    });
    expect(calls).toEqual(['fxp-attempted']);
    // panel proceeded to the loaded state despite the missing preset
    expect(container.textContent).toContain('Chord Pad');
    cleanup();
  });
});

describe('group meta + role mapping', () => {
  it('narrows valid meta and rejects garbage', () => {
    const { asTimbreGroupMeta } = require('../src/timbre-group-meta');
    expect(asTimbreGroupMeta({ groupId: 'g', memberIndex: 2, role: 'hat' }))
      .toEqual({ groupId: 'g', memberIndex: 2, role: 'hat' });
    // unknown role falls back to the member-index role
    expect(asTimbreGroupMeta({ groupId: 'g', memberIndex: 3, role: 'zither' }))
      .toEqual({ groupId: 'g', memberIndex: 3, role: 'bass' });
    expect(asTimbreGroupMeta(null)).toBeNull();
    expect(asTimbreGroupMeta({ memberIndex: 1 })).toBeNull();
  });

  it('group completeness requires the anchor', () => {
    const { timbreGroupIsComplete } = require('../src/timbre-group-meta');
    const member = (i: number) => ({ dbId: `d${i}`, meta: { groupId: 'g', memberIndex: i, role: 'kick' }, track: {} });
    expect(timbreGroupIsComplete({ groupId: 'g', members: [member(0), member(3)] })).toBe(true);
    expect(timbreGroupIsComplete({ groupId: 'g', members: [member(1), member(2)] })).toBe(false);
  });

  it('maps app role tokens to timbre roles and back (perc vocabulary)', () => {
    const { APP_ROLE_TOKENS, toTimbreRole } = require('../src/role-patterns');
    // the canonical taxonomy round-trips
    for (const [timbre, token] of Object.entries(APP_ROLE_TOKENS)) {
      expect(toTimbreRole(token as string)).toBe(timbre);
    }
    // and generation derives the pattern from the role with no typed prompt
    expect(toTimbreRole('kicks')).toBe('kick');
    expect(toTimbreRole('hats')).toBe('hat');
    expect(toTimbreRole('vocals')).toBeNull();
  });

  it('tiles the role pattern across the scene and clips at the loop end', () => {
    const { tiledPattern } = require('../src/role-patterns');
    const twoBars = tiledPattern('kick', 2, 4);          // 8 beats = 2 tiles
    expect(twoBars.length).toBe(10);                     // 5 hits per tile
    expect(Math.max(...twoBars.map((n: { startBeat: number }) => n.startBeat))).toBeLessThan(8);
    const odd = tiledPattern('pad', 1, 3);               // 3/4 bar: clipped tile
    for (const n of odd) {
      expect(n.startBeat + n.durationBeats).toBeLessThanOrEqual(3 + 1e-9);
    }
  });
});

describe('seeded MIDI variation', () => {
  const { variedPattern } = require('../src/role-patterns');

  it('is deterministic for a given seed', () => {
    expect(variedPattern('kick', 4, 4, 123)).toEqual(variedPattern('kick', 4, 4, 123));
  });

  it('regenerating with a new seed changes the pattern (the live complaint)', () => {
    const roles = ['kick', 'snare', 'hat', 'bass', 'pad', 'lead'] as const;
    for (const role of roles) {
      const runs = new Set(
        [1, 2, 3, 4, 5].map((seed) => JSON.stringify(variedPattern(role, 4, 4, seed))),
      );
      expect(runs.size).toBeGreaterThan(1);
    }
  });

  it('variation never breaks role identity or the scene bounds', () => {
    for (let seed = 0; seed < 30; seed++) {
      for (const n of variedPattern('kick', 2, 4, seed)) {
        expect(n.pitch).toBe(36);                     // a kick stays a kick at C1
      }
      for (const n of variedPattern('hat', 2, 4, seed)) {
        expect(n.pitch).toBe(66);
      }
      for (const role of ['bass', 'pad', 'lead'] as const) {
        for (const n of variedPattern(role, 2, 4, seed)) {
          expect(n.startBeat + n.durationBeats).toBeLessThanOrEqual(8 + 1e-9);
          expect(n.velocity).toBeGreaterThanOrEqual(1);
          expect(n.velocity).toBeLessThanOrEqual(127);
        }
      }
    }
  });
});

describe('dial drives LIVE tracks, not stamped ids', () => {
  function render(el: React.ReactElement) {
    const container = document.createElement('div');
    document.body.appendChild(container);
    const root = createRoot(container);
    act(() => root.render(el));
    return { container, cleanup: () => { act(() => root.unmount()); container.remove(); } };
  }

  it('writes to the resolver-supplied ids even when the stamp is stale', async () => {
    const writes: string[] = [];
    const payloads: Array<{ params: Record<string, number>; options?: { relative?: boolean } }> = [];
    const graph = {
      version: 'morph-graph-v1',
      axis: { name: 'softer', vector: [] },
      control_points: [-1, 0, 1],
      roles: {
        kick: {
          role: 'kick', preset_id: 'k', name: 'K', track_id: 'DEAD-1235',
          param_names: ['a'], baseline: [0.5],
          snapshots: [[0.4], [0.5], [0.6]], cosine: [1, 1, 1], declined: false,
        },
      },
    };
    const host = {
      getProjectData: async (k: string) => (k.endsWith('morph') ? graph : null),
      setProjectData: async () => {},
      setSynthParameters: async (
        trackId: string,
        params: Record<string, number>,
        _idx?: number,
        options?: { relative?: boolean },
      ) => { writes.push(trackId); payloads.push({ params, options }); },
    };
    const { container, cleanup } = render(
      createElement(MorphSection, {
        host, activeSceneId: 's1',
        resolveTrackIds: (role: string) => (role === 'kick' ? ['LIVE-1464'] : []),
        onTracksChanged: () => {},
      } as never),
    );
    await act(async () => Promise.resolve());
    const slider = container.querySelector('input[type="range"]') as HTMLInputElement;
    await act(async () => {
      // controlled input: go through the native setter, then fire `input`
      // (what React's onChange actually listens to for range elements)
      const set = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype, 'value',
      )?.set;
      set?.call(slider, '0.9');
      slider.dispatchEvent(new Event('input', { bubbles: true }));
      await new Promise((r) => setTimeout(r, 0));
    });
    expect(writes).toEqual(['LIVE-1464']);            // never DEAD-1235
    // deltas ride the live sound: relative mode, normalized from the centre
    expect(payloads[0].options).toEqual({ relative: true });
    const delta = payloads[0].params['a'];
    expect(Math.sign(delta)).toBe(1);                 // +0.9 of the dial => up
    // the graph's own move is a timid 0.1; normalization must lift the peak
    // toward the strength target (default 'strong' = 0.5) x dial fraction 0.9
    expect(Math.abs(delta)).toBeGreaterThan(0.3);
    cleanup();
  });
});

describe('group rows hidden until generation', () => {
  const { TimbreGroupRow } = require('../src/TimbreGroupRow') as {
    TimbreGroupRow: React.ComponentType<Record<string, unknown>>;
  };

  function makeCtx(rendered: string[]) {
    return {
      collapsed: false,
      onToggleCollapse: () => {},
      handlers: { generate: () => {} },
      deleteGroup: async () => {},
      renderDefaultTrackRow: (t: { handle: { id: string } }) => {
        rendered.push(t.handle.id);
        return null;
      },
    };
  }
  const member = (id: string, hasMidi: boolean, isGenerating = false) => ({
    dbId: `db-${id}`,
    meta: { groupId: 'g', memberIndex: 0, role: 'kick' },
    track: { handle: { id }, hasMidi, isGenerating },
  });

  function render(el: React.ReactElement) {
    const container = document.createElement('div');
    document.body.appendChild(container);
    const root = createRoot(container);
    act(() => root.render(el));
    return { container, cleanup: () => { act(() => root.unmount()); container.remove(); } };
  }

  it('shows a single CTA and NO member rows before any MIDI exists', () => {
    const rendered: string[] = [];
    const { container, cleanup } = render(
      createElement(TimbreGroupRow, {
        group: { groupId: 'g', members: [member('a', false), member('b', false)] },
        ctx: makeCtx(rendered),
      }),
    );
    expect(rendered).toEqual([]);                    // no rows to "press play" on
    expect(container.textContent).toContain('Generate All');
    expect(container.querySelector('[data-testid="timbre-group-unmaterialized"]')).not.toBeNull();
    cleanup();
  });

  it('shows rows as soon as generation starts', () => {
    const rendered: string[] = [];
    const { cleanup } = render(
      createElement(TimbreGroupRow, {
        group: { groupId: 'g', members: [member('a', false, true), member('b', false)] },
        ctx: makeCtx(rendered),
      }),
    );
    expect(rendered).toEqual(['a', 'b']);
    cleanup();
  });

  it('shows rows once MIDI exists', () => {
    const rendered: string[] = [];
    const { cleanup } = render(
      createElement(TimbreGroupRow, {
        group: { groupId: 'g', members: [member('a', true), member('b', true)] },
        ctx: makeCtx(rendered),
      }),
    );
    expect(rendered).toEqual(['a', 'b']);
    cleanup();
  });
});

describe('gesture normalization by AUDIBLE EFFECT', () => {
  /**
   * The verified graph is timid by design (trust radius): roles move their
   * parameters by a mean of 3-10% of range, and unevenly between roles — one
   * role's deltas can be a third of another's. A blunt multiplier keeps that
   * imbalance. Rescaling each role's delta VECTOR to a target peak preserves
   * the measured direction while making every preset move comparably.
   */
  /** The panel's rule: scale by predicted perceptual effect, not delta size. */
  const normalize = (
    raw: number[], sens: number[], strength: number, dialFrac: number,
  ): number[] => {
    const effect = Math.sqrt(
      raw.reduce((a, d, i) => a + (d * (sens[i] ?? 0)) ** 2, 0),
    );
    const gain = effect > 1e-9 ? (strength * dialFrac) / effect : 0;
    return raw.map((v) => Math.max(-0.6, Math.min(0.6, v * gain)));
  };

  it('spends the budget on AUDIBLE params, not the biggest delta', () => {
    // the live lead bug: biggest delta on a near-inert control (a_width,
    // sensitivity 0.46) while the audible one (cutoff, 10.95) barely moved
    const raw = [0.133, 0.005];           // [width, cutoff]
    const sens = [0.46, 10.95];
    const out = normalize(raw, sens, 4, 1);
    // cutoff's move must not be dwarfed: its EFFECT should dominate
    const widthEffect = Math.abs(out[0] * sens[0]);
    const cutoffEffect = Math.abs(out[1] * sens[1]);
    expect(cutoffEffect).toBeGreaterThan(widthEffect * 0.5);
  });

  it('reaches the requested perceptual effect when no clamp is hit', () => {
    const raw = [0.05, -0.02];
    const sens = [40, 20];              // audible enough to need small deltas
    const out = normalize(raw, sens, 4, 1);
    expect(Math.max(...out.map(Math.abs))).toBeLessThan(0.6);   // unclamped
    const effect = Math.sqrt(out.reduce((a, d, i) => a + (d * sens[i]) ** 2, 0));
    expect(effect).toBeCloseTo(4, 1);
  });

  it('caps the effect rather than railing params when sensitivity is low', () => {
    // asking 4 units of change from near-inert controls must clamp, not
    // demand impossible parameter moves
    const raw = [0.02, -0.01];
    const sens = [5, 3];
    const out = normalize(raw, sens, 4, 1);
    expect(Math.max(...out.map(Math.abs))).toBeCloseTo(0.6);
    const effect = Math.sqrt(out.reduce((a, d, i) => a + (d * sens[i]) ** 2, 0));
    expect(effect).toBeLessThan(4);
  });

  it('equalises perceived change across roles of different timidity', () => {
    const eff = (out: number[], sens: number[]): number =>
      Math.sqrt(out.reduce((a, d, i) => a + (d * sens[i]) ** 2, 0));
    const a = normalize([0.19, -0.10], [2, 1], 4, 1);
    const b = normalize([0.03, -0.015], [2, 1], 4, 1);
    expect(eff(a, [2, 1])).toBeCloseTo(eff(b, [2, 1]), 1);
  });

  it('preserves direction and relative proportions below the clamp', () => {
    const raw = [0.10, -0.05, 0.025];
    const sens = [30, 30, 30];
    const out = normalize(raw, sens, 2, 1);
    expect(Math.max(...out.map(Math.abs))).toBeLessThan(0.6);
    expect(out.map((v) => Math.sign(v))).toEqual([1, -1, 1]);
    expect(out[0] / out[1]).toBeCloseTo(raw[0] / raw[1]);
    expect(out[0] / out[2]).toBeCloseTo(raw[0] / raw[2]);
  });

  it('scales with dial position and is a no-op at centre', () => {
    const raw = [0.1, -0.05];
    const sens = [4, 2];
    expect(Math.max(...normalize(raw, sens, 4, 0).map(Math.abs))).toBe(0);
    const half = normalize(raw, sens, 4, 0.5);
    const eff = Math.sqrt(half.reduce((a, d, i) => a + (d * sens[i]) ** 2, 0));
    expect(eff).toBeCloseTo(2, 1);
  });

  it('never asks for an absurd move even when sensitivity is tiny', () => {
    const out = normalize([0.1, 0.1], [0.001, 0.001], 16, 1);
    expect(Math.max(...out.map(Math.abs))).toBeLessThanOrEqual(0.6);
  });

  it('handles an all-zero (declined) role without dividing by zero', () => {
    expect(normalize([0, 0, 0], [1, 1, 1], 4, 1)).toEqual([0, 0, 0]);
  });
});

describe('the plugin ships with a usable graph', () => {
  const { BUNDLED_GRAPH } = require('../src/bundled-graph');

  it('covers all six roles with render-verified snapshots', () => {
    for (const role of ['kick', 'snare', 'hat', 'bass', 'pad', 'lead']) {
      const t = BUNDLED_GRAPH.roles[role];
      expect(t).toBeDefined();
      expect(t.param_names.length).toBeGreaterThan(0);
      expect(t.snapshots.length).toBe(BUNDLED_GRAPH.control_points.length);
      expect(t.snapshots[0].length).toBe(t.param_names.length);
    }
  });

  it('references anchor presets by PORTABLE relative path', () => {
    // an absolute authoring-machine path would not exist on any other machine
    for (const role of Object.keys(BUNDLED_GRAPH.roles)) {
      const fxp = BUNDLED_GRAPH.roles[role].fxp_path;
      expect(fxp).toBeTruthy();
      expect(fxp.startsWith('/')).toBe(false);
      expect(fxp).toMatch(/\.fxp$/);
    }
  });

  it('carries no machine- or project-specific state', () => {
    for (const role of Object.keys(BUNDLED_GRAPH.roles)) {
      // engine track ids belong to a project, never to a shipped artifact
      expect(BUNDLED_GRAPH.roles[role].track_id).toBeUndefined();
    }
  });

  it('has a centre control point so the dial has a true no-op', () => {
    expect(BUNDLED_GRAPH.control_points).toContain(0);
  });
});

describe('one failing role must not silence the others', () => {
  /**
   * A single try around the whole role loop meant the first throw aborted
   * every remaining role — and `lead` iterates LAST, so any earlier error
   * (e.g. a wrongly-adopted track rejecting a parameter) made the lead look
   * permanently dead. Live symptom: "the lead synth has still never changed".
   */
  function render(el: React.ReactElement) {
    const container = document.createElement('div');
    document.body.appendChild(container);
    const root = createRoot(container);
    act(() => root.render(el));
    return { container, cleanup: () => { act(() => root.unmount()); container.remove(); } };
  }

  const roleTrack = (role: string) => ({
    role, preset_id: role, name: role,
    param_names: ['a'], baseline: [0.5],
    snapshots: [[0.4], [0.5], [0.6]], cosine: [1, 1, 1], declined: false,
  });

  it('still writes later roles after an earlier role throws', async () => {
    const written: string[] = [];
    const graph = {
      version: 'morph-graph-v1',
      axis: { name: 'softer', vector: [] },
      control_points: [-1, 0, 1],
      roles: {
        kick: roleTrack('kick'),
        pad: roleTrack('pad'),
        lead: roleTrack('lead'),   // last in order: the canary
      },
    };
    const host = {
      getProjectData: async (k: string) => (k.endsWith('morph') ? graph : null),
      setProjectData: async () => {},
      setSynthParameters: async (trackId: string) => {
        if (trackId === 'track-kick') throw new Error('Unknown parameter(s)');
        written.push(trackId);
      },
    };
    const { container, cleanup } = render(
      createElement(MorphSection, {
        host, activeSceneId: 's1',
        resolveTrackIds: (role: string) => [`track-${role}`],
        onTracksChanged: () => {},
      } as never),
    );
    await act(async () => Promise.resolve());
    const slider = container.querySelector('input[type="range"]') as HTMLInputElement;
    await act(async () => {
      const set = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype, 'value',
      )?.set;
      set?.call(slider, '0.9');
      slider.dispatchEvent(new Event('input', { bubbles: true }));
      await new Promise((r) => setTimeout(r, 0));
    });

    // kick failed, but pad AND lead were still driven
    expect(written).toContain('track-pad');
    expect(written).toContain('track-lead');
    expect(written).not.toContain('track-kick');
    // and the failure is surfaced, not swallowed
    expect(container.textContent).toContain('morph failed on');
    expect(container.textContent).toContain('kick');
    cleanup();
  });
});

describe('generation reports no progress (the core paces the bar)', () => {
  const makeServices = (updates: Array<Record<string, unknown>>) => ({
    host: {
      getMusicalContext: async () => ({ bars: 4, bpm: 120, timeSignature: '4/4' }),
      writeMidiClip: jest.fn(async () => undefined),
    },
    updateTrack: (_id: string, patch: Record<string, unknown>) => {
      updates.push(patch);
    },
  });

  const track = (role: string) => ({
    role,
    handle: { id: 'e1', name: `timbre-${role}`, role },
  });

  /**
   * There is no LLM in this path — six roles finish in ~180 ms — so a bar
   * would paint after the clip is already audible and read as "it generated
   * before I asked". Locked in a test because the tempting fix to a
   * fast-feeling operation is to add progress, not remove it.
   */
  it('never sets generationProgress', async () => {
    const updates: Array<Record<string, unknown>> = [];
    const adapter = createTimbreGraphAdapter({} as never);
    await adapter.generation!.generate!(
      track('kicks') as never,
      makeServices(updates) as never,
    );
    expect(updates.length).toBeGreaterThan(0);
    for (const patch of updates) {
      expect(patch).not.toHaveProperty('generationProgress');
    }
  });

  it('still marks the row as having MIDI', async () => {
    const updates: Array<Record<string, unknown>> = [];
    const adapter = createTimbreGraphAdapter({} as never);
    await adapter.generation!.generate!(
      track('kicks') as never,
      makeServices(updates) as never,
    );
    expect(updates).toEqual([{ hasMidi: true }]);
  });

  it('writes exactly one clip, spanning the scene', async () => {
    const updates: Array<Record<string, unknown>> = [];
    const services = makeServices(updates);
    const adapter = createTimbreGraphAdapter({} as never);
    await adapter.generation!.generate!(track('bass') as never, services as never);

    expect(services.host.writeMidiClip).toHaveBeenCalledTimes(1);
    const [trackId, clip] = services.host.writeMidiClip.mock.calls[0] as unknown as [
      string,
      { startTime: number; endTime: number; notes: unknown[] },
    ];
    expect(trackId).toBe('e1');
    expect(clip.startTime).toBe(0);
    expect(clip.endTime).toBeCloseTo(8, 5);   // 4 bars of 4/4 at 120 bpm
    expect(clip.notes.length).toBeGreaterThan(0);
  });

  it('refuses a track whose role is not a timbre role', async () => {
    const adapter = createTimbreGraphAdapter({} as never);
    await expect(
      adapter.generation!.generate!(
        track('vocals') as never,
        makeServices([]) as never,
      ),
    ).rejects.toThrow(/no timbre role/);
  });
});

describe('generation uses the standard LLM machinery, prompt derived from role', () => {
  const CHORDS = [
    { symbol: 'Cm7', startQn: 0, endQn: 4 },
    { symbol: 'Ab', startQn: 4, endQn: 8 },
  ];

  const musical = {
    key: 'C', mode: 'minor', bpm: 120, bars: 4, genre: 'Techno',
    timeSignature: '4/4', chordProgression: CHORDS,
    contractPrompt: 'dark, driving',
  };

  const makeHost = (content: string) => ({
    getMusicalContext: async () => musical,
    getGenerationContext: async () => ({
      chordProgression: { key: { tonic: 'C', mode: 'minor' }, chordsWithTiming: CHORDS, genre: null },
      concurrentTracks: [],
    }),
    generateWithLLM: jest.fn(async () => ({ content })),
    writeMidiClip: jest.fn(async () => undefined),
  });

  const services = (host: unknown) => ({
    host,
    updateTrack: () => {},
  });

  const track = (role: string) => ({
    role,
    handle: { id: 'e1', name: `timbre-${role}`, role },
  });

  const runFor = async (role: string, content: string) => {
    const host = makeHost(content);
    const adapter = createTimbreGraphAdapter({} as never);
    await adapter.generation!.generate!(
      track(role) as never,
      services(host) as never,
    );
    const llmCalls = host.generateWithLLM.mock.calls as unknown as unknown[][];
    const clipCalls = host.writeMidiClip.mock.calls as unknown as unknown[][];
    const llmCall = llmCalls[0]?.[0] as { system: string; user: string } | undefined;
    const clip = clipCalls[0]?.[1] as
      | { notes: Array<{ pitch: number; startBeat: number; velocity: number }> }
      | undefined;
    return { host, llmCall, clip };
  };

  const NOTES = JSON.stringify({
    notes: [
      { pitch: 60, startBeat: 0, durationBeats: 1, velocity: 100 },
      { pitch: 63, startBeat: 2, durationBeats: 1, velocity: 90 },
    ],
  });

  it('calls the LLM — generation is not local pattern-stamping', async () => {
    const { host } = await runFor('bass', NOTES);
    expect(host.generateWithLLM).toHaveBeenCalledTimes(1);
  });

  it('derives the prompt from the role, with no user text', async () => {
    const { llmCall } = await runFor('kicks', NOTES);
    expect(llmCall!.user).toContain('kick drum');
    // the panel has no prompt box; nothing should look like a quoted request
    expect(llmCall!.user).not.toContain('User request');
  });

  it('sends the scene harmony so pitched roles generate in key', async () => {
    const { llmCall } = await runFor('bass', NOTES);
    expect(llmCall!.user).toContain('Key: C minor');
    expect(llmCall!.user).toContain('Cm7 (beats 0-4)');
    expect(llmCall!.user).toContain('BPM: 120');
    // the scene contract is what carries production intent
    expect(llmCall!.user).toContain('dark, driving');
  });

  it('omits chords for percussion — an unpitched part has no harmony', async () => {
    const { llmCall } = await runFor('hats', NOTES);
    expect(llmCall!.user).not.toContain('Chord Progression');
    expect(llmCall!.user).toContain('Key: C minor'); // key/tempo still useful
  });

  it('pins percussion to the pitch the lab measured the patch at', async () => {
    const { clip } = await runFor('kicks', NOTES);
    // model asked for 60 and 63; a kick patch is only a kick at 36
    expect(clip!.notes.every((n) => n.pitch === 36)).toBe(true);
    expect(clip!.notes.length).toBe(2);
  });

  it('transposes pitched roles by whole octaves, preserving intervals', async () => {
    const { clip } = await runFor('bass', NOTES);
    const pitches = clip!.notes.map((n) => n.pitch);
    // 60/63 shift as a phrase into the measured bass register [28,50]...
    expect(pitches).toEqual([48, 51].map((p) => p - 12));
    // ...and the rising minor third is still a rising minor third
    expect(pitches[1] - pitches[0]).toBe(3);
  });

  it('never lets a phrase shift invert an interval', async () => {
    const wide = JSON.stringify({
      notes: [
        { pitch: 40, startBeat: 0, durationBeats: 1, velocity: 100 },
        { pitch: 47, startBeat: 1, durationBeats: 1, velocity: 100 },
        { pitch: 52, startBeat: 2, durationBeats: 1, velocity: 100 },
      ],
    });
    const { clip } = await runFor('lead', wide);
    const pitches = clip!.notes.map((n) => n.pitch);
    // strictly ascending in, strictly ascending out
    expect(pitches[1]).toBeGreaterThan(pitches[0]);
    expect(pitches[2]).toBeGreaterThan(pitches[1]);
  });

  it('falls back to the probe pattern when the model fails', async () => {
    const host = {
      ...makeHost(NOTES),
      generateWithLLM: jest.fn(async () => {
        throw new Error('offline');
      }),
    };
    const adapter = createTimbreGraphAdapter({} as never);
    await adapter.generation!.generate!(
      track('leads') as never,
      services(host) as never,
    );
    const clipCalls = host.writeMidiClip.mock.calls as unknown as unknown[][];
    const clip = clipCalls[0]?.[1] as { notes: unknown[] };
    // a failed model must never leave a silent track
    expect(clip.notes.length).toBeGreaterThan(0);
  });

  it('falls back when the model returns unusable content', async () => {
    const { clip, host } = await runFor('pads', 'I am not JSON');
    expect(host.generateWithLLM).toHaveBeenCalled();
    expect(clip!.notes.length).toBeGreaterThan(0);
  });
});

describe('MIDI generation never touches the sound', () => {
  /**
   * The shipped anchors are what the morph was MEASURED against — the dial's
   * verified behaviour is a property of those exact patches. If generating
   * notes could re-pick or re-apply a preset, the graph would silently stop
   * describing the instrument the user is hearing.
   */
  it('calls no preset, shuffle, or parameter API while writing notes', async () => {
    const forbidden = {
      applySurgeFxpPreset: jest.fn(),
      shufflePreset: jest.fn(),
      setSynthParameters: jest.fn(),
      loadPlugin: jest.fn(),
      setTrackPreset: jest.fn(),
    };
    const host = {
      ...forbidden,
      getMusicalContext: async () => ({
        key: 'C', mode: 'minor', bpm: 120, bars: 4, genre: null,
        timeSignature: '4/4', chordProgression: [], contractPrompt: null,
      }),
      getGenerationContext: async () => ({
        chordProgression: { key: { tonic: 'C', mode: 'minor' }, chordsWithTiming: [], genre: null },
        concurrentTracks: [],
      }),
      generateWithLLM: async () => ({
        content: JSON.stringify({
          notes: [{ pitch: 60, startBeat: 0, durationBeats: 1, velocity: 100 }],
        }),
      }),
      writeMidiClip: jest.fn(async () => undefined),
    };

    const adapter = createTimbreGraphAdapter({} as never);
    for (const role of ['kicks', 'snares', 'hats', 'bass', 'pads', 'leads']) {
      await adapter.generation!.generate!(
        { role, handle: { id: `e-${role}`, name: role, role } } as never,
        { host, updateTrack: () => {} } as never,
      );
    }

    expect(host.writeMidiClip).toHaveBeenCalledTimes(6);
    for (const [name, fn] of Object.entries(forbidden)) {
      expect(fn).not.toHaveBeenCalled();
      expect(`${name}:${fn.mock.calls.length}`).toBe(`${name}:0`);
    }
  });
});
