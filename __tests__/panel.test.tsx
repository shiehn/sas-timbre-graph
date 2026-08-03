import { createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react-dom/test-utils';
import TimbreGraphPlugin, { timbreGraphManifest } from '../index';
import { MorphSection, TimbreGraphPanel, paramsAt, reachableDirections } from '../TimbreGraphPanel';
import pluginJson from '../plugin.json';

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

  it('explains how to build a graph when none is loaded', () => {
    const { container, cleanup } = render(
      createElement(MorphSection, { host: {}, activeSceneId: 'scene-1', onTracksChanged: () => {} } as never),
    );
    expect(container.textContent).toContain('No morph graph loaded');
    expect(container.textContent).toContain('tglab morph');
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
      createElement(MorphSection, { host, activeSceneId: 'scene-1', onTracksChanged: () => {} } as never),
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

  it('offers an import affordance in the empty state', () => {
    const { container, cleanup } = render(
      createElement(MorphSection, { host: {}, activeSceneId: 'scene-1', onTracksChanged: () => {} } as never),
    );
    expect(container.textContent).toContain('Import morph graph');
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
      createElement(MorphSection, { host, activeSceneId: 'scene-1', onTracksChanged: () => {} } as never),
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
      createElement(MorphSection, { host, activeSceneId: 'scene-1', onTracksChanged: () => {} } as never),
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
