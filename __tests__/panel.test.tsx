import { createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react-dom/test-utils';
import TimbreGraphPlugin, { timbreGraphManifest } from '../index';
import { MorphSection, TimbreGraphPanel } from '../TimbreGraphPanel';
import pluginJson from '../plugin.json';
import { createTimbreGraphAdapter } from '../src/timbre-graph-adapter';

const TIMBRE_ROLES = ['kick', 'snare', 'hat', 'bass', 'pad', 'lead'] as const;

const PAD_PX = 220;

/**
 * Drive the X/Y pad. jsdom gives every element a zero-sized rect and the pad
 * converts pixels to unit coordinates through it, so it needs a real one.
 */
function movePad(container: HTMLElement, xy: readonly [number, number]): void {
  const el = container.querySelector('canvas') as HTMLCanvasElement;
  el.getBoundingClientRect = () =>
    ({ left: 0, top: 0, width: PAD_PX, height: PAD_PX, right: PAD_PX,
       bottom: PAD_PX, x: 0, y: 0, toJSON: () => ({}) }) as DOMRect;
  act(() => {
    el.dispatchEvent(new MouseEvent('mousedown', {
      clientX: xy[0] * PAD_PX,
      clientY: (1 - xy[1]) * PAD_PX,
      bubbles: true,
    }));
  });
}

/**
 * A minimal map: three points at known coordinates over two parameters.
 * Roles differ only by name, so any role-specific behaviour under test is the
 * panel's and not the fixture's.
 */
function tourFixture(
  overrides: Partial<{
    snapshots: number[][];
    param_names: string[];
    declined: boolean;
  }> = {},
): Record<string, unknown> {
  const xys: Array<[number, number]> = [[0, 0], [0.5, 0.5], [1, 1]];
  return {
    version: 'map-graph-v2',
    roles: Object.fromEntries(
      TIMBRE_ROLES.map((role) => [
        role,
        {
          role,
          lenses: [
            {
              lens: { preset_id: `${role}-lens`, name: `${role} lens`, category: 'X' },
              param_names: overrides.param_names ?? ['a', 'b'],
              points: xys.map((xy, i) => ({
                preset_id: `${role}-${i}`,
                name: `${role} ${i}`,
                fxp_path: `Basses/${role}-${i}.fxp`,
                xy,
              })),
              snapshots: overrides.snapshots ?? [
                [0.4, 0.6],
                [0.5, 0.5],
                [0.6, 0.4],
              ],
              sharpness: 12,
              neighbours: 4,
              snap: 0.9,
            },
          ],
          declined: overrides.declined ?? false,
        },
      ]),
    ),
  };
}

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

  /**
   * The host REFUSES an undeclared capability at the IPC boundary, so a
   * manifest that lags the code is a total generation failure, not a
   * degradation. Live symptom: six rows of 'requires capability "requiresLLM"
   * but it is not declared in the manifest' the first time Generate All ran
   * after generation moved from probe patterns to the LLM.
   */
  it('declares every capability the panel actually uses', () => {
    const src = require('fs').readFileSync(
      require('path').join(__dirname, '..', 'src', 'timbre-graph-adapter.ts'),
      'utf8',
    );
    const caps = pluginJson.capabilities as Record<string, boolean>;
    if (/host\.generateWithLLM|services\.host\.generateWithLLM/.test(src)) {
      expect(caps.requiresLLM).toBe(true);
    }
    if (/Surge XT|applySurgeFxpPreset|setSynthParameters/.test(src)) {
      expect(caps.requiresSurgeXT).toBe(true);
    }
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
    // the shipped map means a pad, not a "go run a Python CLI" message
    expect(container.querySelector('canvas')).not.toBeNull();
    expect(container.textContent).not.toContain('tglab');
    cleanup();
  });

  it('shows the map without naming what is on it', () => {
    const { container, cleanup } = render(
      createElement(MorphSection, { host: {}, activeSceneId: 'scene-1', resolveTrackIds: () => [], onTracksChanged: () => {} } as never),
    );
    // the dots are drawn, but the surface stays unlabelled until you hover:
    // naming every point would turn a search surface into a preset browser
    expect(container.querySelector('[data-testid="timbre-xy-pad"]')).not.toBeNull();
    expect(container.textContent).not.toMatch(/anchor|preset \d/i);
    cleanup();
  });

  it('lists all six roles once a tour is present', async () => {
    const graph = tourFixture();
    const host = {
      getProjectData: async (k: string) => (k.endsWith('map') ? graph : null),
      setProjectData: async () => {},
      setSynthParameters: async () => {},
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
      version: 'map-graph-v2',
      roles: {
        kick: {
          role: 'kick',
          lenses: [{
            lens: { preset_id: 'kl', name: 'K lens', category: 'X' },
            param_names: ['a'],
            points: [
              { preset_id: 'k0', name: 'Kick 909ish', fxp_path: 'Kick 909ish.fxp', xy: [0, 0] },
              { preset_id: 'k1', name: 'Kick Tech', fxp_path: 'Kick Tech.fxp', xy: [1, 1] },
            ],
            snapshots: [[0.4], [0.6]],
            sharpness: 12, neighbours: 4, snap: 0.9,
          }],
          declined: false,
        },
      },
    };
    const file = new File([JSON.stringify(graph)], 'patchmap.json', {
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
      // anchor 0 only: it is the structural lens for the whole tour
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
      version: 'map-graph-v2',
      roles: {
        pad: {
          role: 'pad',
          lenses: [{
            lens: { preset_id: 'pl', name: 'P lens', category: 'X' },
            param_names: ['a'],
            points: [
              { preset_id: 'p0', name: 'Gone', fxp_path: 'missing.fxp', xy: [0, 0] },
              { preset_id: 'p1', name: 'Also gone', fxp_path: 'missing-2.fxp', xy: [1, 1] },
            ],
            snapshots: [[0.4], [0.6]],
            sharpness: 12, neighbours: 4, snap: 0.9,
          }],
          declined: false,
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

  it('refuses a retired 1-D tour file and keeps the shipped map', async () => {
    const created: string[] = [];
    const host = {
      getProjectData: async () => null,
      setProjectData: async () => {},
      createTrack: async ({ role }: { role: string }) => {
        created.push(role);
        return { id: `engine-${role}`, name: role, dbId: role };
      },
      setSceneData: async () => {},
    };
    // the retired artifact: a 1-D route rather than a surface
    const legacy = {
      version: 'tour-graph-v1',
      roles: {
        kick: {
          role: 'kick', param_names: ['a'], control_points: [0, 1],
          anchors: [{ preset_id: 'k0', name: 'K', fxp_path: 'a.fxp' }],
          snapshots: [[0.4], [0.6]], declined: false,
        },
      },
    };
    const file = new File([JSON.stringify(legacy)], 'tour.json');
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

    expect(created).toEqual([]);            // nothing was stood up
    expect(container.textContent).toContain('map-graph-v2');
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

function renderIn(el: React.ReactElement) {
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => root.render(el));
  return { container, cleanup: () => { act(() => root.unmount()); container.remove(); } };
}

/** Put the pad at `xy` and let the coalesced apply settle. */
async function settleAt(
  container: HTMLElement, xy: readonly [number, number],
): Promise<void> {
  movePad(container, xy);
  await act(async () => { await new Promise((r) => setTimeout(r, 0)); });
}

describe('the pad drives LIVE tracks, not stamped ids', () => {
  const kickMap = {
    version: 'map-graph-v2',
    roles: {
      kick: {
        role: 'kick', track_id: 'DEAD-1235',
        lenses: [{
          lens: { preset_id: 'kl', name: 'K lens', category: 'X' },
          param_names: ['a'],
          points: [
            { preset_id: 'k0', name: 'K0', fxp_path: 'a.fxp', xy: [0, 0] },
            { preset_id: 'k1', name: 'K1', fxp_path: 'b.fxp', xy: [1, 1] },
          ],
          snapshots: [[0.4], [0.6]],
          sharpness: 12, neighbours: 4, snap: 0.9,
        }],
        declined: false,
      },
    },
  };

  it('writes to the resolver-supplied ids even when the stamp is stale', async () => {
    const writes: string[] = [];
    const payloads: Array<{ params: Record<string, number>; options?: { relative?: boolean } }> = [];
    const host = {
      getProjectData: async (k: string) => (k.endsWith('map') ? kickMap : null),
      setProjectData: async () => {},
      setSynthParameters: async (
        trackId: string,
        params: Record<string, number>,
        _idx?: number,
        options?: { relative?: boolean },
      ) => { writes.push(trackId); payloads.push({ params, options }); },
    };
    const { container, cleanup } = renderIn(
      createElement(MorphSection, {
        host, activeSceneId: 's1',
        resolveTrackIds: (role: string) => (role === 'kick' ? [{ id: 'LIVE-1464', lensIndex: 0 }] : []),
        onTracksChanged: () => {},
      } as never),
    );
    await act(async () => Promise.resolve());
    await settleAt(container, [1, 1]);

    expect(writes).toEqual(['LIVE-1464']);            // never DEAD-1235
    // deltas ride the live sound: relative mode, measured from the origin
    expect(payloads[0].options).toEqual({ relative: true });
    // standing on point 1 reproduces it exactly: 0.6 - 0.4
    expect(payloads[0].params['a']).toBeCloseTo(0.2, 6);
    cleanup();
  });
});

describe('the pad sends RAW deltas, and the surface is path-independent', () => {
  async function padTo(
    xy: readonly [number, number],
    snapshots: number[][],
    param_names = ['a', 'b'],
    via: Array<readonly [number, number]> = [],
  ): Promise<Record<string, number>[]> {
    const payloads: Record<string, number>[] = [];
    const graph = tourFixture({ snapshots, param_names });
    const host = {
      getProjectData: async (k: string) => (k.endsWith('map') ? graph : null),
      setProjectData: async () => {},
      setSynthParameters: async (_id: string, params: Record<string, number>) => {
        payloads.push(params);
      },
    };
    const { container, cleanup } = renderIn(
      createElement(MorphSection, {
        host, activeSceneId: 's1',
        resolveTrackIds: (role: string) => (role === 'kick' ? [{ id: 'T1', lensIndex: 0 }] : []),
        onTracksChanged: () => {},
      } as never),
    );
    await act(async () => Promise.resolve());
    for (const v of [...via, xy]) await settleAt(container, v);
    cleanup();
    return payloads.slice(-1);
  }

  const SNAPS = [[0.4, 0.6], [0.5, 0.5], [0.6, 0.4]];

  it('reproduces a point exactly when you stand on it', async () => {
    // fixture points sit at (0,0), (0.5,0.5), (1,1)
    const [mid] = await padTo([0.5, 0.5], SNAPS);
    expect(mid['a']).toBeCloseTo(0.1, 6);     // 0.5 - 0.4
    expect(mid['b']).toBeCloseTo(-0.1, 6);
    const [end] = await padTo([1, 1], SNAPS);
    expect(end['a']).toBeCloseTo(0.2, 6);     // 0.6 - 0.4
    expect(end['b']).toBeCloseTo(-0.2, 6);
  });

  it('RETURNS home rather than sending nothing', async () => {
    /**
     * A zero delta is an instruction, not a no-op: the host applies
     * `base + delta`, so a parameter that is never sent keeps whatever the
     * last position left it at. Skipping zeros made the control
     * path-dependent and put the origin sound out of reach (2026-08-04).
     */
    const [home] = await padTo([0, 0], SNAPS, ['a', 'b'], [[0.8, 0.8]]);
    expect(home).toBeDefined();
    expect(home['a']).toBe(0);
    expect(home['b']).toBe(0);
  });

  it('is PATH-INDEPENDENT: the same place always sounds the same', async () => {
    const direct = (await padTo([0.5, 0.5], SNAPS))[0];
    const viaFar = (await padTo([0.5, 0.5], SNAPS, ['a', 'b'], [[1, 1], [0, 0]]))[0];
    expect(viaFar).toEqual(direct);
  });

  it('applies no clamp — a full 0 to 1 swing survives intact', async () => {
    const [end] = await padTo([1, 1], [[0, 1], [0.5, 0.5], [1, 0]]);
    expect(end['a']).toBeCloseTo(1, 6);
    expect(end['b']).toBeCloseTo(-1, 6);
  });

  it('writes a parameter that moves anywhere, even where its delta is zero', async () => {
    const [end] = await padTo([1, 1], [[0.2, 0.5], [0.5, 0.9], [0.8, 0.5]]);
    expect(Object.keys(end).sort()).toEqual(['a', 'b']);
    expect(end['b']).toBe(0);
  });

  it('omits a parameter the map NEVER moves', async () => {
    const [end] = await padTo([1, 1], [[0.2, 0.5], [0.5, 0.5], [0.8, 0.5]]);
    expect(Object.keys(end)).toEqual(['a']);
  });

  it('does not claim the tracks are missing when there is nothing to send', async () => {
    const graph = tourFixture();
    const host = {
      getProjectData: async (k: string) => (k.endsWith('map') ? graph : null),
      setProjectData: async () => {},
      setSynthParameters: async () => {},
    };
    const { container, cleanup } = renderIn(
      createElement(MorphSection, {
        host, activeSceneId: 's1', resolveTrackIds: () => [{ id: 'T1', lensIndex: 0 }],
        onTracksChanged: () => {},
      } as never),
    );
    await act(async () => Promise.resolve());
    await settleAt(container, [0.5, 0.5]);
    await settleAt(container, [0, 0]);
    expect(container.textContent).not.toContain('No live tracks');
    cleanup();
  });
});

describe('one failing role must not silence the others', () => {
  it('still writes later roles after an earlier role throws', async () => {
    const written: string[] = [];
    const graph = tourFixture();
    const host = {
      getProjectData: async (k: string) => (k.endsWith('map') ? graph : null),
      setProjectData: async () => {},
      setSynthParameters: async (trackId: string) => {
        if (trackId === 'track-kick') throw new Error('Unknown parameter(s)');
        written.push(trackId);
      },
    };
    const { container, cleanup } = renderIn(
      createElement(MorphSection, {
        host, activeSceneId: 's1',
        resolveTrackIds: (role: string) => [{ id: `track-${role}`, lensIndex: 0 }],
        onTracksChanged: () => {},
      } as never),
    );
    await act(async () => Promise.resolve());
    await settleAt(container, [1, 1]);

    // `lead` iterates LAST, so an early throw used to make it look dead
    expect(written).toContain('track-pad');
    expect(written).toContain('track-lead');
    expect(written).not.toContain('track-kick');
    expect(container.textContent).toContain('failed on');
    expect(container.textContent).toContain('kick');
    cleanup();
  });
});

describe('the plugin ships with a usable map', () => {
  const { BUNDLED_GRAPH } = require('../src/bundled-graph');

  it('is a patch map, not the retired 1-D tour', () => {
    expect(BUNDLED_GRAPH.version).toBe('map-graph-v2');
  });

  it('offers more than one world for at least one role', () => {
    // a lens is a hard ceiling on what dragging can reach, so a single lens
    // per role would make the tool exhaustible
    const counts = (Object.values(BUNDLED_GRAPH.roles) as Array<{ lenses: unknown[] }>)
      .map((r) => r.lenses.length);
    expect(Math.max(...counts)).toBeGreaterThan(1);
  });

  it('covers all six roles with coherent, in-range data', () => {
    for (const role of ['kick', 'snare', 'hat', 'bass', 'pad', 'lead']) {
      const t = BUNDLED_GRAPH.roles[role];
      expect(t).toBeDefined();
      expect(t.declined).toBe(t.lenses.length === 0);
      for (const l of t.lenses) {
        expect(l.param_names.length).toBeGreaterThan(0);
        expect(l.points.length).toBe(l.snapshots.length);
        expect(l.points.length).toBeGreaterThan(1);
        for (const snap of l.snapshots) {
          expect(snap.length).toBe(l.param_names.length);
          for (const v of snap) {
            expect(v).toBeGreaterThanOrEqual(0);
            expect(v).toBeLessThanOrEqual(1);
          }
        }
      }
    }
  });

  it('lays every point inside the unit square the pad addresses', () => {
    for (const role of Object.keys(BUNDLED_GRAPH.roles)) {
      for (const l of BUNDLED_GRAPH.roles[role].lenses) {
        for (const p of l.points) {
          expect(p.xy[0]).toBeGreaterThanOrEqual(0);
          expect(p.xy[0]).toBeLessThanOrEqual(1);
          expect(p.xy[1]).toBeGreaterThanOrEqual(0);
          expect(p.xy[1]).toBeLessThanOrEqual(1);
        }
      }
    }
  });

  it('visits a DIFFERENT preset at every point', () => {
    for (const role of Object.keys(BUNDLED_GRAPH.roles)) {
      for (const l of BUNDLED_GRAPH.roles[role].lenses) {
        const ids = l.points.map((p: { preset_id: string }) => p.preset_id);
        expect(new Set(ids).size).toBe(ids.length);
      }
    }
  });

  it('references every preset by PORTABLE relative path', () => {
    for (const role of Object.keys(BUNDLED_GRAPH.roles)) {
      for (const l of BUNDLED_GRAPH.roles[role].lenses) {
        for (const p of l.points) {
          expect(p.fxp_path).toMatch(/\.fxp$/);
          expect(p.fxp_path.startsWith('/')).toBe(false);
        }
      }
    }
  });

  it('carries the blend settings the runtime needs, and no project state', () => {
    for (const role of Object.keys(BUNDLED_GRAPH.roles)) {
      const t = BUNDLED_GRAPH.roles[role];
      expect(t.track_id).toBeUndefined();
      for (const l of t.lenses) {
        expect(l.sharpness).toBeGreaterThan(1);
        expect(l.neighbours).toBeGreaterThanOrEqual(2);
        expect(l.snap).toBeGreaterThan(0.5);
      }
    }
  });
});

describe('the success patch resolves the row (the core only clears on error)', () => {
  const GOOD = JSON.stringify({
    notes: [
      { pitch: 60, startBeat: 0, durationBeats: 1, velocity: 100 },
      { pitch: 63, startBeat: 2, durationBeats: 1, velocity: 90 },
    ],
  });

  /** A row mid-generation: what the core hands us when generate() succeeds. */
  const GENERATING = {
    handle: { id: 'e1' },
    isGenerating: true,
    error: 'a stale error from last time',
    hasMidi: false,
    generationProgress: 0.4,
    editNotes: [],
    editBars: 0,
    editBpm: 0,
    editBeatsPerBar: 0,
  };

  const makeServices = (patched: Array<Record<string, unknown>>) => ({
    host: {
      getMusicalContext: async () => ({ bars: 4, bpm: 120, timeSignature: '4/4' }),
      generateWithLLM: jest.fn(async () => ({ content: GOOD })),
      writeMidiClip: jest.fn(async () => undefined),
    },
    // Apply the patch the way the core's setTracks does, so assertions see
    // the row the user is actually left looking at.
    updateTrack: (
      _id: string,
      patch:
        | Record<string, unknown>
        | ((t: Record<string, unknown>) => Record<string, unknown>),
    ) => {
      patched.push(
        typeof patch === 'function'
          ? patch({ ...GENERATING })
          : { ...GENERATING, ...patch },
      );
    },
  });

  const track = (role: string) => ({
    role,
    handle: { id: 'e1', name: `timbre-${role}`, role },
  });

  /**
   * The live bug: panel-core clears `isGenerating` on the ERROR path only, so
   * a SUCCESSFUL generation left every row's progress bar spinning forever.
   * Clearing it is the adapter's job — the same contract bass, pad, ensemble
   * and arp all follow.
   */
  it('clears isGenerating so the progress bar actually resolves', async () => {
    const patched: Array<Record<string, unknown>> = [];
    const adapter = createTimbreGraphAdapter({} as never);
    await adapter.generation!.generate!(
      track('kicks') as never,
      makeServices(patched) as never,
    );
    expect(patched).toHaveLength(1);
    expect(patched[0].isGenerating).toBe(false);
    expect(patched[0].generationProgress).toBe(0);
  });

  it('marks the row as having MIDI and drops any stale error', async () => {
    const patched: Array<Record<string, unknown>> = [];
    const adapter = createTimbreGraphAdapter({} as never);
    await adapter.generation!.generate!(
      track('kicks') as never,
      makeServices(patched) as never,
    );
    expect(patched[0].hasMidi).toBe(true);
    expect(patched[0].error).toBeNull();
  });

  it('seeds the piano roll with the notes it just wrote', async () => {
    const patched: Array<Record<string, unknown>> = [];
    const services = makeServices(patched);
    const adapter = createTimbreGraphAdapter({} as never);
    await adapter.generation!.generate!(track('bass') as never, services as never);

    const [, clip] = services.host.writeMidiClip.mock.calls[0] as unknown as [
      string,
      { notes: unknown[] },
    ];
    expect(patched[0].editNotes).toEqual(clip.notes);
    expect(patched[0].editBars).toBe(4);
    expect(patched[0].editBpm).toBe(120);
  });

  it('writes exactly one clip, spanning the scene', async () => {
    const patched: Array<Record<string, unknown>> = [];
    const services = makeServices(patched);
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

  /**
   * There used to be a fallback here: the training lab's probe pattern,
   * written whenever the model failed so a track was "never silent". It made
   * every failure inaudible AS a failure — six tracks would instantly play
   * probe MIDI and nothing said the LLM had not run. A failed generation must
   * now surface, exactly as it does on bass and pad.
   */
  it('surfaces a model failure instead of writing filler', async () => {
    const host = {
      ...makeHost(NOTES),
      generateWithLLM: jest.fn(async () => {
        throw new Error('offline');
      }),
    };
    const adapter = createTimbreGraphAdapter({} as never);
    await expect(
      adapter.generation!.generate!(track('leads') as never, services(host) as never),
    ).rejects.toThrow('offline');
    expect(host.writeMidiClip).not.toHaveBeenCalled();
  });

  it('surfaces unusable model content instead of writing filler', async () => {
    const host = makeHost('I am not JSON');
    const adapter = createTimbreGraphAdapter({} as never);
    await expect(
      adapter.generation!.generate!(track('pads') as never, services(host) as never),
    ).rejects.toThrow(/no usable notes/i);
    expect(host.generateWithLLM).toHaveBeenCalled();
    expect(host.writeMidiClip).not.toHaveBeenCalled();
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

describe('the lead plays arpeggios, at a rate that varies per generation', () => {
  const { buildTimbreSystemPrompt, leadSubdivision, LEAD_SUBDIVISIONS, ROLE_REQUEST } =
    require('../src/timbre-prompts');

  it('asks the lead for an arpeggio, not a held melody', () => {
    expect(ROLE_REQUEST.lead).toMatch(/arpeggio/i);
    const p = buildTimbreSystemPrompt('lead', '4/4', 0);
    expect(p).toMatch(/ARPEGGIO/);
    expect(p).toMatch(/one note at a time/i);
  });

  it('offers quarter, eighth and sixteenth rates', () => {
    expect(LEAD_SUBDIVISIONS.map((s: { beats: number }) => s.beats)).toEqual([1, 0.5, 0.25]);
  });

  it('rolls a different subdivision across seeds, deterministically', () => {
    const seen = new Set<number>();
    for (let s = 0; s < 30; s++) seen.add(leadSubdivision(s).beats);
    expect(seen.size).toBe(3);                       // all three are reachable
    expect(leadSubdivision(7)).toEqual(leadSubdivision(7));   // same seed, same roll
  });

  it('names the chosen rate in the prompt so the model cannot drift', () => {
    for (let s = 0; s < 3; s++) {
      const sub = leadSubdivision(s);
      expect(buildTimbreSystemPrompt('lead', '4/4', s)).toContain(sub.label);
    }
  });

  it('leaves the other five roles alone', () => {
    for (const role of ['kick', 'snare', 'hat', 'bass', 'pad']) {
      expect(buildTimbreSystemPrompt(role, '4/4', 1)).not.toMatch(/ARPEGGIO/);
    }
  });
});

describe('every timbre track gets a safety limiter', () => {
  /**
   * Two patches "started screaming" and hurt the user during a dial sweep.
   * Anchors are loudness-normalized at build time, but the space BETWEEN
   * them cannot be exhaustively enumerated, so the track carries a brickwall
   * as a last line. Since SDK 3.0.0 the host arms it by INTENT
   * (`applyManagedFxPreset(id, 'safety-limiter')`) — there is no FX category
   * or preset index left for this repo to keep in sync.
   */
  async function addGraph(host: Record<string, unknown>): Promise<void> {
    const adapter = createTimbreGraphAdapter(host as never);
    await adapter.onTrackCreated!(
      { id: 'engine-kick', name: 'timbre-kick', dbId: 'db-kick' } as never,
      {
        activeSceneId: 'scene-1',
        trackDataKey: (dbId: string, key: string) => `track:${dbId}:${key}`,
      } as never,
    );
  }

  function makeHost(calls: string[]): Record<string, unknown> {
    return {
      setTrackRole: async () => {},
      setSceneData: async () => {},
      applySurgeFxpPreset: async () => {},
      createTrack: async ({ role }: { role: string }) => ({
        id: `engine-${role}`, name: role, dbId: `db-${role}`,
      }),
      applyManagedFxPreset: async (id: string, intent: string) => {
        calls.push(`arm:${id}:${intent}`);
      },
    };
  }

  it('arms the brickwall on all six tracks at creation', async () => {
    const calls: string[] = [];
    await addGraph(makeHost(calls));
    const armed = calls.filter((c) => c.startsWith('arm:'));
    expect(armed).toHaveLength(6);
    expect(armed.every((c) => c.endsWith(':safety-limiter'))).toBe(true);
    expect(armed[0]).toBe('arm:engine-kick:safety-limiter'); // anchor first
  });

  it('still creates the group on an older host, and SAYS the guard is gone', async () => {
    const created: string[] = [];
    const host = {
      setTrackRole: async () => {},
      setSceneData: async () => {},
      applySurgeFxpPreset: async () => {},
      createTrack: async ({ role }: { role: string }) => {
        created.push(role);
        return { id: `engine-${role}`, name: role, dbId: `db-${role}` };
      },
      // no applyManagedFxPreset at all — a pre-3.0.0 host
    };
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      await addGraph(host as never);
      expect(created).toHaveLength(5);   // the five siblings still got made
      // degradation is visible, not silent: one warning per unprotected track
      const limiterWarns = warn.mock.calls.filter((c) =>
        /safety limiter/i.test(String(c[0])),
      );
      expect(limiterWarns).toHaveLength(6);
      expect(String(limiterWarns[0][0])).toMatch(/applyManagedFxPreset/);
    } finally {
      warn.mockRestore();
    }
  });

  it('an arm failure warns and continues — creation never aborts on the guard', async () => {
    const created: string[] = [];
    const host = {
      setTrackRole: async () => {},
      setSceneData: async () => {},
      applySurgeFxpPreset: async () => {},
      createTrack: async ({ role }: { role: string }) => {
        created.push(role);
        return { id: `engine-${role}`, name: role, dbId: `db-${role}` };
      },
      applyManagedFxPreset: async () => {
        throw new Error('engine busy');
      },
    };
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      await addGraph(host as never);
      expect(created).toHaveLength(5);
      const limiterWarns = warn.mock.calls.filter((c) =>
        /could not arm the safety limiter/i.test(String(c[0])),
      );
      expect(limiterWarns).toHaveLength(6);
      expect(String(limiterWarns[0][0])).toMatch(/painful levels/);
    } finally {
      warn.mockRestore();
    }
  });
});

describe('the limiter survives project reloads (re-arm on adoption passes)', () => {
  /**
   * The host strips ALL built-in FX — the ear-safety brickwall included —
   * from the project file on every load, so arming only at track creation
   * left every reopened project unprotected. The panel re-arms via
   * rearmSafetyLimiters on each adoption/discovery pass: panel-core's
   * loadTracks hands back FRESH handle objects per pass, while in-place row
   * patches (progress ticks, mixer churn) keep their handle references —
   * handle identity is the pass detector.
   */
  const { armSafetyLimiter, rearmSafetyLimiters } =
    require('../src/timbre-graph-adapter');

  const track = (id: string) => ({ handle: { id, dbId: `db-${id}` } });

  function armHost(calls: string[]): Record<string, unknown> {
    return {
      applyManagedFxPreset: async (id: string, intent: string) => {
        calls.push(`${id}:${intent}`);
      },
    };
  }

  it('arms every owned track once per discovery pass', () => {
    const calls: string[] = [];
    rearmSafetyLimiters(
      armHost(calls) as never,
      [track('e1'), track('e2'), track('e3')],
      new WeakSet(),
    );
    expect(calls).toEqual([
      'e1:safety-limiter',
      'e2:safety-limiter',
      'e3:safety-limiter',
    ]);
  });

  it('does not spam on in-place row patches (same handle objects)', () => {
    const calls: string[] = [];
    const host = armHost(calls) as never;
    const armed = new WeakSet<object>();
    const rows = [track('e1'), track('e2')];
    rearmSafetyLimiters(host, rows, armed);
    // a progress tick / mute toggle spreads the row but KEEPS t.handle —
    // running the pass again with the same handles must send nothing
    rearmSafetyLimiters(host, rows.map((r) => ({ ...r })), armed);
    rearmSafetyLimiters(host, rows, armed);
    expect(calls).toHaveLength(2);
  });

  it('re-arms when a pass hands back fresh handles — the project-reload shape', () => {
    const calls: string[] = [];
    const host = armHost(calls) as never;
    const armed = new WeakSet<object>();
    rearmSafetyLimiters(host, [track('e1'), track('e2')], armed);
    // reopening a project: onEngineReady → loadTracks → getPluginTracks
    // returns NEW handle objects for the SAME engine ids, and the host has
    // just stripped the limiter — this pass is the one that must fire again
    rearmSafetyLimiters(host, [track('e1'), track('e2')], armed);
    expect(calls).toEqual([
      'e1:safety-limiter', 'e2:safety-limiter',
      'e1:safety-limiter', 'e2:safety-limiter',
    ]);
  });

  it('warns and continues on an older host instead of throwing mid-render', () => {
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      expect(() =>
        rearmSafetyLimiters({} as never, [track('e1')], new WeakSet()),
      ).not.toThrow();
      expect(warn).toHaveBeenCalledTimes(1);
      expect(String(warn.mock.calls[0][0])).toMatch(/safety limiter/i);
    } finally {
      warn.mockRestore();
    }
  });

  it('a rejected arm warns and resolves', async () => {
    const warn = jest.spyOn(console, 'warn').mockImplementation(() => {});
    try {
      await armSafetyLimiter(
        {
          applyManagedFxPreset: async () => {
            throw new Error('no such track');
          },
        } as never,
        'e9',
      );
      expect(warn).toHaveBeenCalledTimes(1);
      expect(String(warn.mock.calls[0][0])).toContain('e9');
      expect(String(warn.mock.calls[0][0])).toMatch(/painful levels/);
    } finally {
      warn.mockRestore();
    }
  });

  it('the panel wires the re-arm into its track-discovery effect', () => {
    // The effect itself needs the full panel-core host surface to mount, so
    // pin the wiring at the source level: the helper must be called from
    // TimbreGraphPanel (the discovery seam), not only from track creation.
    const src = require('fs').readFileSync(
      require('path').join(__dirname, '..', 'TimbreGraphPanel.tsx'),
      'utf8',
    );
    expect(src).toMatch(/rearmSafetyLimiters\(/);
    expect(src).toMatch(/core\.tracks/);
  });
});

describe('layered mode: one role, one part, six different worlds', () => {
  /**
   * The point of the mode. Six tracks share a role and a MIDI part, so the
   * ONLY thing distinguishing them is the lens they are heard through. If the
   * panel wrote one set of parameters per role they would be six identical
   * voices, and the stack would be a volume boost rather than a sound.
   */
  const threeWorlds = {
    version: 'map-graph-v2',
    roles: {
      bass: {
        role: 'bass',
        lenses: [0, 1, 2].map((w) => ({
          lens: { preset_id: `L${w}`, name: `world ${w}`, category: 'Basses' },
          param_names: ['a'],
          points: [
            { preset_id: `p${w}0`, name: `p${w}0`, fxp_path: `${w}a.fxp`, xy: [0, 0] },
            { preset_id: `p${w}1`, name: `p${w}1`, fxp_path: `${w}b.fxp`, xy: [1, 1] },
          ],
          // each world moves the parameter by a different amount
          snapshots: [[0.1 * w], [0.1 * w + 0.1 * (w + 1)]],
          sharpness: 12, neighbours: 4, snap: 0.9,
        })),
        declined: false,
      },
    },
  };

  async function drive(
    targets: Array<{ id: string; lensIndex: number }>,
  ): Promise<Array<[string, Record<string, number>]>> {
    const writes: Array<[string, Record<string, number>]> = [];
    const host = {
      getProjectData: async (k: string) => (k.endsWith('map') ? threeWorlds : null),
      setProjectData: async () => {},
      setSynthParameters: async (id: string, params: Record<string, number>) => {
        writes.push([id, params]);
      },
    };
    const { container, cleanup } = renderIn(
      createElement(MorphSection, {
        host, activeSceneId: 's1',
        resolveTrackIds: (role: string) => (role === 'bass' ? targets : []),
        onTracksChanged: () => {},
      } as never),
    );
    await act(async () => Promise.resolve());
    await settleAt(container, [1, 1]);
    cleanup();
    return writes;
  }

  it('gives each layer its own world, so the stack is not one voice', async () => {
    const writes = await drive([
      { id: 'L0', lensIndex: 0 },
      { id: 'L1', lensIndex: 1 },
      { id: 'L2', lensIndex: 2 },
    ]);
    expect(writes).toHaveLength(3);
    const by = Object.fromEntries(writes);
    // world w moves the param by 0.1*(w+1), so all three differ
    expect(by['L0']['a']).toBeCloseTo(0.1, 6);
    expect(by['L1']['a']).toBeCloseTo(0.2, 6);
    expect(by['L2']['a']).toBeCloseTo(0.3, 6);
  });

  it('wraps when there are more layers than worlds, rather than piling up', async () => {
    // six layers over three worlds: 3,4,5 must wrap to 0,1,2 — clamping would
    // put half the stack on the same lens and waste them
    const writes = await drive(
      [0, 1, 2, 3, 4, 5].map((i) => ({ id: `L${i}`, lensIndex: i })),
    );
    const by = Object.fromEntries(writes);
    expect(by['L3']['a']).toBeCloseTo(by['L0']['a'], 6);
    expect(by['L4']['a']).toBeCloseTo(by['L1']['a'], 6);
    expect(by['L5']['a']).toBeCloseTo(by['L2']['a'], 6);
  });

  it('still writes one set of parameters per world, not per track', async () => {
    // two layers sharing a world must receive identical parameters
    const writes = await drive([
      { id: 'A', lensIndex: 1 },
      { id: 'B', lensIndex: 1 },
    ]);
    expect(writes).toHaveLength(2);
    expect(writes[0][1]).toEqual(writes[1][1]);
  });
});

describe('switching a group between ensemble and layered', () => {
  const { TimbreGroupRow } = require('../src/TimbreGroupRow') as {
    TimbreGroupRow: React.ComponentType<Record<string, unknown>>;
  };

  const member = (i: number, role: string) => ({
    dbId: `db-${i}`,
    meta: { groupId: 'g', memberIndex: i, role },
    track: { handle: { id: `eng-${i}` }, hasMidi: true, isGenerating: false },
  });

  function makeCtx() {
    return {
      collapsed: false,
      onToggleCollapse: () => {},
      handlers: { generate: () => {} },
      deleteGroup: async () => {},
      services: { host: { getTrackInfo: async () => ({ hasMidi: true }) },
                  activeSceneId: 'scene-1' },
      renderDefaultTrackRow: () => null,
    };
  }

  function mount(props: Record<string, unknown>) {
    const container = document.createElement('div');
    document.body.appendChild(container);
    const root = createRoot(container);
    act(() => {
      root.render(createElement(TimbreGroupRow, {
        group: { groupId: 'g', members: [member(0, 'kick'), member(1, 'snare')] },
        ctx: makeCtx(),
        ...props,
      }));
    });
    return { container, cleanup: () => { act(() => root.unmount()); container.remove(); } };
  }

  it('offers the mode choice on the group header', () => {
    const { container, cleanup } = mount({ onModeChange: () => {} });
    const sel = container.querySelector('[data-testid="timbre-group-mode"]') as HTMLSelectElement;
    expect(sel).not.toBeNull();
    expect([...sel.options].map((o) => o.value)).toEqual(['ensemble', 'layered']);
    cleanup();
  });

  it('hides the role picker until layered is chosen', () => {
    const a = mount({ mode: 'ensemble', onModeChange: () => {} });
    expect(a.container.querySelector('[data-testid="timbre-group-role"]')).toBeNull();
    a.cleanup();
    const b = mount({ mode: 'layered', layerRole: 'bass', onModeChange: () => {} });
    expect(b.container.querySelector('[data-testid="timbre-group-role"]')).not.toBeNull();
    b.cleanup();
  });

  it('reports the chosen mode, keeping the current role', () => {
    const seen: Array<[string, string]> = [];
    const { container, cleanup } = mount({
      mode: 'ensemble', layerRole: 'lead',
      onModeChange: (m: string, r: string) => seen.push([m, r]),
    });
    const sel = container.querySelector('[data-testid="timbre-group-mode"]') as HTMLSelectElement;
    act(() => {
      const set = Object.getOwnPropertyDescriptor(
        window.HTMLSelectElement.prototype, 'value',
      )?.set;
      set?.call(sel, 'layered');
      sel.dispatchEvent(new Event('change', { bubbles: true }));
    });
    expect(seen).toEqual([['layered', 'lead']]);
    cleanup();
  });

  it('reports a role change while staying layered', () => {
    const seen: Array<[string, string]> = [];
    const { container, cleanup } = mount({
      mode: 'layered', layerRole: 'bass',
      onModeChange: (m: string, r: string) => seen.push([m, r]),
    });
    const sel = container.querySelector('[data-testid="timbre-group-role"]') as HTMLSelectElement;
    act(() => {
      const set = Object.getOwnPropertyDescriptor(
        window.HTMLSelectElement.prototype, 'value',
      )?.set;
      set?.call(sel, 'lead');
      sel.dispatchEvent(new Event('change', { bubbles: true }));
    });
    expect(seen).toEqual([['layered', 'lead']]);
    cleanup();
  });

  it('stays out of the way when the host cannot switch modes', () => {
    // no onModeChange (older shell): the header renders without the controls
    const { container, cleanup } = mount({});
    expect(container.querySelector('[data-testid="timbre-group-mode"]')).toBeNull();
    expect(container.textContent).toContain('Timbre Graph');
    cleanup();
  });
});

describe('layered groups generate ONE part, not six', () => {
  const { TimbreGroupRow } = require('../src/TimbreGroupRow') as {
    TimbreGroupRow: React.ComponentType<Record<string, unknown>>;
  };

  const member = (i: number) => ({
    dbId: `db-${i}`,
    meta: { groupId: 'g', memberIndex: i, role: 'bass' },
    track: { handle: { id: `eng-${i}` }, hasMidi: true, isGenerating: false },
  });

  function mount(mode: string) {
    const generated: string[] = [];
    const container = document.createElement('div');
    document.body.appendChild(container);
    const root = createRoot(container);
    act(() => {
      root.render(createElement(TimbreGroupRow, {
        group: { groupId: 'g', members: [0, 1, 2, 3].map(member) },
        ctx: {
          collapsed: false, onToggleCollapse: () => {},
          handlers: { generate: (id: string) => generated.push(id) },
          deleteGroup: async () => {},
          services: { host: { getTrackInfo: async () => ({ hasMidi: true }) },
                      activeSceneId: 's1' },
          renderDefaultTrackRow: () => null,
        },
        mode, layerRole: 'bass', onModeChange: () => {},
      }));
    });
    const btn = container.querySelector(
      '[data-testid="timbre-group-generate-all"]',
    ) as HTMLButtonElement;
    return { btn, generated, container,
             cleanup: () => { act(() => root.unmount()); container.remove(); } };
  }

  it('an ensemble generates a part per member', () => {
    const { btn, generated, cleanup } = mount('ensemble');
    act(() => btn.dispatchEvent(new MouseEvent('click', { bubbles: true })));
    expect(generated).toEqual(['eng-0', 'eng-1', 'eng-2', 'eng-3']);
    cleanup();
  });

  it('a layered stack generates ONCE, from the anchor', () => {
    // six independently generated bass lines would fight rather than thicken,
    // and cost six LLM calls to produce the mess
    const { btn, generated, cleanup } = mount('layered');
    act(() => btn.dispatchEvent(new MouseEvent('click', { bubbles: true })));
    expect(generated).toEqual(['eng-0']);
    cleanup();
  });

  it('says what it will do', () => {
    const a = mount('ensemble');
    expect(a.container.textContent).toContain('Generate All');
    a.cleanup();
    const b = mount('layered');
    expect(b.container.textContent).toContain('Generate Part');
    b.cleanup();
  });
});

describe('a layered group survives reopening the project', () => {
  const { primeGroupConfigs } = require('../src/timbre-graph-adapter');
  const { TimbreGroupRow } = require('../src/TimbreGroupRow') as {
    TimbreGroupRow: React.ComponentType<Record<string, unknown>>;
  };

  /**
   * The mode is persisted to scene data, and for a while nothing read it back:
   * the cache was write-only, so reopening showed a layered group as
   * `ensemble`. The pad still drove each track through its own lens (that comes
   * from the members' own meta), but the dropdown lied and Generate would write
   * six competing parts instead of one.
   */
  it('restores the mode and role from scene data', () => {
    primeGroupConfigs({
      'group:g1:timbreGroupConfig': { mode: 'layered', role: 'lead' },
    });
    const { createTimbreGraphAdapter } = require('../src/timbre-graph-adapter');
    const adapter = createTimbreGraphAdapter({} as never);
    const spec = adapter.groupExtensions![0];
    const el = spec.renderGroup!(
      { groupId: 'g1', members: [] } as never,
      { services: { activeSceneId: 's1' }, collapsed: false,
        handlers: {}, deleteGroup: async () => {},
        renderDefaultTrackRow: () => null } as never,
    ) as { props: Record<string, unknown> };
    expect(el.props.mode).toBe('layered');
    expect(el.props.layerRole).toBe('lead');
  });

  it('leaves an untouched group as an ensemble', () => {
    primeGroupConfigs({});
    const { createTimbreGraphAdapter } = require('../src/timbre-graph-adapter');
    const adapter = createTimbreGraphAdapter({} as never);
    const el = adapter.groupExtensions![0].renderGroup!(
      { groupId: 'never-configured', members: [] } as never,
      { services: { activeSceneId: 's1' }, collapsed: false,
        handlers: {}, deleteGroup: async () => {},
        renderDefaultTrackRow: () => null } as never,
    ) as { props: Record<string, unknown> };
    expect(el.props.mode).toBe('ensemble');
  });

  it('ignores foreign and malformed scene-data keys', () => {
    expect(() => primeGroupConfigs({
      'track:db-1:timbreGroup': { groupId: 'g', memberIndex: 0, role: 'kick' },
      'group:g2:timbreGroupConfig': null,
      'group:g3:timbreGroupConfig': { mode: 'nonsense', role: 'zither' },
    })).not.toThrow();
    const { createTimbreGraphAdapter } = require('../src/timbre-graph-adapter');
    const adapter = createTimbreGraphAdapter({} as never);
    const el = adapter.groupExtensions![0].renderGroup!(
      { groupId: 'g3', members: [] } as never,
      { services: { activeSceneId: 's1' }, collapsed: false,
        handlers: {}, deleteGroup: async () => {},
        renderDefaultTrackRow: () => null } as never,
    ) as { props: Record<string, unknown> };
    // garbage narrows to the safe default, it does not propagate
    expect(el.props.mode).toBe('ensemble');
    expect(el.props.layerRole).toBe('bass');
  });
});
