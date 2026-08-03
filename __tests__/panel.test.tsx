import { createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react-dom/test-utils';
import TimbreGraphPlugin, { timbreGraphManifest } from '../index';
import { TimbreGraphPanel, paramsAt, reachableDirections } from '../TimbreGraphPanel';
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
      createElement(TimbreGraphPanel, { host: {} } as never),
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
      createElement(TimbreGraphPanel, { host } as never),
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
