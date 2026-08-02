import { createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react-dom/test-utils';
import TimbreGraphPlugin from '../index';
import { TimbreGraphPanel } from '../TimbreGraphPanel';
import pluginJson from '../plugin.json';

describe('TimbreGraphPlugin', () => {
  it('keeps class metadata in sync with plugin.json', () => {
    const plugin = new TimbreGraphPlugin();
    expect(plugin.id).toBe(pluginJson.id);
    expect(plugin.displayName).toBe(pluginJson.displayName);
    expect(plugin.version).toBe(pluginJson.version);
    expect(plugin.generatorType).toBe(pluginJson.generatorType);
  });

  it('exposes the panel component', () => {
    const plugin = new TimbreGraphPlugin();
    expect(plugin.getUIComponent()).toBe(TimbreGraphPanel);
  });

  it('renders the six fixed roles in the stub panel', () => {
    const container = document.createElement('div');
    document.body.appendChild(container);
    const root = createRoot(container);
    act(() => {
      root.render(createElement(TimbreGraphPanel, {} as never));
    });
    for (const role of ['Kick', 'Snare', 'Hat', 'Bass', 'Chord Pad', 'Lead']) {
      expect(container.textContent).toContain(role);
    }
    act(() => root.unmount());
    container.remove();
  });
});
