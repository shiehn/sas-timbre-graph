/**
 * The pad's control behaviour. Canvas drawing is not asserted (jsdom has no
 * real 2-D context); what matters here is the mapping from a click to a
 * position, and that the puck can never leave the validated square.
 */

import { createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { act } from 'react-dom/test-utils';
import { XYPad } from '../src/XYPad';

const POINTS = [
  { xy: [0.1, 0.1] as [number, number], name: 'low left' },
  { xy: [0.9, 0.9] as [number, number], name: 'high right' },
  { xy: [0.5, 0.5] as [number, number], name: 'middle' },
];

function mount(props: Record<string, unknown>) {
  const container = document.createElement('div');
  document.body.appendChild(container);
  const root = createRoot(container);
  act(() => {
    root.render(createElement(XYPad, { points: POINTS, ...props } as never));
  });
  const el = container.querySelector('canvas') as HTMLCanvasElement;
  // jsdom gives every element a zero-sized rect; the pad reads it to convert
  // pixels to unit coordinates, so it needs a real one
  el.getBoundingClientRect = () =>
    ({ left: 0, top: 0, width: 200, height: 200, right: 200, bottom: 200,
       x: 0, y: 0, toJSON: () => ({}) }) as DOMRect;
  return {
    el,
    cleanup: () => { act(() => root.unmount()); container.remove(); },
  };
}

function drag(el: HTMLCanvasElement, type: string, clientX: number, clientY: number) {
  act(() => {
    el.dispatchEvent(new MouseEvent(type, { clientX, clientY, bubbles: true }));
  });
}

describe('XYPad', () => {
  it('maps a click to a position, with Y pointing up', () => {
    const seen: Array<[number, number]> = [];
    const { el, cleanup } = mount({
      value: [0.5, 0.5], onChange: (v: [number, number]) => seen.push(v),
      hovered: null, onHover: () => {},
    });
    // top-left of the canvas is x=0, y=1 — screens grow downward, maps do not
    drag(el, 'mousedown', 0, 0);
    expect(seen[0][0]).toBeCloseTo(0);
    expect(seen[0][1]).toBeCloseTo(1);

    drag(el, 'mousedown', 100, 100);
    expect(seen[1][0]).toBeCloseTo(0.5);
    expect(seen[1][1]).toBeCloseTo(0.5);
    cleanup();
  });

  it('clamps to the map: the puck cannot leave validated ground', () => {
    const seen: Array<[number, number]> = [];
    const { el, cleanup } = mount({
      value: [0.5, 0.5], onChange: (v: [number, number]) => seen.push(v),
      hovered: null, onHover: () => {},
    });
    drag(el, 'mousedown', -500, -500);
    drag(el, 'mousedown', 9999, 9999);
    for (const [x, y] of seen) {
      expect(x).toBeGreaterThanOrEqual(0);
      expect(x).toBeLessThanOrEqual(1);
      expect(y).toBeGreaterThanOrEqual(0);
      expect(y).toBeLessThanOrEqual(1);
    }
    cleanup();
  });

  it('only moves while the button is held', () => {
    const seen: Array<[number, number]> = [];
    const { el, cleanup } = mount({
      value: [0.5, 0.5], onChange: (v: [number, number]) => seen.push(v),
      hovered: null, onHover: () => {},
    });
    drag(el, 'mousemove', 20, 20);          // hover only
    expect(seen).toHaveLength(0);
    drag(el, 'mousedown', 40, 40);
    drag(el, 'mousemove', 60, 60);
    expect(seen).toHaveLength(2);
    drag(el, 'mouseup', 60, 60);
    drag(el, 'mousemove', 80, 80);          // released — no more writes
    expect(seen).toHaveLength(2);
    cleanup();
  });

  it('reports the nearest patch while hovering, and forgets it on leave', () => {
    const hovers: Array<number | null> = [];
    const { el, cleanup } = mount({
      value: [0.5, 0.5], onChange: () => {},
      hovered: null, onHover: (i: number | null) => hovers.push(i),
    });
    drag(el, 'mousemove', 20, 180);         // near (0.1, 0.1)
    expect(POINTS[hovers[0] as number].name).toBe('low left');
    drag(el, 'mousemove', 180, 20);         // near (0.9, 0.9)
    expect(POINTS[hovers[1] as number].name).toBe('high right');
    drag(el, 'mouseout', 0, 0);
    expect(hovers[hovers.length - 1]).toBeNull();
    cleanup();
  });

  it('stops dragging when the cursor leaves the pad', () => {
    const seen: Array<[number, number]> = [];
    const { el, cleanup } = mount({
      value: [0.5, 0.5], onChange: (v: [number, number]) => seen.push(v),
      hovered: null, onHover: () => {},
    });
    drag(el, 'mousedown', 40, 40);
    drag(el, 'mouseout', 0, 0);
    drag(el, 'mousemove', 90, 90);
    expect(seen).toHaveLength(1);
    cleanup();
  });
});
