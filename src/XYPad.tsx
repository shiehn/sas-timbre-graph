/**
 * The patch map's control surface.
 *
 * A canvas rather than a pair of sliders because the map IS a picture: the
 * anchors are laid out by how they sound, so seeing them is how you navigate
 * ("that cluster over there is the bright ones"). Dots are drawn unlabelled —
 * this is a search surface, and naming every point would turn it back into a
 * preset browser. The nearest patch names itself only while you hover, so
 * curiosity is answerable without the labels being in the way.
 *
 * Mouse events, not pointer events: this runs in an Electron desktop app where
 * that is sufficient, and jsdom implements MouseEvent natively while
 * PointerEvent needs a shim (the piano-roll editor learned this the hard way).
 */

import { useCallback, useEffect, useRef } from 'react';

export interface XYPadPoint {
  xy: [number, number];
  name: string;
}

const DOT = 3.5;
const PUCK = 7;

export function XYPad({
  points,
  value,
  onChange,
  accent = '#2DD4BF',
  size = 220,
  hovered,
  onHover,
  onRelease,
}: {
  points: readonly XYPadPoint[];
  value: readonly [number, number];
  onChange: (xy: [number, number]) => void;
  accent?: string;
  size?: number;
  /** Index of the point named in the caption, or null. */
  hovered: number | null;
  onHover: (index: number | null) => void;
  /**
   * Fired when the gesture ends. The panel uses it to SETTLE: while dragging
   * the surface is continuous and exploratory, and on release it resolves to
   * the nearest checked point — so searching sounds like searching and landing
   * sounds solid, and the safety nudge happens exactly when the user has
   * stopped to listen.
   */
  onRelease?: (xy: [number, number]) => void;
}) {
  const canvas = useRef<HTMLCanvasElement | null>(null);
  const dragging = useRef(false);

  const draw = useCallback(() => {
    const el = canvas.current;
    if (!el) return;
    const ctx = el.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    if (el.width !== size * dpr) {
      el.width = size * dpr;
      el.height = size * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    ctx.fillStyle = 'rgba(127,127,127,0.10)';
    ctx.fillRect(0, 0, size, size);

    points.forEach((p, i) => {
      const [x, y] = p.xy;
      ctx.beginPath();
      ctx.arc(x * size, (1 - y) * size, i === hovered ? DOT + 2 : DOT, 0, Math.PI * 2);
      ctx.fillStyle = i === hovered ? accent : 'rgba(200,200,200,0.55)';
      ctx.fill();
    });

    const [vx, vy] = value;
    ctx.beginPath();
    ctx.arc(vx * size, (1 - vy) * size, PUCK, 0, Math.PI * 2);
    ctx.strokeStyle = accent;
    ctx.lineWidth = 2;
    ctx.stroke();
  }, [points, value, hovered, accent, size]);

  useEffect(() => {
    draw();
  }, [draw]);

  /** Canvas pixels -> unit square, clamped: the puck cannot leave the map. */
  const toUnit = useCallback(
    (e: { clientX: number; clientY: number }): [number, number] => {
      const el = canvas.current;
      if (!el) return [0, 0];
      const r = el.getBoundingClientRect();
      const x = Math.min(1, Math.max(0, (e.clientX - r.left) / (r.width || size)));
      const y = Math.min(1, Math.max(0, (e.clientY - r.top) / (r.height || size)));
      return [x, 1 - y];
    },
    [size],
  );

  const nearest = useCallback(
    (xy: [number, number]): number | null => {
      let best: number | null = null;
      let bd = Infinity;
      points.forEach((p, i) => {
        const d = Math.hypot(p.xy[0] - xy[0], p.xy[1] - xy[1]);
        if (d < bd) {
          bd = d;
          best = i;
        }
      });
      return best;
    },
    [points],
  );

  return (
    <canvas
      ref={canvas}
      data-testid="timbre-xy-pad"
      aria-label="patch map"
      role="application"
      style={{
        width: size, height: size, borderRadius: 6, cursor: 'crosshair',
        border: '1px solid rgba(127,127,127,0.3)', touchAction: 'none',
      }}
      onMouseDown={(e) => {
        dragging.current = true;
        const xy = toUnit(e);
        onChange(xy);
        onHover(nearest(xy));
      }}
      onMouseMove={(e) => {
        const xy = toUnit(e);
        onHover(nearest(xy));
        if (dragging.current) onChange(xy);
      }}
      onMouseUp={(e) => {
        if (dragging.current) onRelease?.(toUnit(e));
        dragging.current = false;
      }}
      // `mouseout`, not `mouseleave`: React synthesises leave from mouseout
      // delegation, and a canvas has no children for the two to differ over.
      // Releasing the drag here also covers letting go outside the pad.
      onMouseOut={(e) => {
        if (dragging.current) onRelease?.(toUnit(e));
        dragging.current = false;
        onHover(null);
      }}
    />
  );
}
