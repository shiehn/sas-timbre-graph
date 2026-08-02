/**
 * Timbre Graph panel — stub.
 *
 * Final shape: six always-visible Surge XT tracks (Kick, Snare, Hat, Bass,
 * Chord Pad, Lead) plus a morph surface. Dragging the morph point moves all
 * linked patches along perceptually coherent parameter trajectories computed
 * from the trained forward/delta timbre model. Each track row gets a
 * link/unlink toggle so a loved patch can be frozen or hand-tweaked while
 * the rest keep morphing.
 */

import type { PluginUIProps } from '@signalsandsorcery/plugin-sdk';

const ROLES = ['Kick', 'Snare', 'Hat', 'Bass', 'Chord Pad', 'Lead'] as const;

export function TimbreGraphPanel(_props: PluginUIProps) {
  return (
    <div style={{ padding: 12, fontSize: 12, opacity: 0.85 }}>
      <div style={{ fontWeight: 600, marginBottom: 8 }}>Timbre Graph</div>
      <div style={{ marginBottom: 8 }}>
        Training phase in progress — see <code>training/</code>. This panel will
        host six coupled Surge XT tracks and a perceptual morph surface.
      </div>
      <ul style={{ margin: 0, paddingLeft: 16 }}>
        {ROLES.map((role) => (
          <li key={role}>{role}</li>
        ))}
      </ul>
    </div>
  );
}
