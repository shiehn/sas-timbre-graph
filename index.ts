/**
 * Plugin entry point — @signalsandsorcery/timbre-graph
 *
 * Timbre Graph hosts six fixed Surge XT tracks (Kick, Snare, Hat, Bass,
 * Chord Pad, Lead) and moves all six patches together through one
 * perceptual control surface (an X/Y morph pad). The heavy lifting — the
 * forward/delta timbre model and the follower solver — is trained offline
 * in training/ and consumed here as a precomputed morph graph artifact.
 *
 * The panel is currently a stub: the training phase comes first. See
 * docs/TRAINING.md for the amended plan and training runbook.
 */

import type { ComponentType } from 'react';
import type {
  GeneratorPlugin,
  GeneratorType,
  PluginHost,
  PluginUIProps,
  PluginSettingsSchema,
} from '@signalsandsorcery/plugin-sdk';
import { TimbreGraphPanel } from './TimbreGraphPanel';
import manifest from './plugin.json';

export class TimbreGraphPlugin implements GeneratorPlugin {
  readonly id = '@signalsandsorcery/timbre-graph';
  readonly displayName = 'Timbre Graph';
  readonly version = '0.1.0';
  readonly description =
    'Six coordinated Surge XT tracks morphed together through one perceptual control surface';

  readonly generatorType: GeneratorType = 'midi';

  private host: PluginHost | null = null;

  async activate(host: PluginHost): Promise<void> {
    this.host = host;
  }

  async deactivate(): Promise<void> {
    this.host = null;
  }

  getUIComponent(): ComponentType<PluginUIProps> {
    return TimbreGraphPanel;
  }

  getSettingsSchema(): PluginSettingsSchema | null {
    return null;
  }
}

export default TimbreGraphPlugin;
export { TimbreGraphPanel };
/** sas-app's `src/plugins/index.ts` imports the class AND manifest from here. */
export const timbreGraphManifest = manifest;
