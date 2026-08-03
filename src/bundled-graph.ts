/**
 * The morph graph that SHIPS with the plugin.
 *
 * Requiring an end user to run a Python training CLI before the dial works is
 * not a product; the lab is for RE-training, and its output is shipped as an
 * asset. Imported statically so it is bundled into dist by tsup (no runtime
 * file IO, no packaging path to get wrong).
 *
 * Anchor presets are referenced by path RELATIVE to the Surge content root
 * ("Percussion/Kick 909ish.fxp"), which the host resolves against this
 * machine's installed library — an absolute path from the authoring machine
 * would not exist anywhere else.
 */

import graph from '../assets/morph-softer.json';

export const BUNDLED_GRAPH = graph as unknown as {
  version: string;
  axis: { name: string; vector: number[] };
  control_points: number[];
  roles: Record<string, {
    role: string;
    preset_id: string;
    fxp_path?: string;
    name: string;
    param_names: string[];
    baseline: number[];
    snapshots: number[][];
    cosine: number[];
    declined: boolean;
  }>;
  quality?: Record<string, unknown>;
};
