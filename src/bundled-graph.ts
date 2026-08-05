/**
 * The patch map that SHIPS with the plugin.
 *
 * Requiring an end user to run a Python training CLI before the pad works is
 * not a product; the lab is for RE-building, and its output is shipped as an
 * asset. Imported statically so it is bundled into dist by tsup (no runtime
 * file IO, no packaging path to get wrong).
 *
 * Anchor presets are referenced by path RELATIVE to the Surge content root
 * ("Percussion/Kick Tech 2.fxp"), which the host resolves against this
 * machine's installed library — an absolute path from the authoring machine
 * would not exist anywhere else.
 */

import graph from '../assets/patchmap.json';
import type { MapGraph } from './patchmap';

export const BUNDLED_GRAPH = graph as unknown as MapGraph;
