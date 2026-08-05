/**
 * Timbre-graph group metadata — the SDK's generic group seam (bass pattern).
 *
 * A timbre group is SIX normal tracks (kick/snare/hat/bass/pad/lead) linked by
 * per-member scene-data keys `track:<dbId>:timbreGroup` sharing a groupId.
 * The ANCHOR is member 0 (kick); `groupId === anchorDbId`. Keys are DB-UUID
 * based, never engine ids.
 */

import type {
  GeneratorTrackState,
  GroupParseSpec,
  ResolvedTrackGroup,
} from '@signalsandsorcery/plugin-sdk';
import { TIMBRE_ROLES, type TimbreRole } from './role-patterns';

export const TIMBRE_GROUP_META_KEY = 'timbreGroup';

export interface TimbreGroupMeta {
  /** dbId of the anchor (member 0) track. */
  groupId: string;
  memberIndex: number;
  role: TimbreRole;
  /**
   * Which WORLD this member explores — an index into its role's lens list.
   *
   * In a LAYERED group every member shares one role and one MIDI part, so the
   * only thing distinguishing them is the structure they are heard through.
   * Six positions inside one lens would just be six variations of one patch;
   * six different lenses are six genuinely different instruments playing in
   * unison. Absent (or 0) for the classic ensemble, where members differ by
   * role instead.
   */
  lensIndex?: number;
}

/**
 * What a group IS.
 *
 * `ensemble` — the original: six roles, six parts, one band.
 * `layered`  — six copies of ONE role playing ONE part, each on its own lens,
 *   stacked. The interesting mode for sound design rather than arrangement.
 */
export type TimbreGroupMode = 'ensemble' | 'layered';

/** Stored once per group, on the anchor member. */
export interface TimbreGroupConfig {
  mode: TimbreGroupMode;
  /** The single role every member plays, when mode is 'layered'. */
  role: TimbreRole;
}

export const TIMBRE_GROUP_CONFIG_KEY = 'timbreGroupConfig';

export function asTimbreGroupConfig(val: unknown): TimbreGroupConfig | null {
  if (!val || typeof val !== 'object') return null;
  const c = val as Partial<TimbreGroupConfig>;
  const mode: TimbreGroupMode = c.mode === 'layered' ? 'layered' : 'ensemble';
  const role = (TIMBRE_ROLES as readonly string[]).includes(c.role as string)
    ? (c.role as TimbreRole)
    : 'bass';
  return { mode, role };
}

/** Defensive narrow — survives partial/foreign blobs. */
export function asTimbreGroupMeta(val: unknown): TimbreGroupMeta | null {
  if (!val || typeof val !== 'object') return null;
  const m = val as Partial<TimbreGroupMeta>;
  if (typeof m.groupId !== 'string' || typeof m.memberIndex !== 'number') {
    return null;
  }
  const role = (TIMBRE_ROLES as readonly string[]).includes(m.role as string)
    ? (m.role as TimbreRole)
    : TIMBRE_ROLES[Math.min(Math.max(m.memberIndex, 0), TIMBRE_ROLES.length - 1)];
  const lensIndex =
    typeof m.lensIndex === 'number' && m.lensIndex >= 0
      ? Math.floor(m.lensIndex)
      : undefined;
  return { groupId: m.groupId, memberIndex: m.memberIndex, role, lensIndex };
}

export const timbreGroupSpec: GroupParseSpec<TimbreGroupMeta> = {
  metaKey: TIMBRE_GROUP_META_KEY,
  asMeta: asTimbreGroupMeta,
  groupIdOf: (m) => m.groupId,
  sortMembers: (a, b) => a.meta.memberIndex - b.meta.memberIndex,
};

/**
 * Completeness: the anchor must be live. A group missing a non-anchor member
 * stays a group (thinner ensemble); one missing its anchor degrades to
 * normal rows.
 */
export function timbreGroupIsComplete(
  group: ResolvedTrackGroup<TimbreGroupMeta, GeneratorTrackState>,
): boolean {
  return group.members.some((m) => m.meta.memberIndex === 0);
}
