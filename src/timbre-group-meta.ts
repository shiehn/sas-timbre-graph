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
  /** dbId of the anchor (kick) track. */
  groupId: string;
  memberIndex: number;
  role: TimbreRole;
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
  return { groupId: m.groupId, memberIndex: m.memberIndex, role };
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
