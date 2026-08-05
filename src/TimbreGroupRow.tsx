/**
 * The Timbre Graph group row — six member tracks under one header.
 *
 * Header offers the group-level verbs: Generate All (each member's role
 * pattern), collapse, and Remove Group (two-click confirm; deletes all member
 * tracks plus their group scene-data keys via the core's deleteGroup).
 *
 * Member rows are the shell's own TrackRows via `ctx.renderDefaultTrackRow`
 * — the ~50-prop plumbing stays in panel-core. Prompts are hidden: this
 * family derives generation entirely from each track's role.
 */

import { useEffect, useState } from 'react';
import type {
  GeneratorTrackState,
  GroupRenderContext,
  ResolvedTrackGroup,
} from '@signalsandsorcery/plugin-sdk';
import { GroupCollapseChevron } from '@signalsandsorcery/plugin-sdk';
import type { TimbreGroupMeta, TimbreGroupMode } from './timbre-group-meta';
import { TIMBRE_GROUP_META_KEY } from './timbre-group-meta';
import { TIMBRE_ROLES, type TimbreRole } from './role-patterns';

const ACCENT = '#2DD4BF';

const ROLE_LABELS: Record<TimbreRole, string> = {
  kick: 'Kick', snare: 'Snare', hat: 'Hat',
  bass: 'Bass', pad: 'Chord Pad', lead: 'Lead',
};

export function TimbreGroupRow({
  group,
  ctx,
  mode = 'ensemble',
  layerRole = 'bass',
  onModeChange,
}: {
  group: ResolvedTrackGroup<TimbreGroupMeta, GeneratorTrackState>;
  ctx: GroupRenderContext;
  /**
   * `ensemble` — six roles, six parts, a band.
   * `layered`  — six copies of ONE role playing ONE part, each heard through a
   *   different structure. A sound-design instrument rather than an arrangement.
   */
  mode?: TimbreGroupMode;
  layerRole?: TimbreRole;
  onModeChange?: (mode: TimbreGroupMode, role: TimbreRole) => void;
}) {
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  const generateAll = () => {
    if (mode === 'layered') {
      /*
       * One call, not six. Every layer plays the SAME part — the adapter
       * copies it across after writing — so generating per member would spend
       * six LLM calls producing six competing lines and then throw five away.
       */
      const anchor = group.members.find((m) => m.meta.memberIndex === 0)
        ?? group.members[0];
      if (anchor) ctx.handlers.generate(anchor.track.handle.id);
      return;
    }
    for (const m of group.members) {
      ctx.handlers.generate(m.track.handle.id);
    }
  };

  const removeGroup = async () => {
    await ctx.deleteGroup(
      group.members.map((m) => ({
        engineId: m.track.handle.id,
        dbId: m.dbId,
      })),
      [TIMBRE_GROUP_META_KEY],
    );
  };

  const anyGenerating = group.members.some((m) => m.track.isGenerating);

  /**
   * Does any member ACTUALLY have MIDI on disk?
   *
   * Deliberately not `m.track.hasMidi`. panel-core derives that field with a
   * role-presence fallback ("if (!hasMidi && handle.role) hasMidi = true"),
   * and every track in this family is created with a role — so all six report
   * hasMidi the instant Add Graph makes them, and the gate below opened on
   * six empty rows. `host.getTrackInfo` is the truthful reading: it comes
   * straight from the tracks table's has_midi column.
   *
   * Re-queried whenever generation finishes (anyGenerating falling to false)
   * or the membership changes — the only two moments the answer can move.
   */
  const memberIds = group.members.map((m) => m.track.handle.id).join(',');
  const [membersHaveMidi, setMembersHaveMidi] = useState(false);
  const host = ctx.services.host;
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      for (const id of memberIds ? memberIds.split(',') : []) {
        try {
          const info = await host.getTrackInfo(id);
          if (cancelled) return;
          if (info?.hasMidi) {
            setMembersHaveMidi(true);
            return;
          }
        } catch {
          /* unreadable track can't prove MIDI — keep checking the rest */
        }
      }
      if (!cancelled) setMembersHaveMidi(false);
    })();
    return () => {
      cancelled = true;
    };
  }, [memberIds, anyGenerating, host]);

  /**
   * Member rows stay hidden until generation has begun. Six visible empty
   * tracks read as "press play" — and play nothing. Until there is MIDI, the
   * group is a single header with one obvious next step. `isGenerating` is
   * still part of the gate so the rows are present for the whole operation,
   * however brief it is.
   */
  const materialized = membersHaveMidi || anyGenerating;

  return (
    <div data-testid="timbre-group-row">
      <div
        style={{
          display: 'flex', alignItems: 'center', gap: 8,
          padding: '6px 10px', fontSize: 12,
          borderLeft: `3px solid ${ACCENT}`,
          background: 'rgba(45,212,191,0.07)',
        }}
      >
        <GroupCollapseChevron
          collapsed={ctx.collapsed}
          onToggle={ctx.onToggleCollapse}
          what="timbre graph"
        />
        <span style={{ fontWeight: 600 }}>Timbre Graph</span>
        <span style={{ opacity: 0.6 }}>
          {group.members.length} track{group.members.length === 1 ? '' : 's'}
        </span>
        {onModeChange && (
          <>
            <select
              aria-label="group mode"
              data-testid="timbre-group-mode"
              value={mode}
              onChange={(e) => onModeChange(e.target.value as TimbreGroupMode, layerRole)}
              style={{ font: 'inherit', fontSize: 11 }}
            >
              <option value="ensemble">ensemble</option>
              <option value="layered">layered</option>
            </select>
            {mode === 'layered' && (
              <select
                aria-label="layered role"
                data-testid="timbre-group-role"
                value={layerRole}
                onChange={(e) => onModeChange(mode, e.target.value as TimbreRole)}
                style={{ font: 'inherit', fontSize: 11 }}
              >
                {TIMBRE_ROLES.map((r) => (
                  <option key={r} value={r}>{ROLE_LABELS[r]}</option>
                ))}
              </select>
            )}
          </>
        )}
        <span style={{ flex: 1 }} />
        <button
          type="button"
          onClick={generateAll}
          disabled={anyGenerating}
          data-testid="timbre-group-generate-all"
          style={{
            font: 'inherit', fontSize: 11, padding: '3px 10px',
            borderRadius: 4, border: `1px solid ${ACCENT}`,
            background: 'transparent', cursor: anyGenerating ? 'default' : 'pointer',
            opacity: anyGenerating ? 0.5 : 1,
          }}
        >
          {anyGenerating
            ? 'Generating…'
            : mode === 'layered' ? 'Generate Part' : 'Generate All'}
        </button>
        {confirmingDelete ? (
          <button
            type="button"
            onClick={() => void removeGroup()}
            data-testid="timbre-group-delete-confirm"
            style={{
              font: 'inherit', fontSize: 11, padding: '3px 8px',
              borderRadius: 4, border: '1px solid #d66', color: '#d66',
              background: 'transparent', cursor: 'pointer',
            }}
          >
            Remove {group.members.length} tracks?
          </button>
        ) : (
          <button
            type="button"
            onClick={() => setConfirmingDelete(true)}
            title="Remove this timbre-graph group"
            data-testid="timbre-group-delete"
            style={{
              font: 'inherit', fontSize: 12, background: 'none',
              border: 'none', cursor: 'pointer', opacity: 0.55,
            }}
          >
            ✕
          </button>
        )}
      </div>
      {!materialized ? (
        <div
          data-testid="timbre-group-unmaterialized"
          style={{ padding: '8px 12px', fontSize: 12, opacity: 0.7 }}
        >
          {group.members.length} tracks ready — press <b>Generate All</b> to
          write their patterns and hear them.
        </div>
      ) : (
        !ctx.collapsed &&
        group.members.map((m) =>
          // Prompts hidden: generation is role-derived, nothing to type.
          ctx.renderDefaultTrackRow(m.track, { onPromptChange: undefined }),
        )
      )}
    </div>
  );
}
