/**
 * The Timbre Graph group row — six member tracks under one header.
 *
 * Header offers the group-level verbs: Generate All (each member's
 * deterministic role pattern, with the standard per-row progress bars),
 * collapse, and Remove Group (two-click confirm; deletes all member tracks
 * plus their group scene-data keys via the core's deleteGroup).
 *
 * Member rows are the shell's own TrackRows via `ctx.renderDefaultTrackRow`
 * — the ~50-prop plumbing stays in panel-core. Prompts are hidden: this
 * family derives generation entirely from each track's role.
 */

import { useState } from 'react';
import type {
  GeneratorTrackState,
  GroupRenderContext,
  ResolvedTrackGroup,
} from '@signalsandsorcery/plugin-sdk';
import { GroupCollapseChevron } from '@signalsandsorcery/plugin-sdk';
import type { TimbreGroupMeta } from './timbre-group-meta';
import { TIMBRE_GROUP_META_KEY } from './timbre-group-meta';

const ACCENT = '#2DD4BF';

export function TimbreGroupRow({
  group,
  ctx,
}: {
  group: ResolvedTrackGroup<TimbreGroupMeta, GeneratorTrackState>;
  ctx: GroupRenderContext;
}) {
  const [confirmingDelete, setConfirmingDelete] = useState(false);

  const generateAll = () => {
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
          {anyGenerating ? 'Generating…' : 'Generate All'}
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
      {!ctx.collapsed &&
        group.members.map((m) =>
          // Prompts hidden: generation is role-derived, nothing to type.
          ctx.renderDefaultTrackRow(m.track, { onPromptChange: undefined }),
        )}
    </div>
  );
}
