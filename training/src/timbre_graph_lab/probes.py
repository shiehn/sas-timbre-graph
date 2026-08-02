"""Six role-specific MIDI probes, short (dataset) and canonical (listening).

The SHORT probes are the primary dataset probes: 2-4s of audio that still
exercise register, transient, sustain, and velocity behaviour for the role.
The CANONICAL four-bar probes exist for human A/B listening and validation
renders only — using short probes for the perturbation corpus cuts render
and feature cost ~4x with negligible information loss for local timbre
deltas (docs/TRAINING.md, challenge #3).

All probes are deterministic: plain note-on/note-off, no pitch bend, no CCs.
"""

from __future__ import annotations

from dataclasses import dataclass

from synth2surge.audio.midi import MidiMessages, NOTE_OFF, NOTE_ON

from timbre_graph_lab.config import PROBE_VERSION, ROLES

BPM = 110.0
BEAT = 60.0 / BPM  # 0.545s


@dataclass(frozen=True)
class Probe:
    role: str
    kind: str  # "short" | "canonical"
    messages: MidiMessages
    duration: float  # includes tail
    version: str = PROBE_VERSION


def _events(notes: list[tuple[int, float, float, int]], tail: float) -> tuple[MidiMessages, float]:
    """notes: (midi_note, onset_beats, length_beats, velocity) -> messages + duration."""
    msgs: MidiMessages = []
    end = 0.0
    for note, onset_b, len_b, vel in notes:
        t_on = onset_b * BEAT
        t_off = (onset_b + len_b) * BEAT
        msgs.append(([NOTE_ON, note, vel], t_on))
        msgs.append(([NOTE_OFF, note, 0], t_off))
        end = max(end, t_off)
    msgs.sort(key=lambda m: m[1])
    return msgs, end + tail


# ---------------------------------------------------------------------------
# Short probes (~2-4s): the dataset workhorses
# ---------------------------------------------------------------------------

def _short_kick() -> tuple[MidiMessages, float]:
    # four hits, one velocity dip, one syncopated extra
    return _events(
        [(36, 0.0, 0.15, 118), (36, 1.0, 0.15, 96), (36, 2.0, 0.15, 118),
         (36, 2.75, 0.15, 80), (36, 3.0, 0.15, 110)],
        tail=1.2,
    )


def _short_snare() -> tuple[MidiMessages, float]:
    # backbeat pair + ghost
    return _events(
        [(50, 0.0, 0.12, 116), (50, 1.0, 0.12, 62), (50, 2.0, 0.12, 118),
         (50, 3.0, 0.12, 90)],
        tail=1.2,
    )


def _short_hat() -> tuple[MidiMessages, float]:
    # eighths with accents and one gap
    notes = []
    for i in range(6):
        vel = 110 if i % 2 == 0 else 72
        notes.append((66, i * 0.5, 0.1, vel))
    notes.append((66, 3.5, 0.4, 100))  # longer final hit exposes decay
    return _events(notes, tail=0.8)


def _short_bass() -> tuple[MidiMessages, float]:
    # sustained, staccato, repeated, octave
    return _events(
        [(36, 0.0, 0.9, 108), (36, 1.0, 0.2, 108), (36, 1.5, 0.2, 92),
         (43, 2.0, 0.9, 108), (48, 3.0, 0.9, 100)],
        tail=1.0,
    )


def _short_pad() -> tuple[MidiMessages, float]:
    # two chords with overlapping release
    return _events(
        [(48, 0.0, 2.1, 96), (52, 0.0, 2.1, 96), (55, 0.0, 2.1, 96),
         (53, 2.0, 2.0, 96), (57, 2.0, 2.0, 96), (60, 2.0, 2.0, 96)],
        tail=1.6,
    )


def _short_lead() -> tuple[MidiMessages, float]:
    # phrase spanning ~octave+fifth, mixed articulation
    return _events(
        [(60, 0.0, 0.45, 110), (64, 0.5, 0.45, 100), (67, 1.0, 0.9, 112),
         (72, 2.0, 0.2, 104), (67, 2.5, 0.2, 92), (79, 3.0, 0.9, 116)],
        tail=1.2,
    )


# ---------------------------------------------------------------------------
# Canonical four-bar probes: human listening / validation only
# ---------------------------------------------------------------------------

def _canonical(role: str) -> tuple[MidiMessages, float]:
    short_msgs, short_dur = _SHORT_BUILDERS[role]()
    # four bars = the short pattern repeated across 16 beats
    msgs: MidiMessages = []
    pattern_beats = 4.0
    for rep in range(4):
        offset = rep * pattern_beats * BEAT
        for midi_bytes, t in short_msgs:
            msgs.append((list(midi_bytes), t + offset))
    msgs.sort(key=lambda m: m[1])
    duration = 3 * pattern_beats * BEAT + short_dur
    return msgs, duration


_SHORT_BUILDERS = {
    "kick": _short_kick,
    "snare": _short_snare,
    "hat": _short_hat,
    "bass": _short_bass,
    "pad": _short_pad,
    "lead": _short_lead,
}


def get_probe(role: str, kind: str = "short") -> Probe:
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}")
    if kind == "short":
        msgs, dur = _SHORT_BUILDERS[role]()
    elif kind == "canonical":
        msgs, dur = _canonical(role)
    else:
        raise ValueError(f"unknown probe kind {kind!r}")
    return Probe(role=role, kind=kind, messages=msgs, duration=dur)


def all_probes(kind: str = "short") -> dict[str, Probe]:
    return {role: get_probe(role, kind) for role in ROLES}
