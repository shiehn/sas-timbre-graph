from timbre_graph_lab.config import ROLES
from timbre_graph_lab.probes import all_probes, get_probe


def test_all_roles_have_short_and_canonical_probes():
    for kind in ("short", "canonical"):
        probes = all_probes(kind)
        assert set(probes) == set(ROLES)
        for p in probes.values():
            assert p.duration > 0
            assert len(p.messages) >= 4


def test_probe_messages_sorted_and_in_range():
    for role in ROLES:
        p = get_probe(role, "short")
        times = [t for _, t in p.messages]
        assert times == sorted(times)
        assert times[-1] <= p.duration
        for midi_bytes, _ in p.messages:
            status, note, vel = midi_bytes
            assert status in (0x80, 0x90)
            assert 0 <= note <= 127
            assert 0 <= vel <= 127


def test_probes_deterministic():
    a = get_probe("bass", "short")
    b = get_probe("bass", "short")
    assert a.messages == b.messages
    assert a.duration == b.duration


def test_canonical_is_four_repeats_of_short():
    s = get_probe("kick", "short")
    c = get_probe("kick", "canonical")
    assert len(c.messages) == 4 * len(s.messages)
    assert c.duration > s.duration * 2
