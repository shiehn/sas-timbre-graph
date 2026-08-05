from timbre_graph_lab.corpus import _load_rules, assign_roles

RULES = _load_rules()


def test_kick_keyword_in_percussion():
    # named AND in a drum folder: unambiguously a kick, and only a kick
    assert assign_roles("Kick 909ish", ["Percussion"], RULES) == ["kick"]


def test_snare_and_clap():
    assert "snare" in assign_roles("Snare Tight", ["Percussion"], RULES)
    assert "snare" in assign_roles("Big Clap", ["Percussion"], RULES)


def test_fx_category_rejected_globally():
    assert assign_roles("Whoosh Kick", ["FX"], RULES) == []


def test_riser_keyword_rejected_globally():
    assert assign_roles("Riser Lead", ["Leads"], RULES) == []


def test_bass_category_without_keyword():
    assert "bass" in assign_roles("Warm Fingers", ["Basses"], RULES)


def test_pad_categories():
    assert "pad" in assign_roles("Silky Strings", ["Pads"], RULES)
    assert "pad" in assign_roles("EP Poly Thing", ["Polysynths"], RULES)


def test_lead_via_plucks_category():
    assert "lead" in assign_roles("Glass Pluck", ["Plucks"], RULES)


def test_drum_folder_alone_admits_an_unlabelled_drum():
    """A file inside `Drums/` or `Percussion/` is percussion even when its
    name does not say WHICH drum.

    This used to return [] — toms and unlabelled hits were dropped — so the
    drum pools were pure filename matches: 51/60/61 candidates against the
    239 files actually sitting in drum folders. It also meant the folder
    signal contributed nothing at all (the old guard was a tautology).
    A tom is a perfectly good neighbour on a hat's timbre tour; a marimba,
    which is what the thin pools produced instead, was not.
    """
    roles = assign_roles("Synth Tom 1", ["Percussion"], RULES)
    assert set(roles) == {"kick", "snare", "hat"}
    # ...but only inside a drum folder — an unlabelled melodic patch is not a drum
    assert assign_roles("Synth Tom 1", ["Some Author", "Pads"], RULES) == ["pad"]


def test_3rdparty_kick_by_keyword_only():
    # 3rd-party trees have author folders, not category folders
    assert "kick" in assign_roles("Deep Kick 04", ["Some Author"], RULES)
