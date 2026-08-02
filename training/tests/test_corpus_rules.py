from timbre_graph_lab.corpus import _load_rules, assign_roles

RULES = _load_rules()


def test_kick_keyword_in_percussion():
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


def test_percussion_without_keyword_gets_nothing():
    # toms and unlabeled drums stay out of v1 roles
    assert assign_roles("Synth Tom 1", ["Percussion"], RULES) == []


def test_3rdparty_kick_by_keyword_only():
    # 3rd-party trees have author folders, not category folders
    assert "kick" in assign_roles("Deep Kick 04", ["Some Author"], RULES)
