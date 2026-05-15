from harnessx.aegis.stages.adjudicate import compute_hit_rate, should_revert


def test_hit_rate_half():
    assert compute_hit_rate(predicted=["a", "b", "c", "d"],
                            actually_passed=["a", "b"]) == 0.5


def test_hit_rate_empty_prediction_none():
    assert compute_hit_rate(predicted=[], actually_passed=["a"]) is None


def test_should_revert_below_threshold():
    assert should_revert(hit_rate=0.4, threshold=0.5)
    assert not should_revert(hit_rate=0.5, threshold=0.5)
    assert not should_revert(hit_rate=0.6, threshold=0.5)


def test_should_revert_none_hit_rate_explorer_slot():
    assert not should_revert(hit_rate=None, threshold=0.5)


def test_auto_revert_disabled_never_reverts():
    from harnessx.aegis.stages.adjudicate import should_revert
    assert not should_revert(hit_rate=0.1, threshold=0.5, enabled=False)
    assert not should_revert(hit_rate=0.0, threshold=0.5, enabled=False)


def test_auto_revert_enabled_respects_threshold():
    from harnessx.aegis.stages.adjudicate import should_revert
    assert should_revert(hit_rate=0.1, threshold=0.5, enabled=True)
    assert not should_revert(hit_rate=0.9, threshold=0.5, enabled=True)
