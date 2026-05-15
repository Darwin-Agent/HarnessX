from harnessx.aegis.data.reputation import Reputation, BUCKETS


def test_unknown_bucket_has_boost():
    rep = Reputation(window=5)
    assert rep.score("new_sub_agent") > 0.5


def test_bucket_hit_lowers_when_miss():
    rep = Reputation(window=5)
    baseline = rep.score("prompt")
    for _ in range(3):
        rep.record("prompt", hit=False)
    assert rep.score("prompt") < baseline


def test_bucket_hit_raises_when_hit():
    rep = Reputation(window=5)
    for _ in range(3):
        rep.record("tools", hit=True)
    assert rep.score("tools") > 0.5


def test_moving_window_drops_old():
    rep = Reputation(window=3)
    for _ in range(3):
        rep.record("prompt", hit=False)
    low = rep.score("prompt")
    for _ in range(3):
        rep.record("prompt", hit=True)
    assert rep.score("prompt") > low


def test_all_four_buckets_tracked():
    rep = Reputation(window=5)
    for b in ("prompt", "tools", "config", "processor"):
        rep.record(b, hit=True)
        assert rep.score(b) >= 0.5


def test_downweight_all(tmp_path):
    rep = Reputation(window=5)
    for _ in range(3):
        rep.record("tools", hit=True)
    high = rep.score("tools")
    rep.downweight_all(factor=0.9)
    assert rep.score("tools") < high
