import io
import contextlib

from recipe.gaia_evolver.run import print_multiround_comparison


def test_prints_named_columns_and_deltas():
    rounds = [
        [
            {"task_id": "c61d22de-5f6f", "passed": False, "steps": 7, "cost_usd": 0.42, "tokens": 18432},
            {"task_id": "8e867cd7-cff0", "passed": False, "steps": 20, "cost_usd": 1.12, "tokens": 52019},
        ],
        [
            {"task_id": "c61d22de-5f6f", "passed": True, "steps": 6, "cost_usd": 0.40, "tokens": 17500},
            {"task_id": "8e867cd7-cff0", "passed": True, "steps": 18, "cost_usd": 0.98, "tokens": 45500},
        ],
    ]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_multiround_comparison(rounds)
    out = buf.getvalue()
    assert "R0 result" in out
    assert "R1 " in out and "vs-R0" in out
    assert "PASS" in out
    assert "+100" in out
    assert "Round totals" in out
    assert "pass_rate" in out
    assert "tokens" in out


def test_compares_historical_best_against_r0():
    rounds = [
        [
            {"task_id": "t1", "passed": False, "steps": 5, "cost_usd": 0.2, "total_tokens": 500},
            {"task_id": "t2", "passed": False, "steps": 5, "cost_usd": 0.2, "total_tokens": 500},
        ],
        [
            {"task_id": "t1", "passed": True, "steps": 4, "cost_usd": 0.1, "total_tokens": 400},
            {"task_id": "t2", "passed": True, "steps": 4, "cost_usd": 0.1, "total_tokens": 400},
        ],
        [
            {"task_id": "t1", "passed": True, "steps": 6, "cost_usd": 0.3, "total_tokens": 600},
            {"task_id": "t2", "passed": False, "steps": 6, "cost_usd": 0.3, "total_tokens": 600},
        ],
    ]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_multiround_comparison(rounds)
    out = buf.getvalue()
    assert "R1-vs-R0 pass" in out
    assert "best-vs-R0 pass_rate" in out


def test_handles_single_round():
    rounds = [[{"task_id": "t1", "passed": True, "steps": 1, "cost_usd": 0.1, "tokens": 100}]]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_multiround_comparison(rounds)
    out = buf.getvalue()
    assert "R0 result" in out
    assert "vs-R" not in out
