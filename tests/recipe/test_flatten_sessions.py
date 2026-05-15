"""Unit test for ``_flatten_sessions_to_raw`` (Important Fix 7).

Previous impl used substring match on ``parent.name`` and ``break``-ed after
the first match, silently discarding additional rollouts when k_rollouts > 1.
This test asserts the fixed behaviour: one flattened file per rollout, with
``{task_id}_r{i}.jsonl`` naming so Stage P's ``stem.rsplit("_r", 1)[0]`` still
extracts the original task_id.
"""
from pathlib import Path

from recipe.gaia_evolver.run_meta_aegis import _flatten_sessions_to_raw


def _mk_session(sessions_dir: Path, session_id: str, run_id: str, body: str) -> None:
    """Write a ``{run_id}.jsonl`` inside ``sessions_dir/{session_id}/`` (where
    session_id may contain a ``/`` e.g. ``"aegis/R0-<task_id>"``)."""
    d = sessions_dir / session_id
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{run_id}.jsonl").write_text(body, encoding="utf-8")


def test_multi_rollout_emits_one_file_per_rollout(tmp_path: Path) -> None:
    sessions = tmp_path / "sessions"
    raw = tmp_path / "raw"

    tid_a = "aaaa-1111"
    tid_b = "bbbb-2222"

    # Two rollouts for tid_a, one for tid_b. Session name is
    # f"{label}-{task_id}" where label is e.g. "aegis/R0".
    _mk_session(sessions, f"aegis/R0-{tid_a}", "run0", '{"r": 0}\n')
    # Second rollout: GAIA writes to a sibling session dir when the runner
    # is invoked twice. Simulate that by giving it a distinct label suffix —
    # but the current driver uses k_rollouts=1, so we instead place the two
    # rollouts as two .jsonl files in the same session dir (equivalent to
    # two segments / runs writing to the same session). The flatten should
    # pick up BOTH.
    _mk_session(sessions, f"aegis/R0-{tid_a}", "run1", '{"r": 1}\n')
    _mk_session(sessions, f"aegis/R0-{tid_b}", "run0", '{"r": 0}\n')

    # Trace jsonl that must be skipped.
    (sessions / f"aegis/R0-{tid_a}" / "run0_trace.jsonl").write_text("ignore", encoding="utf-8")

    records = [{"task_id": tid_a}, {"task_id": tid_b}]
    _flatten_sessions_to_raw(sessions, raw, records)

    emitted = sorted(p.name for p in raw.glob("*.jsonl"))
    # tid_a has two rollouts, tid_b has one.
    assert emitted == [
        f"{tid_a}_r0.jsonl",
        f"{tid_a}_r1.jsonl",
        f"{tid_b}_r0.jsonl",
    ]

    # Stage P extracts task_id via stem.rsplit("_r", 1)[0]; confirm that
    # yields the original task_id for each emitted file.
    for name in emitted:
        stem = name.replace(".jsonl", "")
        recovered = stem.rsplit("_r", 1)[0]
        assert recovered in {tid_a, tid_b}


def test_exact_suffix_match_does_not_collide_across_similar_task_ids(tmp_path: Path) -> None:
    """Substring match would have paired ``tid_short`` with a session named
    ``...-tid_short_extra``. Exact suffix match prevents this."""
    sessions = tmp_path / "sessions"
    raw = tmp_path / "raw"

    short = "t1"
    longer = "t1_extra"  # session name ends with "-t1_extra", not "-t1"
    _mk_session(sessions, f"aegis/R0-{longer}", "run0", "x\n")

    records = [{"task_id": short}]
    _flatten_sessions_to_raw(sessions, raw, records)

    # Nothing should be emitted for ``short`` — its session does not exist.
    assert list(raw.glob("*.jsonl")) == []


def test_empty_sessions_dir_is_noop(tmp_path: Path) -> None:
    _flatten_sessions_to_raw(tmp_path / "missing", tmp_path / "raw", [{"task_id": "x"}])
    # raw_dir is created but empty.
    assert (tmp_path / "raw").exists()
    assert list((tmp_path / "raw").glob("*.jsonl")) == []
