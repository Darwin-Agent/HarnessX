"""
TB2-specific scoring function for Layer 1 signal extraction.

TB2 data layout (from session_dir):
  session_dir  = {trial_dir}/agent/oh_runs/{session_id}/
  trial_dir    = session_dir.parents[2]
  result.json  = trial_dir/result.json          (verifier_result.rewards.reward)
  ctrf.json    = trial_dir/verifier/ctrf.json   (CTRF test results with names + traces)

Usage::

    from recipe.tb2_hx_evolver.score import tb2_score_by_session_dir
    from harnessx.experimental.harness_evol.signals.runner import extract_signals

    extract_signals(pattern, out, score_fn=tb2_score_by_session_dir)
"""
from __future__ import annotations

import json
from pathlib import Path


def tb2_score_by_session_dir(session_dir: Path) -> tuple[bool, float, str]:
    """
    Return (eval_passed, eval_score, eval_feedback) for a TB2 session.

    - eval_passed / eval_score: read from result.json (verifier_result.rewards.reward).
      TB2's episode_end.passed in the trace is always null; result.json is authoritative.
    - eval_feedback: CTRF test results from verifier/ctrf.json, containing per-test
      name, status, and stack trace. DigestAgent uses this to identify which specific
      checks the agent failed.
    """
    trial_dir = session_dir.parents[2]

    # ── score from result.json ────────────────────────────────────────────────
    result_path = trial_dir / "result.json"
    score = 0.0
    if result_path.exists():
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
            score = float(
                data.get("verifier_result", {}).get("rewards", {}).get("reward", 0.0)
            )
        except Exception:
            pass
    passed = score >= 1.0

    # ── feedback from verifier/ctrf.json ─────────────────────────────────────
    ctrf_path = trial_dir / "verifier" / "ctrf.json"
    feedback = ""
    if ctrf_path.exists():
        try:
            ctrf = json.loads(ctrf_path.read_text(encoding="utf-8"))
            tests = ctrf.get("results", {}).get("tests", [])
            lines = []
            for t in tests:
                status = t.get("status", "unknown")
                name = t.get("name", "?")
                msg = t.get("message", "") or t.get("trace", "")
                line = f"[{status}] {name}"
                if msg and status != "passed":
                    line += f": {msg[:300]}"
                lines.append(line)
            feedback = "\n".join(lines)
        except Exception:
            pass

    return passed, score, feedback
