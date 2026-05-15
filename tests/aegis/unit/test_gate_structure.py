from pathlib import Path
from harnessx.aegis.gates.structure import (
    validate_digest_anchors, validate_candidate_manifest,
    validate_critic_verdict, GateResult,
)


# v0.9.1 manifests must carry `capability_evidence` on `regular` slot.
# `locus` was dropped in v0.9.1 — interaction check moved to Critic.
_DEFAULT_CAPS: list = []


def _mf(**overrides):
    """Build a v0.9.1-shaped candidate manifest with sensible defaults."""
    mf = {
        "candidate_id": "C-R1-01",
        "bucket": "processor",
        "file_changes": [{"path": "a.py", "diff_sha_after": "abc"}],
        "predicted_impact": {"tasks_will_pass": ["t1"], "tasks_at_risk": []},
        "capability_evidence": list(_DEFAULT_CAPS),
    }
    mf.update(overrides)
    return mf


def test_digest_anchor_valid(tmp_path):
    session = tmp_path / "sessions" / "task_01_r1.jsonl"
    session.parent.mkdir(parents=True)
    session.write_text("msg0\nmsg1\nmsg2\n")
    digest = "Observation: agent did X.\n[sessions/task_01_r1.jsonl#msg_1]"
    result = validate_digest_anchors(digest, digest_root=tmp_path)
    assert result.ok


def test_digest_anchor_missing_file(tmp_path):
    digest = "Observation.\n[sessions/does_not_exist.jsonl#msg_0]"
    result = validate_digest_anchors(digest, digest_root=tmp_path)
    assert not result.ok
    assert "does_not_exist" in result.reason


def test_digest_anchor_line_out_of_range(tmp_path):
    session = tmp_path / "sessions" / "task_01_r1.jsonl"
    session.parent.mkdir(parents=True)
    session.write_text("msg0\n")
    digest = "[sessions/task_01_r1.jsonl#msg_99]"
    result = validate_digest_anchors(digest, digest_root=tmp_path)
    assert not result.ok


def test_candidate_manifest_missing_evidence():
    manifest = _mf(bucket="tools")
    body_md = "## Failure Evidence\n\n(empty)\n"
    result = validate_candidate_manifest(manifest, body_md, slot_type="regular")
    assert not result.ok


def test_candidate_manifest_explorer_allows_empty_evidence():
    manifest = {
        "candidate_id": "C-R5-01", "bucket": "tools",
        "file_changes": [{"path": "a.py", "diff_sha_after": "abc"}],
        "predicted_impact": {"tasks_will_pass": [], "tasks_at_risk": ["t99"]},
    }
    body_md = "## Failure Evidence\n\n(speculative)\n"
    result = validate_candidate_manifest(manifest, body_md, slot_type="explorer")
    assert result.ok


def test_candidate_manifest_accepts_backtick_anchors():
    manifest = _mf(bucket="prompt")
    manifest["file_changes"] = [{"path": "a.md", "diff_sha_after": "abc"}]
    body_md = (
        "## Failure Evidence\n\n"
        "- `trajectories/abc-def_r0.jsonl#step_4` — did X\n"
        "- `trajectories/xyz-789_r0.jsonl#step_12` — did Y\n"
    )
    result = validate_candidate_manifest(manifest, body_md, slot_type="regular")
    assert result.ok, result.reason


def test_candidate_manifest_accepts_bracket_anchors():
    manifest = _mf(candidate_id="C-R1-02", bucket="processor")
    body_md = (
        "## Failure Evidence\n\n"
        "- [trajectories/abc-def_r0.jsonl#step_4] — did X\n"
    )
    result = validate_candidate_manifest(manifest, body_md, slot_type="regular")
    assert result.ok, result.reason


def test_candidate_manifest_accepts_round_prefixed_anchors():
    manifest = _mf(candidate_id="C-R6-01", bucket="config")
    manifest["file_changes"] = [{"path": "config.yaml", "diff_sha_after": "abc"}]
    body_md = (
        "## Failure Evidence\n\n"
        "- `R6/trajectories/abc_r0.jsonl#step_5` — wasted a turn\n"
        "- `R5/trajectories/xyz_r1.jsonl#step_12` — same failure in R5\n"
    )
    result = validate_candidate_manifest(manifest, body_md, slot_type="regular")
    assert result.ok, result.reason


def test_candidate_manifest_accepts_digest_without_locator():
    manifest = _mf(candidate_id="C-R6-01", bucket="config")
    manifest["file_changes"] = [{"path": "config.yaml", "diff_sha_after": "abc"}]
    body_md = (
        "## Failure Evidence\n\n"
        "- Per `digests/task_X.md`: agent wasted a turn on stale path\n"
        "- See `digests/task_Y.md` for the same pattern\n"
    )
    result = validate_candidate_manifest(manifest, body_md, slot_type="regular")
    assert result.ok, result.reason


def test_validate_digest_anchors_tolerates_missing_locator(tmp_path):
    from harnessx.aegis.gates.structure import validate_digest_anchors
    digest_root = tmp_path
    (digest_root / "digests").mkdir()
    (digest_root / "digests" / "peer.md").write_text("peer content")
    (digest_root / "trajectories").mkdir()
    (digest_root / "trajectories" / "traj.jsonl").write_text("\n".join(["{}"] * 20))
    main_digest_text = (
        "## Observations\n"
        "- `trajectories/traj.jsonl#step_5` — reason here\n"
        "- See also `digests/peer.md` for similar pattern\n"
    )
    result = validate_digest_anchors(main_digest_text, digest_root)
    assert result.ok, f"Should accept digest-without-locator; got: {result.reason}"


def test_validate_digest_anchors_still_checks_line_range(tmp_path):
    from harnessx.aegis.gates.structure import validate_digest_anchors
    digest_root = tmp_path
    (digest_root / "trajectories").mkdir()
    (digest_root / "trajectories" / "short.jsonl").write_text("1\n2\n3\n")
    main_digest_text = (
        "## Observations\n"
        "- `trajectories/short.jsonl#step_99` — line 99 in a 3-line file\n"
    )
    result = validate_digest_anchors(main_digest_text, digest_root)
    assert not result.ok
    assert "out of range" in result.reason


def test_candidate_manifest_round_prefix_and_no_locator_combined():
    manifest = _mf(candidate_id="C-R6-01", bucket="config")
    manifest["file_changes"] = [{"path": "config.yaml", "diff_sha_after": "abc"}]
    body_md = (
        "## Failure Evidence\n\n"
        "Error paths verified in `R6/trajectories/33d8ea3b_r1.jsonl`\n"
        "- `digests/d1af70ea.md` line 8: stale base_dir wastes turn\n"
        "- `digests/023e9d44.md` line 10: same pattern on different task\n"
    )
    result = validate_candidate_manifest(manifest, body_md, slot_type="regular")
    assert result.ok, result.reason


def test_critic_verdict_no_anchor_fails():
    verdict = "This candidate looks good to me."
    result = validate_critic_verdict(verdict)
    assert not result.ok


def test_critic_verdict_with_yaml_list_anchor_passes():
    verdict = (
        "---\n"
        "candidate_id: C-R1-01\n"
        "verdict: accept\n"
        "evidence_anchors:\n"
        "  - trajectories/task_01_r1.jsonl#step_5\n"
        "confidence: 0.7\n"
        "---\n\n## Reasoning\nEvidence valid.\n"
    )
    result = validate_critic_verdict(verdict)
    assert result.ok


# --------------------------------------------------------------------------
# v0.9.1 additions — capability_evidence schema (locus dropped)
# --------------------------------------------------------------------------

_BODY_WITH_ANCHOR = (
    "## Failure Evidence\n"
    "`trajectories/abc.jsonl#step_4` — the step where things went wrong.\n\n"
    "## Root Cause\nStuff.\n\n"
    "## Targeted Fix\nStuff.\n\n"
    "## Why this won't break tasks_at_risk\nStuff.\n"
)


def test_v09_manifest_requires_capability_evidence_field():
    mf = _mf()
    del mf["capability_evidence"]
    r = validate_candidate_manifest(mf, _BODY_WITH_ANCHOR)
    assert not r.ok and "capability_evidence" in r.reason


def test_v09_manifest_accepts_empty_capability_evidence():
    mf = _mf()
    r = validate_candidate_manifest(mf, _BODY_WITH_ANCHOR)
    assert r.ok, r.reason


def test_v09_manifest_rejects_capability_evidence_entry_missing_fields():
    mf = _mf()
    mf["capability_evidence"] = [{"claim": "pubchem"}]  # missing type + evidence
    r = validate_candidate_manifest(mf, _BODY_WITH_ANCHOR)
    assert not r.ok and "capability_evidence" in r.reason


def test_v09_manifest_accepts_capability_evidence_list():
    mf = _mf()
    mf["capability_evidence"] = [
        {
            "type": "python_package",
            "claim": "httpx",
            "evidence": "bash: `pip show httpx` → present",
        },
        {
            "type": "http_endpoint",
            "claim": "https://api.example/v1",
            "evidence": "web_fetch 200 with JSON schema",
        },
    ]
    r = validate_candidate_manifest(mf, _BODY_WITH_ANCHOR)
    assert r.ok, r.reason


def test_v09_manifest_legacy_mode_tolerates_missing_fields():
    # Pre-v0.9 run artefacts don't have capability_evidence.
    mf = _mf()
    del mf["capability_evidence"]
    r = validate_candidate_manifest(mf, _BODY_WITH_ANCHOR, slot_type="legacy")
    assert r.ok, r.reason


def test_v09_manifest_does_not_require_locus():
    # Extraneous `locus:` field (from v0.9.0 runs) is harmless — we don't
    # validate it; legacy pre-v0.9 runs with no locus are equally fine.
    mf = _mf()
    mf["locus"] = [{"hook": "on_llm_call"}]  # even garbage is fine now
    r = validate_candidate_manifest(mf, _BODY_WITH_ANCHOR)
    assert r.ok, r.reason


# --------------------------------------------------------------------------
# v0.9.2 IV-9 + IV-10 — bucket-vs-file consistency + Jinja syntax gate
# --------------------------------------------------------------------------

def test_v092_config_bucket_rejects_md_in_file_changes():
    # Real bug from aegis_64_v091_r15_v2 R4: config-bucket candidate wrote
    # a prompt file (disguising prompt change as config). Must be rejected.
    # (Was .j2 in v0.9.2.0; migrated to .md in v0.9.2.1.)
    mf = _mf(bucket="config")
    mf["file_changes"] = [
        {"path": "/scratch/config.yaml", "action": "modify", "diff_summary": "kwarg"},
        {"path": "/scratch/gaia_agent.md", "action": "create", "diff_summary": "prompt"},
    ]
    r = validate_candidate_manifest(mf, _BODY_WITH_ANCHOR)
    assert not r.ok
    assert "IV-9" in r.reason
    assert "gaia_agent.md" in r.reason


def test_v092_config_bucket_rejects_py_in_file_changes():
    mf = _mf(bucket="config")
    mf["file_changes"] = [
        {"path": "/scratch/config.yaml", "action": "modify", "diff_summary": "x"},
        {"path": "/scratch/processor.py", "action": "create", "diff_summary": "y"},
    ]
    r = validate_candidate_manifest(mf, _BODY_WITH_ANCHOR)
    assert not r.ok and "IV-9" in r.reason


def test_v092_config_bucket_accepts_yaml_only():
    mf = _mf(bucket="config")
    mf["file_changes"] = [
        {"path": "/scratch/config.yaml", "action": "modify", "diff_summary": "kwarg"},
    ]
    r = validate_candidate_manifest(mf, _BODY_WITH_ANCHOR)
    assert r.ok, r.reason


def test_v092_prompt_bucket_accepts_md_and_yaml():
    mf = _mf(bucket="prompt")
    mf["file_changes"] = [
        {"path": "/scratch/gaia_agent.md", "action": "create", "diff_summary": "prompt"},
        {"path": "/scratch/config.yaml", "action": "modify", "diff_summary": "template_path swap"},
    ]
    r = validate_candidate_manifest(mf, _BODY_WITH_ANCHOR)
    assert r.ok, r.reason


def test_v092_prompt_bucket_rejects_py():
    mf = _mf(bucket="prompt")
    mf["file_changes"] = [
        {"path": "/scratch/gaia_agent.md", "action": "create", "diff_summary": "x"},
        {"path": "/scratch/processor.py", "action": "create", "diff_summary": "y"},
    ]
    r = validate_candidate_manifest(mf, _BODY_WITH_ANCHOR)
    assert not r.ok and "IV-9" in r.reason


def test_v092_prompt_bucket_rejects_j2_legacy():
    # Post-migration: prompts must be .md. .j2 is legacy/external and not
    # produced by new candidates — reject to force uniform format.
    mf = _mf(bucket="prompt")
    mf["file_changes"] = [
        {"path": "/scratch/gaia_agent.j2", "action": "create", "diff_summary": "prompt"},
    ]
    r = validate_candidate_manifest(mf, _BODY_WITH_ANCHOR)
    assert not r.ok and "IV-9" in r.reason


def test_v092_processor_bucket_accepts_py():
    mf = _mf(bucket="processor")
    mf["file_changes"] = [
        {"path": "/scratch/my_processor.py", "action": "create", "diff_summary": "x"},
        {"path": "/scratch/config.yaml", "action": "modify", "diff_summary": "register"},
    ]
    r = validate_candidate_manifest(mf, _BODY_WITH_ANCHOR)
    assert r.ok, r.reason


def test_v092_tools_bucket_accepts_py():
    mf = _mf(bucket="tools")
    mf["file_changes"] = [
        {"path": "/scratch/my_tool.py", "action": "create", "diff_summary": "x"},
    ]
    r = validate_candidate_manifest(mf, _BODY_WITH_ANCHOR)
    assert r.ok, r.reason


def test_v092_prompt_with_literal_curly_braces_is_accepted(tmp_path):
    # Post-migration: prompts are plain markdown, NOT Jinja templates.
    # The R3 bug (literal `{{cite tweet}}` crashing Jinja render at runtime)
    # is eliminated because PlainMarkdownSystemPromptBuilder does not render.
    # Gate must accept such prose without any Jinja syntax check.
    prose_md = tmp_path / "good.md"
    prose_md.write_text(
        "Rule 15: When counting Wikipedia citations, check `{{cite tweet}}` "
        "templates and `{{cite web}}` templates. Do NOT rely on regex alone."
    )
    mf = _mf(bucket="prompt")
    mf["file_changes"] = [
        {"path": str(prose_md), "action": "create", "diff_summary": "prompt"},
    ]
    r = validate_candidate_manifest(mf, _BODY_WITH_ANCHOR)
    assert r.ok, r.reason


def test_v092_md_delete_ok(tmp_path):
    mf = _mf(bucket="prompt")
    mf["file_changes"] = [
        {"path": "/nonexistent/old.md", "action": "delete", "diff_summary": "remove"},
    ]
    r = validate_candidate_manifest(mf, _BODY_WITH_ANCHOR)
    assert r.ok, r.reason


# --------------------------------------------------------------------------
# v0.9.3 — cross-bucket bundle (bucket as list) + IV-11 exploration gate
# --------------------------------------------------------------------------

def test_v093_bucket_list_union_extensions():
    # bucket=[prompt, processor] allows .md + .py + .yaml
    mf = _mf(bucket=["prompt", "processor"])
    mf["file_changes"] = [
        {"path": "/scratch/prompt.md", "action": "create", "diff_summary": "prompt"},
        {"path": "/scratch/guard.py", "action": "create", "diff_summary": "processor"},
        {"path": "/scratch/config.yaml", "action": "modify", "diff_summary": "wire"},
    ]
    r = validate_candidate_manifest(mf, _BODY_WITH_ANCHOR)
    assert r.ok, r.reason


def test_v093_bucket_list_rejects_extension_outside_union():
    # bucket=[prompt, processor] still rejects unrelated extensions
    mf = _mf(bucket=["prompt", "processor"])
    mf["file_changes"] = [
        {"path": "/scratch/prompt.md", "action": "create", "diff_summary": "prompt"},
        {"path": "/scratch/data.json", "action": "create", "diff_summary": "???"},
    ]
    r = validate_candidate_manifest(mf, _BODY_WITH_ANCHOR)
    assert not r.ok and "IV-9" in r.reason


def test_v093_bucket_single_str_still_works():
    # Backward compat: string form still accepted.
    mf = _mf(bucket="processor")
    mf["file_changes"] = [
        {"path": "/scratch/my.py", "action": "create", "diff_summary": "x"},
    ]
    r = validate_candidate_manifest(mf, _BODY_WITH_ANCHOR)
    assert r.ok, r.reason


# IV-11 exploration enforcement

_BODY_WITH_INFEASIBILITY = (
    "## Failure Evidence\n"
    "`trajectories/x.jsonl#step_1` — failure anchor.\n\n"
    "## Root Cause\nx\n\n"
    "## Targeted Fix\nx\n\n"
    "## Why flagged direction is infeasible\n"
    "Bash: `pip install tesseract-python` → package not found on any index; "
    "PyMuPDF OCR requires binary build step unavailable in this sandbox. "
    "Web fetch to api.ocr.space returns 403. Tools bucket for OCR is "
    "genuinely not feasible in this environment.\n\n"
    "## Why this won't break tasks_at_risk\nx\n"
)


def _prompt_mf(**overrides):
    """Helper: prompt-bucket manifest with file_changes that pass IV-9."""
    mf = _mf(bucket="prompt")
    mf["file_changes"] = [{"path": "/scratch/p.md", "action": "create", "diff_summary": "x"}]
    mf.update(overrides)
    return mf


def test_v093_iv11_noop_when_no_flagged_set():
    # If caller doesn't pass strategy_concern_flagged, IV-11 is dormant.
    mf = _prompt_mf()
    r = validate_candidate_manifest(mf, _BODY_WITH_ANCHOR)
    assert r.ok, r.reason


def test_v093_iv11_noop_when_flagged_empty():
    mf = _prompt_mf()
    r = validate_candidate_manifest(
        mf, _BODY_WITH_ANCHOR, strategy_concern_flagged=set(),
    )
    assert r.ok, r.reason


def test_v093_iv11_passes_when_candidate_targets_flagged():
    # candidate bucket matches flagged → pass
    mf = _mf(bucket="tools")
    mf["file_changes"] = [{"path": "/x/t.py", "action": "create", "diff_summary": "x"}]
    r = validate_candidate_manifest(
        mf, _BODY_WITH_ANCHOR, strategy_concern_flagged={"tools"},
    )
    assert r.ok, r.reason


def test_v093_iv11_passes_when_list_bucket_intersects_flagged():
    # bucket list includes flagged bucket → pass
    mf = _mf(bucket=["prompt", "tools"])
    mf["file_changes"] = [
        {"path": "/x/p.md", "action": "create", "diff_summary": "x"},
        {"path": "/x/t.py", "action": "create", "diff_summary": "x"},
    ]
    r = validate_candidate_manifest(
        mf, _BODY_WITH_ANCHOR, strategy_concern_flagged={"tools"},
    )
    assert r.ok, r.reason


def test_v093_iv11_rejects_when_avoided_without_justification():
    # Evolver ignores flagged and doesn't provide infeasibility section
    mf = _prompt_mf()
    r = validate_candidate_manifest(
        mf, _BODY_WITH_ANCHOR, strategy_concern_flagged={"tools"},
    )
    assert not r.ok and "IV-11" in r.reason
    assert "tools" in r.reason


def test_v093_iv11_accepts_with_concrete_infeasibility_section():
    # Evolver provides infeasibility section with evidence
    mf = _prompt_mf()
    r = validate_candidate_manifest(
        mf, _BODY_WITH_INFEASIBILITY, strategy_concern_flagged={"tools"},
    )
    assert r.ok, r.reason


# --------------------------------------------------------------------------
# v0.9.5 IV-12 — iterates_from validation (revert/improve prior ship)
# --------------------------------------------------------------------------

_BODY_WITH_ITERATE_EVIDENCE = (
    "## Failure Evidence\n"
    "Target C-R5-01 has hit_rate 0/5 per data/ship_outcomes.json and "
    "regressed tasks:\n"
    "- `trajectories/abc_r4.jsonl#step_4` — passed before C-R5-01\n"
    "- `trajectories/abc_r5.jsonl#step_8` — fails after C-R5-01\n\n"
    "## Root Cause\nx\n\n"
    "## Targeted Fix\nrevert C-R5-01\n\n"
    "## Why this won't break tasks_at_risk\nx\n"
)


def _iter_mf(**overrides):
    mf = _mf(bucket="config")
    mf["file_changes"] = [
        {"path": "/scratch/config.yaml", "action": "modify", "diff_summary": "revert"},
    ]
    mf.update(overrides)
    return mf


def test_v095_iv12_noop_when_no_iterates_from():
    # Regular candidate (no iterates_from) — IV-12 never fires.
    mf = _mf()
    r = validate_candidate_manifest(
        mf, _BODY_WITH_ANCHOR,
        prior_ships={"C-R5-01": {"round": 5, "bucket": "config"}},
        current_round=6,
    )
    assert r.ok, r.reason


def test_v095_iv12_noop_when_no_ledger_context():
    # Back-compat: if caller passes no prior_ships, IV-12 can't do much.
    # A malformed target (non-string / empty) is still rejected.
    mf = _iter_mf(iterates_from="C-R5-01")
    r = validate_candidate_manifest(mf, _BODY_WITH_ITERATE_EVIDENCE)
    assert r.ok, r.reason


def test_v095_iv12_rejects_non_string_target():
    mf = _iter_mf(iterates_from=42)
    r = validate_candidate_manifest(mf, _BODY_WITH_ITERATE_EVIDENCE)
    assert not r.ok and "IV-12" in r.reason


def test_v095_iv12_rejects_unknown_target():
    mf = _iter_mf(iterates_from="C-R5-99")
    r = validate_candidate_manifest(
        mf, _BODY_WITH_ITERATE_EVIDENCE,
        prior_ships={"C-R5-01": {"round": 5}},
        current_round=6,
    )
    assert not r.ok and "IV-12" in r.reason
    assert "not found" in r.reason


def test_v095_iv12_rejects_same_round_target():
    # Cannot iterate on a ship from the same round — the state isn't settled yet.
    mf = _iter_mf(iterates_from="C-R6-01")
    r = validate_candidate_manifest(
        mf, _BODY_WITH_ITERATE_EVIDENCE,
        prior_ships={"C-R6-01": {"round": 6, "bucket": "config"}},
        current_round=6,
    )
    assert not r.ok and "IV-12" in r.reason
    assert "prior round" in r.reason or "must be <" in r.reason


def test_v095_iv12_rejects_already_superseded_target():
    mf = _iter_mf(iterates_from="C-R5-01")
    r = validate_candidate_manifest(
        mf, _BODY_WITH_ITERATE_EVIDENCE,
        prior_ships={"C-R5-01": {"round": 5, "superseded_by": "C-R5b-02"}},
        current_round=6,
    )
    assert not r.ok and "IV-12" in r.reason
    assert "superseded" in r.reason


def test_v095_iv12_rejects_missing_evidence_linkage():
    # Body must cite target ship_id or hit_rate / ship_outcomes data
    mf = _iter_mf(iterates_from="C-R5-01")
    r = validate_candidate_manifest(
        mf, _BODY_WITH_ANCHOR,  # generic body — no target citation
        prior_ships={"C-R5-01": {"round": 5}},
        current_round=6,
    )
    assert not r.ok and "IV-12" in r.reason
    assert "cite" in r.reason


def test_v095_iv12_accepts_valid_iterate():
    mf = _iter_mf(iterates_from="C-R5-01")
    r = validate_candidate_manifest(
        mf, _BODY_WITH_ITERATE_EVIDENCE,
        prior_ships={"C-R5-01": {"round": 5, "bucket": "config", "hit_rate": "0/5"}},
        current_round=6,
    )
    assert r.ok, r.reason


def test_v095_iv12_accepts_body_citing_hit_rate_instead_of_id():
    body = _BODY_WITH_ANCHOR + "\n## Why revert\nPrior ship's hit_rate 0/N.\n"
    mf = _iter_mf(iterates_from="C-R5-01")
    r = validate_candidate_manifest(
        mf, body,
        prior_ships={"C-R5-01": {"round": 5}},
        current_round=6,
    )
    assert r.ok, r.reason


def test_v093_iv11_rejects_empty_infeasibility_section():
    # Section exists but content too thin — reject
    body_thin = (
        "## Failure Evidence\n"
        "`trajectories/x.jsonl#step_1` — x.\n\n"
        "## Root Cause\nx\n\n"
        "## Targeted Fix\nx\n\n"
        "## Why flagged direction is infeasible\ntoo hard\n\n"
        "## Why this won't break tasks_at_risk\nx\n"
    )
    mf = _prompt_mf()
    r = validate_candidate_manifest(
        mf, body_thin, strategy_concern_flagged={"tools"},
    )
    assert not r.ok and "IV-11" in r.reason
    assert "too short" in r.reason
