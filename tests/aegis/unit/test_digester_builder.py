from pathlib import Path
from harnessx import HarnessConfig
from harnessx.aegis.agents.digester import build_digester_harness, DigesterInputs


def test_digester_config_is_harness_config(tmp_path):
    inputs = DigesterInputs(
        task_id="task_01",
        pattern="ALL_FAIL",
        trajectory_paths=[tmp_path / "task_01_r1.jsonl"],
        digest_out_path=tmp_path / "digests" / "task_01.md",
    )
    cfg = build_digester_harness(inputs)
    assert isinstance(cfg, HarnessConfig)


def test_digester_has_write_scope_gate(tmp_path):
    inputs = DigesterInputs(
        task_id="task_01",
        pattern="ALL_FAIL",
        trajectory_paths=[tmp_path / "task_01_r1.jsonl"],
        digest_out_path=tmp_path / "digests" / "task_01.md",
    )
    cfg = build_digester_harness(inputs)
    # Check that WriteScopeGateProcessor is registered (either in processors list
    # or in runtime _rt_procs). Allow either — real HarnessConfig may use either.
    from harnessx.meta_harness.processors.write_scope_gate import WriteScopeGateProcessor
    found = False
    for p in cfg._rt_procs if cfg._rt_procs else []:
        if isinstance(p, WriteScopeGateProcessor):
            found = True
            break
    for pdef in cfg.processors if cfg.processors else []:
        if isinstance(pdef, dict):
            t = pdef.get("_target_", "")
            if "WriteScopeGateProcessor" in t:
                found = True
                break
    assert found, "WriteScopeGateProcessor not found in Digester HarnessConfig"


def test_digester_template_selection_by_pattern():
    from harnessx.aegis.agents.digester import _select_template
    assert _select_template("ALL_FAIL").name == "digester_all_fail.md"
    assert _select_template("ALL_PASS").name == "digester_all_pass.md"
    assert _select_template("PARTIAL_PASS").name == "digester_partial_pass.md"


def test_digester_template_embeds_trace_facts(tmp_path):
    """trace_facts_md is injected verbatim into the rendered system prompt so
    the Digester LLM reasons from grounded evidence rather than rediscovering it.

    Tests the render path directly (what build_digester_harness uses
    internally) so we don't couple to where the system_builder ends up in
    the processor tree."""
    from harnessx.aegis.agents.digester import _select_template
    from harnessx.aegis._prompt import render_template

    facts_md = "## Trace Facts (Layer A — mechanical; do not rewrite)\n\nSTUB-CONTENT-TOKEN-42"
    for pattern in ("ALL_FAIL", "ALL_PASS", "PARTIAL_PASS"):
        rendered = render_template(
            _select_template(pattern),
            task_id="task_X",
            digest_out_path=str(tmp_path / "digests" / "task_X.md"),
            trajectory_paths=[str(tmp_path / "task_X_r0.jsonl")],
            trajectory_refs=["trajectories/task_X_r0.jsonl"],
            trace_facts_md=facts_md,
        )
        assert "STUB-CONTENT-TOKEN-42" in rendered, (
            f"{pattern}: trace_facts_md not embedded in rendered template"
        )
        # No unrendered Jinja placeholder should leak through
        assert "{{ trace_facts_md }}" not in rendered


def test_digester_template_empty_trace_facts_renders_cleanly(tmp_path):
    """trace_facts_md='' should render without leaving unescaped Jinja placeholders."""
    from harnessx.aegis.agents.digester import _select_template
    from harnessx.aegis._prompt import render_template

    for pattern in ("ALL_FAIL", "ALL_PASS", "PARTIAL_PASS"):
        rendered = render_template(
            _select_template(pattern),
            task_id="task_Y",
            digest_out_path=str(tmp_path / "digests" / "task_Y.md"),
            trajectory_paths=[str(tmp_path / "task_Y_r0.jsonl")],
            trajectory_refs=["trajectories/task_Y_r0.jsonl"],
            trace_facts_md="",
        )
        assert "{{ trace_facts_md }}" not in rendered


def test_digester_template_declares_layer_b_contract():
    """All three templates must mention the Pathology signals section with
    the evidence-driven schema (anchor, snippet, observation, severity)."""
    from harnessx.aegis.agents.digester import _select_template

    required = ("Pathology signals", "anchor:", "snippet:", "observation:", "severity:")
    for pattern in ("ALL_FAIL", "ALL_PASS", "PARTIAL_PASS"):
        text = _select_template(pattern).read_text(encoding="utf-8")
        for token in required:
            assert token in text, f"{pattern} missing {token!r}"


def test_digester_template_lists_pathology_vocabulary():
    """Vocabulary must include the failure modes we learned about in pilots."""
    from harnessx.aegis.agents.digester import _select_template

    must_have = (
        "tool_effect_missing",
        "repeat_without_progress",
        "multimodal_silent_drop",  # the render_pdf_page case
    )
    for pattern in ("ALL_FAIL", "ALL_PASS", "PARTIAL_PASS"):
        text = _select_template(pattern).read_text(encoding="utf-8")
        for tok in must_have:
            assert tok in text, f"{pattern} missing vocabulary entry {tok!r}"
