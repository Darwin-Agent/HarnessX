"""Critic verdict YAML-list evidence_anchors format (Low-severity Fix L1).

Real Critic verdicts (per ``critic.j2``) emit ``evidence_anchors`` as a
YAML list in the frontmatter. The previous regex required bracketed anchors
in the body, which caused every production verdict to fail IV-4 validation.
"""
from harnessx.aegis.gates.structure import validate_critic_verdict


def test_critic_verdict_yaml_list_form_passes():
    """Real Critic verdicts use YAML-list evidence_anchors, not bracketed."""
    verdict = """---
candidate_id: C-R1-01
verdict: accept
evidence_anchors:
  - trajectories/abc.jsonl#step_4
  - digests/xyz.md
confidence: 0.7
---

## Reasoning
Looks good.
"""
    result = validate_critic_verdict(verdict)
    assert result.ok, f"should accept YAML-list anchors: {result.reason}"


def test_critic_verdict_empty_anchors_fails():
    verdict = """---
candidate_id: C-R1-01
verdict: accept
evidence_anchors: []
confidence: 0.5
---

## Reasoning
"""
    result = validate_critic_verdict(verdict)
    assert not result.ok


def test_critic_verdict_no_frontmatter_fails():
    result = validate_critic_verdict("just some text no frontmatter")
    assert not result.ok


def test_critic_verdict_digest_anchor_without_locator_passes():
    """Digest anchors don't need #step_N suffix — they reference whole files."""
    verdict = """---
candidate_id: C-R1-02
verdict: accept
evidence_anchors:
  - digests/xyz.md
confidence: 0.6
---

## Reasoning
"""
    result = validate_critic_verdict(verdict)
    assert result.ok, f"digest-only anchor should pass: {result.reason}"


def test_critic_verdict_malformed_anchor_fails():
    verdict = """---
candidate_id: C-R1-03
verdict: accept
evidence_anchors:
  - not_a_valid_prefix/foo.md
confidence: 0.5
---
"""
    result = validate_critic_verdict(verdict)
    assert not result.ok
