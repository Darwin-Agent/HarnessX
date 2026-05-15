from harnessx.aegis.agents.critic import parse_decision


def test_parse_decision_ship():
    md = """---
decision_type: ship
ship_ranking:
  - candidate_id: C-R5-02
    confidence: high
  - candidate_id: C-R5-04
    confidence: medium
archive_recommendations: []
rationale: pick C-02 first
---

## Reasoning
body [trajectories/x.jsonl#step_1]
"""
    d, body = parse_decision(md)
    assert d["decision_type"] == "ship"
    assert d["ship_ranking"][0]["candidate_id"] == "C-R5-02"
    assert "step_1" in body


def test_parse_decision_no_op():
    md = """---
decision_type: no_op
ship_ranking: []
archive_recommendations:
  - candidate_id: C-R5-03
    reason: insufficient evidence
rationale: all candidates inadequate
---

## Reasoning
body [trajectories/x.jsonl#step_1]
"""
    d, body = parse_decision(md)
    assert d["decision_type"] == "no_op"
    assert d["ship_ranking"] == []
