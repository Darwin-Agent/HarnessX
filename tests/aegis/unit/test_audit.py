import json
from pathlib import Path
from harnessx.aegis.data.audit import AuditLog, AuditEvent


def test_append_and_read(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append(AuditEvent(
        round=1, stage="P", kind="preprocess",
        payload={"task_count": 5}, evidence_refs=[],
    ))
    events = list(log.read_all())
    assert len(events) == 1
    assert events[0].kind == "preprocess"
    assert events[0].payload["task_count"] == 5


def test_append_is_durable_across_instances(tmp_path):
    path = tmp_path / "audit.jsonl"
    AuditLog(path).append(AuditEvent(
        round=1, stage="1", kind="plan", payload={}, evidence_refs=[],
    ))
    AuditLog(path).append(AuditEvent(
        round=1, stage="2", kind="propose", payload={}, evidence_refs=[],
    ))
    events = list(AuditLog(path).read_all())
    assert [e.kind for e in events] == ["plan", "propose"]


def test_jsonl_format_one_per_line(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append(AuditEvent(round=1, stage="P", kind="preprocess", payload={}, evidence_refs=[]))
    log.append(AuditEvent(round=1, stage="1", kind="plan", payload={}, evidence_refs=[]))
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    for line in lines:
        obj = json.loads(line)
        assert "ts" in obj and "kind" in obj


def test_query_by_round_and_kind(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    log.append(AuditEvent(round=1, stage="1", kind="plan", payload={}, evidence_refs=[]))
    log.append(AuditEvent(round=2, stage="1", kind="plan", payload={}, evidence_refs=[]))
    log.append(AuditEvent(round=2, stage="3", kind="decision", payload={}, evidence_refs=[]))
    r2_plans = list(log.query(round=2, kind="plan"))
    assert len(r2_plans) == 1
    assert r2_plans[0].stage == "1"
