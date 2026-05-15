from pathlib import Path
from harnessx.aegis.data.journal import Journal, RoundEntry


def test_append_and_read(tmp_path):
    j = Journal(tmp_path / "journal.md")
    j.append(RoundEntry(
        round=5, action="ship", shipped_cid="C-R5-02",
        hypothesis_signatures=["a" * 64],
        refuted_signatures=[], hit_rate=None,
        narrative="shipped tool_order fix",
    ))
    entries = j.read_all()
    assert len(entries) == 1
    assert entries[0].round == 5
    assert entries[0].shipped_cid == "C-R5-02"


def test_refuted_signatures_accumulate(tmp_path):
    j = Journal(tmp_path / "journal.md")
    j.append(RoundEntry(round=1, action="ship", shipped_cid="C-R1-01",
                        hypothesis_signatures=["sig_a"], refuted_signatures=[],
                        hit_rate=None, narrative=""))
    j.append(RoundEntry(round=2, action="no_op", shipped_cid=None,
                        hypothesis_signatures=[], refuted_signatures=["sig_a"],
                        hit_rate=0.2, narrative="R1 refuted"))
    refuted = j.all_refuted_signatures()
    assert "sig_a" in refuted


def test_recent_window(tmp_path):
    j = Journal(tmp_path / "journal.md")
    for r in range(1, 8):
        j.append(RoundEntry(round=r, action="no_op", shipped_cid=None,
                            hypothesis_signatures=[], refuted_signatures=[],
                            hit_rate=None, narrative=f"r{r}"))
    recent = j.recent(window=3)
    assert [e.round for e in recent] == [5, 6, 7]
