from pathlib import Path
from harnessx.aegis.data.archive import Archive, ArchivedCandidate


def test_archive_write_read_round_trip(tmp_path):
    arch = Archive(tmp_path)
    arch.store(round_n=3, cid="C-R3-01", manifest_md="## Fake\nbody",
               failure_context={"gate": "novelty", "reason": "dup"})
    items = arch.list_round(3)
    assert len(items) == 1
    assert items[0].cid == "C-R3-01"
    assert "Fake" in items[0].manifest_md
    assert items[0].failure_context["reason"] == "dup"


def test_archive_across_rounds(tmp_path):
    arch = Archive(tmp_path)
    arch.store(round_n=2, cid="C-R2-01", manifest_md="r2", failure_context=None)
    arch.store(round_n=3, cid="C-R3-05", manifest_md="r3", failure_context=None)
    assert len(arch.list_round(2)) == 1
    assert len(arch.list_round(3)) == 1
    assert len(arch.list_all()) == 2


def test_recent_returns_n_most_recent(tmp_path):
    arch = Archive(tmp_path)
    for r in range(1, 6):
        arch.store(round_n=r, cid=f"C-R{r}-01", manifest_md=f"r{r}", failure_context=None)
    recent = arch.recent(n=2)
    assert len(recent) == 2
    assert {c.round for c in recent} == {4, 5}
