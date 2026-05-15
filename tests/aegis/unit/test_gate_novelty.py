from harnessx.aegis.gates.novelty import check_novelty
from harnessx.aegis.data.signatures import compute_signature, FileChange


def test_novel_signature_passes():
    changes = [FileChange(path="a.py", diff_sha_after="abc")]
    sig = compute_signature(changes)
    refuted = set()
    result = check_novelty(sig, refuted_signatures=refuted)
    assert result.ok


def test_refuted_signature_fails():
    changes = [FileChange(path="a.py", diff_sha_after="abc")]
    sig = compute_signature(changes)
    refuted = {sig}
    result = check_novelty(sig, refuted_signatures=refuted)
    assert not result.ok
    assert "refuted" in result.reason.lower()
