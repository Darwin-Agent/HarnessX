import hashlib
from harnessx.aegis.data.signatures import compute_signature, FileChange


def test_signature_stable_across_call_order():
    changes_a = [
        FileChange(path="a.py", diff_sha_after="abc"),
        FileChange(path="b.py", diff_sha_after="def"),
    ]
    changes_b = [
        FileChange(path="b.py", diff_sha_after="def"),
        FileChange(path="a.py", diff_sha_after="abc"),
    ]
    assert compute_signature(changes_a) == compute_signature(changes_b)


def test_signature_differs_when_content_differs():
    a = [FileChange(path="a.py", diff_sha_after="abc")]
    b = [FileChange(path="a.py", diff_sha_after="xyz")]
    assert compute_signature(a) != compute_signature(b)


def test_signature_length_64():
    sig = compute_signature([FileChange(path="a.py", diff_sha_after="abc")])
    assert len(sig) == 64
    assert all(c in "0123456789abcdef" for c in sig)
