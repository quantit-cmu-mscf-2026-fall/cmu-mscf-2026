"""The same process-owned lock protects archives."""

import pytest

from capstone.locking import exclusive_file


def test_lock_refuses_second_owner_and_releases_after_exception(tmp_path):
    path = tmp_path / "source.lock"
    with pytest.raises(ValueError, match="interrupted"):
        with exclusive_file(path):
            with pytest.raises(RuntimeError, match="busy"):
                with exclusive_file(path, message="busy"):
                    pytest.fail("two owners acquired one lock")
            raise ValueError("interrupted")
    with exclusive_file(path):
        pass
    assert path.read_bytes() == b"0"
    assert {p.name for p in tmp_path.iterdir()} == {"source.lock"}


def test_read_only_lock_never_creates_missing_paths(tmp_path):
    missing = tmp_path / "missing" / "source.lock"
    with pytest.raises(FileNotFoundError):
        with exclusive_file(missing, create=False):
            pytest.fail("nonexistent lock acquired")
    assert not missing.parent.exists()
