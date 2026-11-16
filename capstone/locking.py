"""Process-owned local locks shared by journals and source archives."""

import os
import sys
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def exclusive_file(path, *, message="another process owns this file", create=True):
    """Fail immediately on contention; the operating system releases crash-held locks."""
    path = Path(path)
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b" if create else "r+b") as lock:
        lock.seek(0)
        if os.fstat(lock.fileno()).st_size == 0:
            if not create:
                raise ValueError("existing lock file is empty")
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        if sys.platform == "win32":
            import msvcrt

            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(message) from error
        else:
            import fcntl

            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError(message) from error
        try:
            yield
        finally:
            lock.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
