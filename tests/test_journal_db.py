"""The generic locked journal connection: WAL, synchronous FULL, one owner at a time."""

import pytest

from capstone.journal_db import locked_journal


def test_connection_uses_wal_full_and_creates_no_tables(tmp_path):
    path = tmp_path / "sub" / "journal.db"
    with locked_journal(path) as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert db.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
        assert db.execute("SELECT count(*) FROM sqlite_master").fetchone()[0] == 0
        db.execute("CREATE TABLE t (x INTEGER)")
        db.execute("INSERT INTO t VALUES (1)")
        db.commit()
    with locked_journal(path) as db:
        assert db.execute("SELECT x FROM t").fetchall() == [(1,)]


def test_second_owner_is_refused_with_the_callers_message_and_lock_is_released(tmp_path):
    path = tmp_path / "journal.db"
    with pytest.raises(ValueError, match="interrupted"):
        with locked_journal(path):
            with pytest.raises(RuntimeError, match="owned elsewhere"):
                with locked_journal(path, message="owned elsewhere"):
                    pytest.fail("two owners opened one journal")
            raise ValueError("interrupted")
    with locked_journal(path):
        pass


def test_different_journal_extensions_have_independent_owners(tmp_path):
    first, second = tmp_path / "runs.db", tmp_path / "runs.sqlite"
    with locked_journal(first) as a, locked_journal(second) as b:
        for connection, value in ((a, 1), (b, 2)):
            connection.execute("CREATE TABLE t (x INTEGER)")
            connection.execute("INSERT INTO t VALUES (?)", (value,))
            connection.commit()
    for path, value in ((first, 1), (second, 2)):
        with locked_journal(path) as connection:
            assert connection.execute("SELECT x FROM t").fetchall() == [(value,)]


def test_journal_with_lock_extension_is_a_database_not_its_own_lock(tmp_path):
    path = tmp_path / "journal.lock"
    with locked_journal(path) as connection:
        connection.execute("CREATE TABLE t (x INTEGER)")
        connection.execute("INSERT INTO t VALUES (7)")
        connection.commit()
    with locked_journal(path) as connection:
        assert connection.execute("SELECT x FROM t").fetchall() == [(7,)]
