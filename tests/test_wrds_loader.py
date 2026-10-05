"""The WRDS username is found without anyone typing it on a command line.

A username passed as an argument lands in shell history and captured session
logs, against the repo's rule on account identifiers (#27 review). The pgpass
file the wrds package writes already holds it, in its fourth field.
"""

from __future__ import annotations

from capstone import wrds_loader


def _use_pgpass(monkeypatch, path):
    monkeypatch.setenv("PGPASSFILE", str(path))
    monkeypatch.delenv("WRDS_USERNAME", raising=False)


def test_username_comes_from_the_wrds_line_of_pgpass(tmp_path, monkeypatch):
    pgpass = tmp_path / "pgpass.conf"
    pgpass.write_text(
        "# saved by the wrds package\n"
        "otherhost:5432:db:someone_else:pw\n"
        + wrds_loader.WRDS_HOST
        + r":9737:wrds:test\:user:pa\:ss"
        + "\n"
    )
    _use_pgpass(monkeypatch, pgpass)
    assert wrds_loader.wrds_username() == "test:user"


def test_explicit_then_env_override_pgpass(tmp_path, monkeypatch):
    pgpass = tmp_path / "pgpass.conf"
    pgpass.write_text(f"{wrds_loader.WRDS_HOST}:9737:wrds:from_file:pw\n")
    _use_pgpass(monkeypatch, pgpass)
    monkeypatch.setenv("WRDS_USERNAME", "from_env")
    assert wrds_loader.wrds_username() == "from_env"
    assert wrds_loader.wrds_username("explicit") == "explicit"


def test_no_pgpass_and_no_env_means_no_username(tmp_path, monkeypatch):
    _use_pgpass(monkeypatch, tmp_path / "missing")
    assert wrds_loader.wrds_username() is None
