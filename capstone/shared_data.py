"""The team's shared CRSP/Compustat files, found and verified automatically.

The files live in a Google Drive folder (`mscf-capstone-data`) shared with the
team. Drive for desktop syncs it to each member's machine; this module finds
that synced folder, checks every file against the folder's `SHA256SUMS`, and
keeps a verified local copy in `data_cache/wrds/`. Code and agents only ever
read the local copy, so a sync in progress, a file being replaced, or Drive
not running can never hand them a partial file.

    from capstone import shared_data

    daily = shared_data.load("sp500_daily_1990_2025",
                             columns=["date", "permno", "dlyret", "in_universe"],
                             start="2010-01-01")
    shared_data.data_version()      # short id of the dataset version, for the run ledger

Override the folder search with CAPSTONE_DATA_DIR, and the local copy location
with CAPSTONE_CACHE_DIR.
"""

from __future__ import annotations

import hashlib
import os
import string
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

FOLDER_NAME = "mscf-capstone-data"
MANIFEST = "SHA256SUMS"
LOCK_TIMEOUT = 3600.0  # longest wait for another agent's sync to finish, seconds
LOCK_STALE_AFTER = 120.0  # a lock not refreshed for this long was left by a crash
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data_cache" / "wrds"


class SharedDataError(RuntimeError):
    """The shared data could not be found or verified."""


def _candidates() -> list[Path]:
    """Places Drive for desktop mounts a My Drive folder, per platform."""
    roots: list[Path] = []
    if sys.platform == "win32":
        for letter in string.ascii_uppercase:
            root = Path(f"{letter}:\\")
            if root.exists():
                roots.append(root / "My Drive")
    else:
        roots += sorted((Path.home() / "Library" / "CloudStorage").glob("GoogleDrive-*/My Drive"))
        roots.append(Path.home() / "Google Drive" / "My Drive")
    return [root / FOLDER_NAME for root in roots]


def find_drive_folder() -> Path | None:
    """The synced shared folder on this machine, or None if there isn't one."""
    override = os.environ.get("CAPSTONE_DATA_DIR")
    if override:
        path = Path(override)
        if not (path / MANIFEST).exists():
            raise SharedDataError(f"CAPSTONE_DATA_DIR={path} has no {MANIFEST}")
        return path
    found = [path for path in _candidates() if (path / MANIFEST).exists()]
    if len(found) > 1:
        manifests = {path.joinpath(MANIFEST).read_text() for path in found}
        if len(manifests) > 1:
            raise SharedDataError(
                f"several differing copies found: {found}; set CAPSTONE_DATA_DIR to pick one"
            )
    return found[0] if found else None


def cache_dir() -> Path:
    return Path(os.environ.get("CAPSTONE_CACHE_DIR", DEFAULT_CACHE))


def _read_manifest(path: Path) -> dict[str, str]:
    """{filename: sha256} from a `sha256sum`-format file."""
    entries = {}
    for line in path.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            entries[name.lstrip("*")] = digest
    return entries


def write_manifest(path: Path, entries: dict[str, str]) -> None:
    """Write {filename: sha256} in `sha256sum` format.

    Always LF line endings: on Windows, text mode would write CRLF, and
    `sha256sum -c` then reads each filename with a trailing '\\r' and fails.
    """
    lines = "".join(f"{digest}  {name}\n" for name, digest in sorted(entries.items()))
    path.write_text(lines, newline="\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sync(verbose: bool = False) -> dict[str, str]:
    """Bring the local copy in line with the Drive folder; return what happened per file.

    Only files whose checksum changed are copied. Each copy is verified before
    it replaces the old one, so a failed or partial copy leaves the last good
    version in place. Files no longer on Drive are removed from the local copy.
    With no Drive folder available, the existing local copy is used as-is.
    """
    local = cache_dir()
    local.mkdir(parents=True, exist_ok=True)
    with _sync_lock(local) as heartbeat:
        return _sync_locked(local, verbose, heartbeat)


@contextmanager
def _sync_lock(local: Path):
    """One sync at a time per cache directory, across processes.

    Several agents on one machine share a cache; without this, two syncing at
    once collide on the same files (on Windows, "file in use" errors). Others
    wait up to LOCK_TIMEOUT, then find everything already up to date.

    The holder refreshes the lock's timestamp while it works (the yielded
    heartbeat), so a slow but live sync, such as a first download over a slow
    connection, is never mistaken for a crashed one. A lock not refreshed for
    LOCK_STALE_AFTER seconds was left by a crashed process and is removed.
    """
    lock = local / ".sync.lock"
    deadline = time.monotonic() + LOCK_TIMEOUT
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > LOCK_STALE_AFTER:
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() > deadline:
                raise SharedDataError(f"timed out waiting for another sync ({lock})") from None
            time.sleep(0.2)

    last = [time.monotonic()]

    def heartbeat() -> None:
        if time.monotonic() - last[0] > 5:
            os.utime(lock)
            last[0] = time.monotonic()

    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield heartbeat
    finally:
        lock.unlink(missing_ok=True)


def _copy(src: Path, dst: Path, heartbeat) -> None:
    """Chunked copy that keeps the sync lock alive during long downloads."""
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        for block in iter(lambda: fin.read(1 << 22), b""):
            fout.write(block)
            heartbeat()


def _sync_locked(local: Path, verbose: bool, heartbeat=lambda: None) -> dict[str, str]:
    local_manifest = local / MANIFEST
    have = _read_manifest(local_manifest) if local_manifest.exists() else {}

    drive = find_drive_folder()
    if drive is None:
        if not have:
            raise SharedDataError(
                f"no '{FOLDER_NAME}' Drive folder found and no local copy in {local}. "
                "Add the shared folder to your Drive (Add shortcut to Drive) or set "
                "CAPSTONE_DATA_DIR."
            )
        warnings.warn(f"Drive folder not found; using the local copy in {local}", stacklevel=2)
        return {name: "local only" for name in have}

    want = _read_manifest(drive / MANIFEST)
    status: dict[str, str] = {}
    for name, digest in want.items():
        target = local / name
        if have.get(name) == digest and target.exists():
            status[name] = "up to date"
            continue
        temp = local / f".{name}.{os.getpid()}.partial"
        try:
            _copy(drive / name, temp, heartbeat)
            if _sha256(temp) != digest:
                raise SharedDataError("checksum mismatch (file still syncing?)")
            os.replace(temp, target)
            have[name] = digest
            status[name] = "copied"
        except (OSError, SharedDataError) as exc:
            temp.unlink(missing_ok=True)
            status[name] = f"FAILED: {exc}" + ("; kept previous copy" if target.exists() else "")
        if verbose:
            print(f"  {name}: {status[name]}", flush=True)

    # A file taken off Drive (renamed or retired) must leave the local copy
    # too; otherwise it keeps feeding data_version(), and two teammates synced
    # to the same Drive report different versions.
    for name in sorted(set(have) - set(want)):
        try:
            (local / name).unlink(missing_ok=True)
            status[name] = "removed"
        except OSError as exc:  # e.g. still open on Windows; drop it from the manifest anyway
            status[name] = f"removed from manifest; file left in place: {exc}"
        del have[name]
        if verbose:
            print(f"  {name}: {status[name]}", flush=True)

    write_manifest(local_manifest, have)
    failed = {n: s for n, s in status.items() if s.startswith("FAILED")}
    if failed:
        warnings.warn(f"some files were not updated: {failed}", stacklevel=2)
    return status


def data_version() -> str:
    """12-character id of the local dataset version: log it with every run."""
    manifest = cache_dir() / MANIFEST
    if not manifest.exists():
        raise SharedDataError("no local copy yet; call sync() or load() first")
    return manifest_version(manifest)


def manifest_version(manifest: Path) -> str:
    """Version id of any SHA256SUMS file (the Drive's, or the local copy's)."""
    # Hash the parsed entries, not the file bytes, so line order or the
    # sha256sum '*' binary marker cannot change the id.
    entries = sorted(_read_manifest(manifest).items())
    return hashlib.sha256(repr(entries).encode()).hexdigest()[:12]


def load(
    name: str,
    columns: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    auto_sync: bool = True,
) -> pd.DataFrame:
    """Read one shared file (name without `.parquet`), synced and verified first.

    `start`/`end` filter on `date` (CRSP daily file) inside the parquet read, so
    unneeded rows never reach memory. Text columns load as pyarrow strings,
    about 4x smaller than Python objects; numeric columns stay numpy.

    Rows are deliberately NOT filtered on `in_universe`: use that column to pick
    what to hold, but take returns from every row. A stock usually leaves the
    index before its final day, and dropping those rows drops the delisting
    losses of positions still held.

    `dlyret` already includes the delisting return: CRSP adds a final row per
    delisted stock (`dlydelflg == 'Y'`) whose `dlyret` equals `delret`. Use
    `dlyret` as-is; compounding `delret` on top would count the loss twice.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if auto_sync:
        sync()
    path = cache_dir() / f"{name}.parquet"
    if not path.exists():
        raise SharedDataError(f"{name}.parquet is not in the shared data")

    filters = []
    if start:
        filters.append(("date", ">=", pd.Timestamp(start)))
    if end:
        filters.append(("date", "<=", pd.Timestamp(end)))

    table = pq.read_table(path, columns=columns, filters=filters or None)
    strings = {pa.string(): pd.StringDtype("pyarrow"), pa.large_string(): pd.StringDtype("pyarrow")}
    frame = table.to_pandas(types_mapper=strings.get)
    # Parquet restores pandas' nullable Float64/Int64/boolean, which numpy
    # cannot use directly (to_numpy() gives object arrays). Convert to plain
    # numpy dtypes; a nullable int or bool column with gaps becomes float64.
    for col in frame.columns:
        dtype = frame[col].dtype
        if not isinstance(dtype, pd.api.extensions.ExtensionDtype) or isinstance(
            dtype, pd.StringDtype
        ):
            continue
        if pd.api.types.is_float_dtype(dtype):
            frame[col] = frame[col].astype("float64")
        elif pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            plain = "int64" if pd.api.types.is_integer_dtype(dtype) else "bool"
            frame[col] = frame[col].astype("float64" if frame[col].isna().any() else plain)
    return frame


# --- Treatment of missing values -------------------------------------------
# The team's agreed rules, in one place so every strategy and agent applies them
# identically. Measured on the 1990-2025 file, they leave no missing return and
# no missing spread on any in_universe row. See docs/shared_data.md.

# Bank failures delisted while in the universe with no delisting return recorded.
# Regulators seized both banks and common shareholders were wiped out.
FAILED_BANKS = {11786: "SVB Financial", 90090: "Signature Bank"}
# Missing return for any other performance delisting, and for the first day a
# held stock stops being tracked by CRSP (moved to OTC trading).
PERFORMANCE_LOSS = -0.5
PERFORMANCE_ACTIONS = ("GDR", "GLI", "GEX")
MAX_SPREAD = 0.05  # quoted spreads wider than 500 bps are treated as bad quotes
SPREAD_WINDOW = 21
REPORT_FALLBACK_DAYS = 90


def daily_returns(daily: pd.DataFrame) -> pd.Series:
    """`dlyret` with every missing value on a tradable path resolved.

    Needs columns permno, date, dlyret, dlyretmissflg, dlydelflg, delactiontype.

    - New security's first day (NS) and single missing-price days (MP): 0. Both
      are exact: an NS row is the stock's first row, so nothing held it, and
      after an MP day CRSP's next return spans the gap from the last price.
    - Delisting with no return: -100% for FAILED_BANKS, PERFORMANCE_LOSS for
      other performance delistings.
    - First untracked day (NT) after a tracked one: PERFORMANCE_LOSS. Later
      untracked rows stay NaN, meaning no position can be held there: exit.
    """
    frame = daily.sort_values(["permno", "date"])
    ret = frame["dlyret"].astype("float64").copy()
    missing = ret.isna()
    flag = frame["dlyretmissflg"]

    ret[missing & flag.isin(["NS", "MP"])] = 0.0

    delisting = missing & (frame["dlydelflg"] == "Y")
    bank = delisting & frame["permno"].isin(list(FAILED_BANKS))
    ret[bank] = -1.0
    ret[delisting & ~bank & frame["delactiontype"].isin(PERFORMANCE_ACTIONS)] = PERFORMANCE_LOSS

    untracked = flag == "NT"
    was_tracked = ~untracked.groupby(frame["permno"]).shift(1, fill_value=True)
    ret[missing & untracked & was_tracked] = PERFORMANCE_LOSS
    return ret.reindex(daily.index).rename("ret")


def quoted_spread(daily: pd.DataFrame) -> pd.Series:
    """Relative quoted spread, (ask - bid) / midpoint, with gaps filled without look-ahead.

    Needs columns permno, date, dlybid, dlyask, in_universe. A quote counts if
    ask > bid > 0 and the spread is at most MAX_SPREAD. Otherwise, in order: the
    stock's median valid spread over the previous SPREAD_WINDOW rows (at least 5
    valid), then that day's median valid spread across in_universe stocks.
    Before 2000 about 29% of universe rows rely on the cross-sectional fill, so
    treat pre-2000 cost estimates with care.
    """
    frame = daily.sort_values(["permno", "date"])
    bid, ask = frame["dlybid"].astype("float64"), frame["dlyask"].astype("float64")
    spread = (ask - bid) / ((ask + bid) / 2)
    spread = spread.where((ask > bid) & (bid > 0) & (spread <= MAX_SPREAD))
    trailing = spread.groupby(frame["permno"]).transform(
        lambda s: s.shift(1).rolling(SPREAD_WINDOW, min_periods=5).median()
    )
    daily_median = spread.where(frame["in_universe"]).groupby(frame["date"]).transform("median")
    return spread.fillna(trailing).fillna(daily_median).reindex(daily.index).rename("spread")


def primary_links(link: pd.DataFrame) -> pd.DataFrame:
    """Standard CRSP-Compustat links: LU/LC, primary security, open end dates filled."""
    keep = link["linktype"].isin(["LU", "LC"]) & link["linkprim"].isin(["P", "C"])
    out = link[keep & link["lpermno"].notna()].copy()
    out["lpermno"] = out["lpermno"].astype("int64")
    out["linkdt"] = pd.to_datetime(out["linkdt"])
    out["linkenddt"] = pd.to_datetime(out["linkenddt"]).fillna(pd.Timestamp.max.normalize())
    return out


def available_from(
    fund: pd.DataFrame, trading_days, report_col: str = "rdq", period_col: str = "datadate"
) -> pd.Series:
    """First trading day a fundamental may be used: strictly after its report date.

    Reports often land after the close, so the report date itself is too early.
    With no report date, the period end + REPORT_FALLBACK_DAYS is used, later
    than 99% of actual reports. NaT when that day is past the end of the calendar.
    """
    days = pd.DatetimeIndex(sorted(pd.to_datetime(trading_days).unique()))
    # pd.Timedelta(days=...) warns under numpy 2.5; the unit= form does not.
    fallback = fund[period_col] + pd.Timedelta(REPORT_FALLBACK_DAYS, unit="D")
    base = fund[report_col].fillna(fallback)
    out = pd.Series(pd.NaT, index=fund.index, dtype="datetime64[ns]", name="available_from")
    known = base.notna().to_numpy()
    pos = days.searchsorted(pd.DatetimeIndex(base[known]), side="right")
    inside = pos < len(days)
    out.iloc[known.nonzero()[0][inside]] = days[pos[inside]]
    return out
