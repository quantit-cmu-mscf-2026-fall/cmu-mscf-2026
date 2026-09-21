# Sharing a dataset across the team, verifiably

Five people analysing "the same data" is an assumption until something checks
it. This page describes the mechanism that turns it into a fact, and how to put
a dataset in a shared folder so nobody pulls the same rows twice.

The short version:

```bash
# whoever runs the pull, once:
python scripts/build_manifest.py crsp_sp500_2022_2023   # writes manifests/<name>.json
python scripts/sync_shared.py push --all                # copies to the shared folder

# everyone else, once:
python scripts/sync_shared.py pull --all                # copies in, verifies on arrival

# anyone, any time:
python scripts/verify_universe.py --all                 # exit 0 = match, 1 = diverged
```

## What is actually shared

Two different things travel by two different routes, and keeping them straight
is the whole design:

| | Where it lives | In git? |
|---|---|---|
| The data (`*.parquet`) | `data_cache/`, and the shared folder | **no** — `data_cache/` is gitignored |
| The fingerprint (`*.json`) | `manifests/` | **yes** — it is text, and tiny |

A manifest carries digests, row and column counts, a date range and the pandas
and numpy versions that produced it. It carries no values, so it is safe to
commit and safe to read in a PR diff.

## Why a fingerprint rather than trust

The failure this catches is quiet. Someone refreshes a pull after a vendor
revision, the refreshed file reaches three of five laptops, and the results
stop agreeing six weeks later with no obvious cause. A shared folder cannot
tell you that happened; a digest compared before a run turns it into a message
on day one.

It works. Changing a single return in a 2.1 million row CRSP extract — by 1e-6,
one value — produces:

```
DIVERGED  crsp_sp500_2022_2023: returns
            returns    expected 7c412ff39b46cf2c  got 25f05a5be953fb81
```

and exit code 1.

Note that the digest is taken over the **derived panels**, not the parquet
bytes. Hashing the file does not work: parquet bytes vary with writer version,
compression and row order, so two byte-different files routinely hold identical
data. `dates x securities` is the level at which "the same data" means
something.

The digest is deliberately blind to row and column order and to pandas'
nullable extension dtypes (`Float64` arrives from parquet on sparse columns),
and deliberately sensitive to values, labels, dates and the difference between
`NaN` and `0.0`. Those are the tests in `tests/test_manifest.py`.

## Setting up the shared folder

Any path this machine can see works. Google Drive for desktop mounts as an
ordinary folder, so a plain file copy is all this needs — no OAuth, no API
keys, nothing that could put a credential in a public repository.

```bash
# put this in your shell profile, pointing at the synced folder
export CAPSTONE_SHARED_DIR="G:/Shared drives/mscf-capstone/data"
```

or pass `--dest` per call. On Windows PowerShell:

```powershell
$env:CAPSTONE_SHARED_DIR = "G:\Shared drives\mscf-capstone\data"
```

`push` refuses to publish a dataset whose local copy does not match its own
manifest, so a bad copy cannot quietly become everyone's copy. `pull` always
verifies on arrival.

## Keep the shared copy small

`wrds_loader.crsp_daily` pulls every qualifying common share in the date range
and the S&P 500 filter is applied afterwards. On a 2022–2023 pull that is 4,742
securities where roughly 500 are wanted:

```
full pull      :  2,147,255 rows    57.9 MB   4742 permnos
500-name subset:    233,993 rows     7.2 MB    500 permnos
reduction      : 8.0x smaller
```

Query `sp500_members` first and pass the resulting permnos into the pull if the
shared copy only ever needs index members. 7 MB syncs in seconds; 58 MB is 51 MB
of non-members pushed to four other people.

## Adding a dataset

`build_manifest.py` handles both shapes it will meet:

- a **tidy** vendor extract (one row per date-security, with `date` and
  `permno` columns) is pivoted into OHLCV and returns panels, keyed on PERMNO.
  Tickers are reused across decades, so keying on ticker can silently collide
  and two people comparing digests would be comparing differently-built
  objects.
- an **already-pivoted** file (Ken French factors, sample prices) is
  fingerprinted as the single panel it is.

Ken French tables and synthetic panels are reproducible from code and need no
shared copy at all — `capstone.data.load_french` caches them on first use. They
are worth a manifest anyway, because Ken French restates history.

## When the universe contract lands

The panels here are built directly from a tidy CRSP frame. The
`EquityUniverse` contract on the feature branch carries the same panels plus a
point-in-time eligibility mask; when it merges, add `eligible` to the
fingerprinted set so a change in universe membership shows up as a divergence
too. `manifest.frame_digest` already handles boolean frames.
