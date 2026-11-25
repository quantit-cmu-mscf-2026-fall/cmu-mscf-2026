"""Alpha-decay diagnostics for a run's hypothesis families (report-only).

    python research/decay_report.py --matrix M.parquet --summary S.parquet --run RUN \
        [--block-years 5] [--published published.csv] [--mkt-return mkt.parquet]

Reads an existing matrix (search period only; a later date is refused), logs one
ledger entry, writes experiments/analysis/<run>/decay/. Parameters default to
research/decay_report.toml. `published.csv` has columns `family,date` (ISO date).
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import pandas as pd

from capstone.analysis.decay import run_report

PARAMS = Path(__file__).with_suffix(".toml")


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df = df.set_index("date")
    return df.set_axis(pd.DatetimeIndex(df.index))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matrix", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", type=Path, default=PARAMS)
    ap.add_argument("--block-years", type=int)
    ap.add_argument("--published", type=Path)
    ap.add_argument("--mkt-return", type=Path, help="market return series (optional)")
    a = ap.parse_args()
    params = tomllib.loads(a.config.read_text(encoding="utf-8"))
    if a.block_years:
        params["block_years"] = a.block_years
    returns = _load(a.matrix)
    family = pd.read_parquet(a.summary, columns=["hypothesis_id"])["hypothesis_id"]
    published = None
    if a.published:
        pub = pd.read_csv(a.published)
        published = dict(zip(pub["family"].astype(str), pub["date"].astype(str), strict=True))
    mkt = None
    if a.mkt_return:
        m = _load(a.mkt_return)
        mkt = m.iloc[:, 0]
    inputs = {"matrix": a.matrix.name, "summary": a.summary.name}
    out = run_report(
        returns, family, a.run, published=published, mkt_return=mkt, inputs=inputs, **params
    )
    print(out)


if __name__ == "__main__":
    main()
