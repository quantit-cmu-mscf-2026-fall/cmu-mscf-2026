"""Public reference factors only; never opens the project's licensed/holdout data."""

import hashlib
import io
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from capstone.artifacts import atomic_json

POOL = ("HML", "RMW", "CMA", "Mom", "BAB")
URLS = {
    "five_factors": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "momentum": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_CSV.zip",
    "bab": "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"
    "Betting-Against-Beta-Equity-Factors-Monthly.xlsx",
}


def french_months(content):
    """Convert percent to decimal after selecting only 1990-2020 monthly rows."""
    rows, header = {}, None
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        text = archive.read(archive.namelist()[0]).decode("utf-8-sig")
    for line in text.splitlines():
        cells = [v.strip() for v in line.split(",")]
        key = cells[0]
        if not key and len(cells) > 1:
            header = cells[1:]
        if len(key) != 6 or not key.isdigit() or not "199001" <= key <= "202012":
            continue
        if header is None or len(cells) != len(header) + 1 or key in rows:
            raise ValueError("ambiguous French monthly format")
        values = [float(v) for v in cells[1:]]
        if any(v in {-99.99, -999.0} for v in values):
            raise ValueError("French missing-return sentinel")
        rows[key] = dict(zip(header, np.array(values) / 100, strict=True))
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index = pd.to_datetime(result.index, format="%Y%m")
    return result.sort_index()


def bab_months(content):
    from openpyxl import load_workbook

    workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    rows, column = {}, None
    for row in workbook["BAB Factors"].iter_rows(values_only=True):
        if row[0] == "DATE":
            column = row.index("USA")
            continue
        if column is None or row[0] is None:
            continue
        try:
            date = pd.Timestamp(row[0])
        except (TypeError, ValueError):
            continue
        if not pd.Timestamp("1990-01-01") <= date <= pd.Timestamp("2020-12-31"):
            continue
        date = date.to_period("M").to_timestamp()
        if date in rows or row[column] is None:
            raise ValueError("ambiguous or missing AQR monthly return")
        rows[date] = float(row[column])  # AQR workbook explicitly uses decimals.
    workbook.close()
    return pd.Series(rows, name="BAB").sort_index()


def validate_reference(frame):
    expected = pd.date_range("1990-01-01", "2020-12-01", freq="MS")
    if frame.index.has_duplicates or not frame.index.sort_values().equals(expected):
        raise ValueError("exact complete pre-2021 reference calendar required")
    if set(frame) != {*POOL, "RF", "Mkt-RF"} or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("complete finite reference factors required")
    if (1 + frame[list(POOL)].add(frame.RF, axis=0)).le(0).any().any():
        raise ValueError("invalid synthetic funded sleeve return")
    return frame.sort_index().sort_index(axis=1)


def fetch(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    payloads, provenance = {}, {}
    for name, url in URLS.items():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        payloads[name] = response.content
        provenance[name] = {
            "url": url,
            "sha256": hashlib.sha256(response.content).hexdigest(),
            "retrieved_utc": datetime.now(UTC).isoformat(),
        }
    five = french_months(payloads["five_factors"])
    frame = (
        five[["HML", "RMW", "CMA", "RF", "Mkt-RF"]]
        .join(french_months(payloads["momentum"])[["Mom"]])
        .join(bab_months(payloads["bab"]))
    )
    frame = validate_reference(frame)
    path = destination / "monthly.csv"
    frame.to_csv(path, index_label="month", float_format="%.17g", lineterminator="\n")
    atomic_json(
        destination / "sources.json",
        {
            "sources": provenance,
            "monthly_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "observations": len(frame),
            "start": "1990-01",
            "end": "2020-12",
            "vintage": "current public histories; not historical publication vintages",
            "materialized_post_2020_observations": 0,
            "private_holdout_access": False,
            "normalization": "French percent / 100; AQR decimals unchanged",
            "qualification": "reference allocation benchmark; underlying execution unqualified",
        },
    )
