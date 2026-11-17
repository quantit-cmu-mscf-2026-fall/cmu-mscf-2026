"""Public reference factor inputs: parsing, units, the pre-2021 cutoff, the full calendar."""

import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from capstone.pipeline import reference_inputs
from capstone.pipeline.reference_inputs import POOL, french_months, validate_reference


def fixture_frame():
    index = pd.date_range("1990-01-01", "2020-12-01", freq="MS")
    x = np.random.default_rng(404).normal(0.003, 0.02, size=(len(index), len(POOL)))
    return pd.DataFrame(x, index=index, columns=POOL).assign(RF=0.001, **{"Mkt-RF": 0.004})


def archive(text):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as zipped:
        zipped.writestr("factors.csv", text)
    return stream.getvalue()


def test_french_months_convert_percent_and_drop_everything_after_2020():
    # The 2021 row is not read: its (nonnumeric) values would fail if it were.
    french = french_months(archive(",HML,RF\n199001,1.5,0.2\n202101,secret,ignored\n"))
    assert french.HML.iloc[0] == 0.015 and french.RF.iloc[0] == 0.002
    assert len(french) == 1 and french.index[0] == pd.Timestamp("1990-01-01")


@pytest.mark.parametrize(
    "text,message",
    [
        (",HML,RF\n199001,1.5,0.2\n199001,1.6,0.2\n", "ambiguous"),
        (",HML,RF\n199001,1.5\n", "ambiguous"),
        (",HML,RF\n199001,-99.99,0.2\n", "sentinel"),
        ("199001,1.5,0.2\n", "ambiguous"),
    ],
)
def test_ambiguous_or_missing_french_rows_are_refused(text, message):
    with pytest.raises(ValueError, match=message):
        french_months(archive(text))


def test_the_reference_frame_must_be_complete_finite_and_pre_2021():
    frame = validate_reference(fixture_frame().sample(frac=1, random_state=1))
    assert frame.index.is_monotonic_increasing and list(frame.columns) == sorted(frame.columns)
    with pytest.raises(ValueError, match="calendar"):
        validate_reference(fixture_frame().iloc[:-1])
    extra = fixture_frame()
    extra.loc[pd.Timestamp("2021-01-01")] = extra.iloc[-1]
    with pytest.raises(ValueError, match="calendar"):
        validate_reference(extra)
    missing = fixture_frame()
    missing.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_reference(missing)
    with pytest.raises(ValueError, match="finite"):
        validate_reference(fixture_frame().drop(columns="RF"))
    ruinous = fixture_frame()
    ruinous.loc[ruinous.index[3], "HML"] = -1.5
    with pytest.raises(ValueError, match="funded sleeve"):
        validate_reference(ruinous)


def test_fetch_records_sources_and_hashes_and_refuses_an_existing_folder(tmp_path, monkeypatch):
    frame = fixture_frame()
    percent = frame.mul(100)

    def zipped(columns):
        rows = ["," + ",".join(columns)]
        rows += [
            f"{d:%Y%m}," + ",".join(f"{percent.loc[d, c]:.10f}" for c in columns)
            for d in frame.index
        ]
        return archive("\n".join(rows) + "\n")

    payloads = {
        reference_inputs.URLS["five_factors"]: zipped(["HML", "RMW", "CMA", "RF", "Mkt-RF"]),
        reference_inputs.URLS["momentum"]: zipped(["Mom"]),
        reference_inputs.URLS["bab"]: b"workbook-bytes",
    }

    class Response:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            pass

    monkeypatch.setattr(
        reference_inputs.requests, "get", lambda url, timeout: Response(payloads[url])
    )
    monkeypatch.setattr(reference_inputs, "bab_months", lambda content: frame["BAB"])
    out = tmp_path / "reference"
    reference_inputs.fetch(out)
    sources = json.loads((out / "sources.json").read_text())
    assert sources["observations"] == len(frame) == 372
    assert sources["private_holdout_access"] is False
    assert sources["materialized_post_2020_observations"] == 0
    assert set(sources["sources"]) == set(reference_inputs.URLS)
    stored = pd.read_csv(out / "monthly.csv", index_col="month", parse_dates=True)
    pd.testing.assert_frame_equal(
        stored[sorted(stored)],
        frame[sorted(frame)].rename_axis("month"),
        check_freq=False,
        atol=1e-9,
        rtol=0,
    )
    with pytest.raises(FileExistsError):
        reference_inputs.fetch(out)
