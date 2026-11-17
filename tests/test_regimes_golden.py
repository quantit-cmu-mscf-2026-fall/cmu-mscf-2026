"""The regime series are the same values on pandas 2.2 and 3, any numpy and
any platform.

Run on pandas 2.2.* and 3.*: both must hit the same digest of every value's exact
bits. The series are built from exactly rounded operations only
(`math.fsum`, `math.sqrt`, IEEE products in date order, a sorted-list median), never
from pandas rolling or expanding kernels, whose summation differs between versions.
Synthetic path only: no data, no model, no network.
"""

from __future__ import annotations

import contextlib
import hashlib
import warnings

import pandas as pd
import pytest

from capstone import regimes
from capstone.factors.evaluation import DISCOVERY_END

N = 1300
# sha256 of every row's values as float.hex, comma-joined, one row per line (below).
GOLDEN = "82e2ea412047ef13419c7796e58c80e0aa45d69233db262e1b7b4e4b8247d2f7"


def _path() -> pd.Series:
    """A deterministic market path from integer arithmetic (no RNG whose stream could
    change between numpy versions), with a down leg so bear and panic are populated."""
    values = [
        (((t * 7919 + 13 * t * t) % 2001) - 1000) / 50000.0 - (0.0015 if 500 < t < 950 else -0.0004)
        for t in range(N)
    ]
    return pd.Series(values, index=pd.bdate_range(end=DISCOVERY_END, periods=N))


def _digest(frame: pd.DataFrame) -> str:
    text = "\n".join(",".join(float(v).hex() for v in row) for row in frame.itertuples(index=False))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_golden_digest():
    out = regimes.regime_series(_path())
    assert out["bear"].eq(1).sum() > 100 and out["panic"].gt(1).any()
    assert _digest(out) == GOLDEN, f"pandas {pd.__version__}: regime values changed"


@pytest.mark.parametrize("name", ["rolling", "expanding", "ewm"])
def test_no_pandas_window_kernels(monkeypatch, name):
    def refuse(*_args, **_kwargs):
        raise AssertionError(f"regimes used pandas .{name}()")

    monkeypatch.setattr(pd.Series, name, refuse)
    monkeypatch.setattr(pd.DataFrame, name, refuse)
    assert _digest(regimes.regime_series(_path())) == GOLDEN


def test_copy_on_write_and_string_options_do_not_matter():
    """pandas 3 defaults (copy on write, string dtype inference) switched on where the
    installed pandas still has the option (2.2); on pandas 3 they are the defaults."""
    pairs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for key in ("mode.copy_on_write", "future.infer_string"):
            try:
                pd.get_option(key)
            except (KeyError, pd.errors.OptionError):
                continue
            pairs += [key, True]
        with pd.option_context(*pairs) if pairs else contextlib.nullcontext():
            assert _digest(regimes.regime_series(_path())) == GOLDEN
