"""Case 3 — "The green test".

The pull request under review is titled *Add tests for the weighting function*.
The whole diff is:

    def test_to_weights_gross():
        w = to_weights(SIGNAL, gross=1.0)
        assert w.abs().sum(axis=1).max() >= 0

CI is green.

Write your two lines before running this file.

Run:  python review-practice/case3_green_test.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from capstone.backtest import to_weights

SIGNAL = pd.DataFrame(
    np.random.default_rng(3).standard_normal((50, 8)),
    index=pd.bdate_range("2024-01-01", periods=50),
    columns=[f"A{i}" for i in range(8)],
)


def broken_to_weights(signal, *, demean=True, gross=1.0):
    """A deliberately broken implementation: it does not weight anything."""
    return signal.copy()


def weak_assertion(fn) -> bool:
    """The assertion the pull request actually added."""
    w = fn(SIGNAL, gross=1.0)
    return bool(w.abs().sum(axis=1).max() >= 0)


def strong_assertion(fn) -> bool:
    """What the test was presumably trying to say: each row's gross sums to 1."""
    w = fn(SIGNAL, gross=1.0)
    row_gross = w.abs().sum(axis=1)
    return bool(np.allclose(row_gross, 1.0))


def main() -> None:
    print(__doc__.splitlines()[0])
    print()
    print("                          real implementation | deliberately broken")
    print(
        f"  weak assertion   :  {str(weak_assertion(to_weights)):>17} | "
        f"{str(weak_assertion(broken_to_weights)):>18}"
    )
    print(
        f"  strong assertion :  {str(strong_assertion(to_weights)):>17} | "
        f"{str(strong_assertion(broken_to_weights)):>18}"
    )
    print()
    print("  THE CHECK: break the function on purpose and re-run the test. A test")
    print("  that stays green while the code is broken did not test the code.")
    print()
    print("  An absolute sum is never negative, so the weak assertion is true for")
    print("  every possible implementation, including one that does nothing at all.")
    print("  Green CI told you that Python ran, not that the function works.")


if __name__ == "__main__":
    main()
