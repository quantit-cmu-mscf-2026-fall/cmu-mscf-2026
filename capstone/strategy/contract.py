"""The strategy contract: what a deployed strategy is.

A `StrategySpec` is a validated variant (a factor tree traded by a
`TradingSpec`) frozen into a file: its expression, how it is traded, the data
it was validated on, the date its rebalance calendar counts from, its gross
exposure, and where it came from (factor, variant, hypothesis, gate report).
Its id is a hash of everything else in its file, so the same content always
has the same id and any change makes a new one.

The file is written by code, never by hand: UTF-8, LF line endings, the same
bytes every time. `load_spec` refuses a file that is not exactly what its
spec would write (an edited value, a stale id or derived field). The file
carries a `band` line only for a spec with a no-trade band, so specs written
without one keep their bytes and id.

`target_weights(spec, panel, as_of)` is the book the strategy holds after
`as_of`'s close, permno -> weight, computed by evaluation's own
`factor_signal` and `signal_weights`:

- **Anchored calendar.** The book is rebalanced on every `hold`-th trading day
  counted from `spec.anchor` in the panel's calendar, and held in between, so
  the rebalance dates belong to the spec, not to whichever window was loaded.
  The panel must therefore contain the anchor. Anchored at the panel's first
  date the dates are evaluation's (its backtest counts from row 0).
- **Fixed window.** The signal is computed on exactly the `lookback` rows its
  expression needs before the rebalance date, not on all the history loaded.
  Every operator but `ema` has finite memory, so this is evaluation's value;
  with it, even `ema` gives the same book whatever window is loaded. Too few
  rows raises `InsufficientHistoryError`.
- **No-trade band.** With `trading.band` > 0 the book on a rebalance date
  depends on the book before it, so it is the backtest's `band_book` replayed
  from the anchor (flat there): the same rule, on the same targets, as
  evaluation. The spec's file carries a `band` line only then. A banded book
  is not rescaled, so its gross can differ from the spec's by up to `band`
  per kept name.
- **Holdout guard.** A panel holding any date after `DISCOVERY_END` is refused
  (`CURRENT_ROUTE_MISSING`): there is no flag that admits it.
"""

from __future__ import annotations

import hashlib
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from capstone.backtest import band_book, to_weights
from capstone.factors.evaluation import (
    DISCOVERY_END,
    HoldoutError,
    Panel,
    TradingSpec,
    factor_signal,
    signal_weights,
    trading_signal,
)
from capstone.factors.store import variant_id as _variant_id
from capstone.factors.tree import (
    BINARY_WINDOW_FUNCS,
    FUNCS,
    MARKET_FIELDS,
    BinOp,
    Call,
    Node,
    Unary,
    factor_id,
    features_used,
    parse,
    unparse,
)

HEADER = (
    "# Strategy spec (capstone.strategy.contract). Written by code; the id is a hash of the rest."
)


class InsufficientHistoryError(ValueError):
    """The panel has fewer rows before the rebalance date than the expression reads."""


@dataclass(frozen=True)
class StrategySpec:
    """A validated variant, frozen for trading. See the module docstring.

    `expression` is stored in `unparse`'s canonical spelling and `anchor` as
    YYYY-MM-DD, so two spellings of the same content share one id. The data
    fields, factor id and variant id are derived from the content, never
    passed in, so they cannot disagree with it.
    """

    expression: str
    trading: TradingSpec
    data_version: str  # the shared data version the variant was validated on
    anchor: str  # first rebalance date; the calendar counts trading days from it
    gross: float = 1.0  # sum of |weight| of the book
    hypothesis_id: str = ""
    gate_report: str = ""  # path or link to the gate report; required to register

    def __post_init__(self) -> None:
        object.__setattr__(self, "expression", unparse(parse(self.expression)))
        if not isinstance(self.trading, TradingSpec):
            raise ValueError("trading must be a TradingSpec")
        if not self.data_version:
            raise ValueError("a spec must name the data version it was validated on")
        try:
            anchor = pd.Timestamp(self.anchor)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"anchor {self.anchor!r} is not a date") from exc
        if anchor != anchor.normalize():
            raise ValueError(f"anchor {self.anchor!r} must be a date, not a time")
        object.__setattr__(self, "anchor", anchor.strftime("%Y-%m-%d"))
        gross = float(self.gross)
        if not (math.isfinite(gross) and gross > 0):
            raise ValueError(f"gross must be a positive number, got {self.gross!r}")
        object.__setattr__(self, "gross", gross)
        for name in ("data_version", "hypothesis_id", "gate_report"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")

    @property
    def tree(self) -> Node:
        return parse(self.expression)

    @property
    def fields(self) -> tuple[str, ...]:
        """The data fields the expression reads, sorted."""
        return tuple(sorted(features_used(self.tree)))

    @property
    def factor_id(self) -> str:
        return factor_id(self.tree)

    @property
    def variant_id(self) -> str:
        return _variant_id(self.factor_id, self.trading.as_dict())

    @property
    def id(self) -> str:
        """16 hex digits of the SHA-256 of the spec's file without its id line."""
        return hashlib.sha256(self._body().encode("utf-8")).hexdigest()[:16]

    def _body(self) -> str:
        fields = ", ".join(basic_string(f) for f in self.fields)
        # No line without a band, so specs written before bands keep their bytes and id.
        band = f"band = {self.trading.band!r}\n" if self.trading.band > 0 else ""
        return (
            f"{HEADER}\n"
            f"expression = {basic_string(self.expression)}\n"
            f"anchor = {basic_string(self.anchor)}\n"
            f"gross = {self.gross!r}\n"
            "\n[trading]\n"
            f"hold = {self.trading.hold}\n"
            f"neutral = {basic_string(self.trading.neutral)}\n"
            f"{band}"
            "\n[data]\n"
            f"fields = [{fields}]\n"
            f"version = {basic_string(self.data_version)}\n"
            "\n[provenance]\n"
            f"factor_id = {basic_string(self.factor_id)}\n"
            f"variant_id = {basic_string(self.variant_id)}\n"
            f"hypothesis_id = {basic_string(self.hypothesis_id)}\n"
            f"gate_report = {basic_string(self.gate_report)}\n"
        )

    def to_toml(self) -> str:
        """The spec's file: its id on the line after the header, then the content."""
        header, rest = self._body().split("\n", 1)
        return f"{header}\nid = {basic_string(self.id)}\n{rest}"

    @classmethod
    def from_toml(cls, text: str) -> StrategySpec:
        """The spec a file holds. A file that is not exactly what its spec writes
        (an edited value, a stale id or derived field) is refused."""
        values = tomllib.loads(text)
        try:
            spec = cls(
                expression=values["expression"],
                trading=TradingSpec(
                    hold=values["trading"]["hold"],
                    neutral=values["trading"]["neutral"],
                    band=values["trading"].get("band", 0.0),
                ),
                data_version=values["data"]["version"],
                anchor=values["anchor"],
                gross=values["gross"],
                hypothesis_id=values["provenance"]["hypothesis_id"],
                gate_report=values["provenance"]["gate_report"],
            )
        except KeyError as exc:
            raise ValueError(f"spec file is missing {exc}") from exc
        if values.get("id") != spec.id:
            raise ValueError(f"spec file id {values.get('id')!r} does not match its content")
        if spec.to_toml() != text.replace("\r\n", "\n"):
            raise ValueError(f"spec file {spec.id} is not the file its content writes")
        return spec


def basic_string(text: str) -> str:
    """A TOML basic string (also a YAML double-quoted scalar)."""
    out = []
    for ch in text:
        if ch in '"\\':
            out.append("\\" + ch)
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def save_spec(spec: StrategySpec, path: Path) -> None:
    """Write the spec's file: UTF-8, LF line endings, the same bytes every time."""
    Path(path).write_bytes(spec.to_toml().encode("utf-8"))


def load_spec(path: Path) -> StrategySpec:
    """The spec a file holds (refused if the file is not exactly what its spec writes)."""
    text = Path(path).read_bytes().decode("utf-8")
    schema = tomllib.loads(text).get("schema_version")
    if schema == 5:
        from capstone.strategy.compiled import CompiledStrategySpec

        return CompiledStrategySpec.from_toml(text)
    if schema == 4:
        from capstone.strategy.diversified import DiversifiedStrategySpec

        return DiversifiedStrategySpec.from_toml(text)
    if schema == 2:
        from capstone.strategy.recorded import load_compatible_spec
        from capstone.strategy.reference import ReferenceStrategySpec

        return load_compatible_spec(ReferenceStrategySpec, text)
    return StrategySpec.from_toml(text)


# ---------------------------------------------------------------------------
# The book


def lookback(node: Node) -> int:
    """Rows before a date that the expression's value on that date reads.

    `shift` and `delta` by d read d rows back; a d-day window (rolling
    operators, `ts_corr`, `ts_cov`, `ema`'s first value) reads d - 1. Nested
    operators add up; operands of one operator take the longer of the two.
    """
    if isinstance(node, Unary):
        return lookback(node.operand)
    if isinstance(node, BinOp):
        return max(lookback(node.left), lookback(node.right))
    if isinstance(node, Call):
        inner = lookback(node.arg)
        if node.arg2 is not None:
            inner = max(inner, lookback(node.arg2))
        if node.func in ("shift", "delta"):
            return inner + node.window
        if node.func in BINARY_WINDOW_FUNCS or FUNCS.get(node.func):
            return inner + node.window - 1
        return inner
    return 0  # a field or a constant


def history(node: Node) -> int:
    """Rows a panel needs before a date for the expression's value there: its
    `lookback`, plus one for a market field (the panel's first row has no
    market return: it weights by the previous day's cap)."""
    return lookback(node) + (1 if features_used(node) & set(MARKET_FIELDS) else 0)


def _rows(panel: Panel, start: int, stop: int) -> Panel:
    def cut(frame):
        return None if frame is None else frame.iloc[start:stop]

    fields = {name: cut(value) for name, value in panel.fields.items()}
    return Panel(
        fields, cut(panel.universe), cut(panel.half_spread), cut(panel.sic), panel.data_version
    )


CURRENT_ROUTE_MISSING = (
    "panels reaching past the discovery period require an authorized current-feature route; "
    "none is available here, and a flag is not an access authorization"
)


def target_weights(spec: StrategySpec, panel: Panel, as_of) -> pd.Series:
    """The book the strategy holds after `as_of`'s close: permno -> weight.

    A panel holding any date after `DISCOVERY_END` is refused
    (`CURRENT_ROUTE_MISSING`): the holdout is not read through the contract.
    """
    from capstone.strategy.recorded import require_current_implementation

    require_current_implementation(spec)
    dates = pd.DatetimeIndex(panel.universe.index)
    if len(dates) and dates.max() > pd.Timestamp(DISCOVERY_END):
        raise HoldoutError(CURRENT_ROUTE_MISSING)
    return _target_weights(spec, panel, as_of)


def _target_weights(spec: StrategySpec, panel: Panel, as_of) -> pd.Series:
    """The book held after `as_of`'s close: permno -> weight, nonzero weights
    only, sorted by permno, summing in absolute value to `spec.gross` (a banded
    book: see the module docstring).

    `as_of` and `spec.anchor` must be trading dates of the panel, `as_of` on
    or after the anchor. The book is set on the last anchored rebalance date
    on or before `as_of`. Only `target_weights` calls this, after refusing
    any panel holding a date past `DISCOVERY_END`.
    """
    from capstone.strategy.compiled import CompiledStrategySpec, compiled_weights
    from capstone.strategy.diversified import DiversifiedStrategySpec, diversified_weights

    if isinstance(spec, CompiledStrategySpec):
        return compiled_weights(spec, panel, as_of)
    if isinstance(spec, DiversifiedStrategySpec):
        return diversified_weights(spec, panel, as_of)
    from capstone.strategy.reference import (
        BenchmarkSpec,
        ReferenceStrategySpec,
        benchmark_target,
        reference_weights,
    )

    if isinstance(spec, BenchmarkSpec):
        return benchmark_target(spec, panel, as_of)
    if isinstance(spec, ReferenceStrategySpec):
        if panel.calendar is None:
            raise ValueError("reference strategies require a frozen exchange-session calendar")
        return reference_weights(spec, panel, as_of, panel.calendar)
    dates = pd.DatetimeIndex(panel.universe.index)
    as_of = pd.Timestamp(as_of)
    anchor = pd.Timestamp(spec.anchor)
    if as_of not in dates:
        raise ValueError(f"as_of {as_of.date()} is not a trading date of the panel")
    if anchor not in dates:
        raise ValueError(
            f"the panel does not contain the spec's anchor {spec.anchor}; "
            "its rebalance calendar counts trading days from there"
        )
    if as_of < anchor:
        raise ValueError(f"as_of {as_of.date()} is before the spec's anchor {spec.anchor}")
    a, p = dates.get_loc(anchor), dates.get_loc(as_of)
    rebalance = a + (p - a) // spec.trading.hold * spec.trading.hold
    tree = spec.tree
    need = history(tree)
    start = rebalance - need
    if start < 0:
        raise InsufficientHistoryError(
            f"the book set on {dates[rebalance].date()} needs {need} earlier rows; "
            f"the panel has {rebalance}"
        )
    if spec.trading.band > 0:
        row = _banded_row(spec, panel, a, rebalance, need)
    else:
        window = _rows(panel, start, rebalance + 1)
        signal = factor_signal(tree, window)
        row = signal_weights(signal, window, spec.trading.neutral).iloc[-1]
    if row.isna().all():
        raise InsufficientHistoryError(f"no stock has a signal on {dates[rebalance].date()}")
    book = (row * spec.gross).astype("float64")
    book = book[book != 0].sort_index().rename("weight")
    book.index.name = "permno"
    return book


def _banded_row(spec: StrategySpec, panel: Panel, a: int, rebalance: int, need: int) -> pd.Series:
    """The gross-1 book on row `rebalance` under the spec's no-trade band.

    A banded book depends on the book before it, so it is the backtest's
    `band_book` replayed from the anchor (flat there) over every anchored
    rebalance, on evaluation's target weights. The signal is computed from
    `need` rows before the anchor (or the panel's first row), so each target
    is evaluation's. NaN when the rebalance row has no signal at all.
    """
    first = max(0, a - need)
    window = _rows(panel, first, rebalance + 1)
    neutral = spec.trading.neutral
    traded = trading_signal(factor_signal(spec.tree, window), window, neutral)
    if traded.iloc[-1].isna().all():
        return traded.iloc[-1]
    weights = to_weights(traded, demean=neutral == "none")
    path = band_book(
        weights.iloc[a - first :],
        traded.isna().iloc[a - first :],
        band=spec.trading.band,
        hold=spec.trading.hold,
    )
    return path.iloc[-1]
