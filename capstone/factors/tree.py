"""Factor expressions as syntax trees: a closed grammar, two canonical keys.

A factor is a string in a small grammar over daily stock fields (`FIELDS`),
parsed into an immutable tree. `parse` is the only way in for a string: anything outside
the grammar raises `ParseError`, and nothing calls eval or exec.

Two canonical keys, for two different questions:

- `identity_key` asks "is this the same factor?" It keeps windows and
  constants and ignores operand order under `+` and `*`, so
  `ts_mean(close + volume, 5)` and `ts_mean(volume + close, 5)` share a
  `factor_id`, while `ts_mean(close, 5)` and `ts_mean(close, 20)` do not.
- `structural_key` asks "is this the same shape?" It also drops windows and
  constants. Originality (AlphaAgent, Tang et al. 2025, Eq. 5-6) is the
  largest subtree a candidate shares with a reference zoo, compared by shape,
  so changing a lookback does not make a copied alpha original.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

# Daily fields, all available in the CRSP CIZ pull:
#   open, high, low, close, volume: the day's prices and share volume;
#   returns: CRSP's total return (dlyret), dividends included;
#   shares: shares outstanding (shrout), so turnover is volume / shares;
#   cap: market capitalisation, abs(close) * shares.
# Whatever evaluates factors must split-adjust prices and shares: a ratio of
# raw closes across a split is not a return.
OHLCV = ("open", "high", "low", "close", "volume")
FIELDS = (*OHLCV, "returns", "shares", "cap")

# func name -> takes a window argument. Windows must be integer literals.
FUNCS = {
    "shift": True,
    "delta": True,
    "ts_mean": True,
    "ts_std": True,
    "ts_min": True,
    "ts_max": True,
    "ts_sum": True,
    "ts_rank": True,
    "ts_argmax": True,
    "ts_argmin": True,
    "ts_product": True,
    "decay_linear": True,
    "ema": True,
    "zscore": True,
    "rank": False,
    "log": False,
    "abs": False,
    "sign": False,
    "scale": False,
    "winsorize": False,
}
BINARY_WINDOW_FUNCS = {"ts_corr", "ts_cov"}
BINARY_PARAMETER_FUNCS = {"signed_power"}

# Five trading years, the horizon of long-run reversal (De Bondt & Thaler
# 1985). At 252 a three-to-five-year lookback had to be written as nested
# shifts. Originality is unaffected: the shape key ignores windows.
WINDOW_MIN, WINDOW_MAX = 1, 1260

# A parser must not be the thing a pathological input exhausts.
MAX_EXPRESSION_NODES = 200


class ParseError(ValueError):
    """Raised when an expression is not valid factor grammar."""


# ---------------------------------------------------------------------------
# Tree


@dataclass(frozen=True)
class Node:
    pass


@dataclass(frozen=True)
class Field_(Node):
    name: str


@dataclass(frozen=True)
class Const(Node):
    value: float


@dataclass(frozen=True)
class Unary(Node):
    op: str  # only "neg"
    operand: Node


@dataclass(frozen=True)
class BinOp(Node):
    op: str  # + - * /
    left: Node
    right: Node


@dataclass(frozen=True)
class Call(Node):
    func: str
    arg: Node
    window: int | None = None
    arg2: Node | None = None


def walk(node: Node) -> Iterator[Node]:
    """Every node of the tree, parents before children."""
    yield node
    if isinstance(node, Unary):
        yield from walk(node.operand)
    elif isinstance(node, BinOp):
        yield from walk(node.left)
        yield from walk(node.right)
    elif isinstance(node, Call):
        yield from walk(node.arg)
        if node.arg2 is not None:
            yield from walk(node.arg2)


def node_count(node: Node) -> int:
    """Symbolic length: the number of nodes in the tree."""
    return sum(1 for _ in walk(node))


def param_count(node: Node) -> int:
    """Free parameters: window lengths and numeric constants."""
    return sum(1 for n in walk(node) if isinstance(n, Const) or (isinstance(n, Call) and n.window))


def features_used(node: Node) -> frozenset[str]:
    """The raw fields the expression reads."""
    return frozenset(n.name for n in walk(node) if isinstance(n, Field_))


def funcs_used(node: Node) -> frozenset[str]:
    """The functions and operators the expression uses."""
    out = {n.func for n in walk(node) if isinstance(n, Call)}
    if any(isinstance(n, Unary) for n in walk(node)):
        out.add("neg")
    out.update(n.op for n in walk(node) if isinstance(n, BinOp))
    return frozenset(out)


# ---------------------------------------------------------------------------
# Parser: a closed grammar, and the only way in for an expression string.

# Exponents are accepted so that every constant `unparse` writes reads back.
_TOKEN = re.compile(
    r"\s*(?:(?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)"
    r"|(?P<name>\$?[A-Za-z_][A-Za-z_0-9]*)|(?P<op>[-+*/(),]))"
)


def _tokenize(text: str) -> list[tuple[str, str]]:
    text = text.strip()
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN.match(text, pos)
        if match is None:
            raise ParseError(f"illegal character {text[pos]!r} at position {pos}")
        if match.lastgroup == "num":
            tokens.append(("num", match.group("num")))
        elif match.lastgroup == "name":
            tokens.append(("name", match.group("name").lstrip("$").lower()))
        else:
            tokens.append(("op", match.group("op")))
        pos = match.end()
    return tokens


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> tuple[str, str]:
        token = self.peek()
        if token is None:
            raise ParseError("unexpected end of expression")
        self.pos += 1
        return token

    def expect(self, value: str) -> None:
        token = self.take()
        if token != ("op", value):
            raise ParseError(f"expected {value!r}, got {token[1]!r}")

    def expr(self) -> Node:
        node = self.term()
        while self.peek() in (("op", "+"), ("op", "-")):
            op = self.take()[1]
            node = BinOp(op, node, self.term())
        return node

    def term(self) -> Node:
        node = self.unary()
        while self.peek() in (("op", "*"), ("op", "/")):
            op = self.take()[1]
            node = BinOp(op, node, self.unary())
        return node

    def unary(self) -> Node:
        if self.peek() == ("op", "-"):
            self.take()
            return Unary("neg", self.unary())
        return self.atom()

    def atom(self) -> Node:
        kind, value = self.take()
        if kind == "num":
            return Const(float(value))
        if kind == "op" and value == "(":
            node = self.expr()
            self.expect(")")
            return node
        if kind == "name":
            if value in FIELDS and self.peek() != ("op", "("):
                return Field_(value)
            if value in FUNCS or value in BINARY_WINDOW_FUNCS or value in BINARY_PARAMETER_FUNCS:
                return self.call(value)
            raise ParseError(f"unknown identifier {value!r}")
        raise ParseError(f"unexpected token {value!r}")

    def window(self, func: str) -> int:
        kind, value = self.take()
        if kind != "num" or not value.isdigit():
            raise ParseError(f"{func} window must be an integer literal, got {value!r}")
        window = int(value)
        if not WINDOW_MIN <= window <= WINDOW_MAX:
            raise ParseError(f"{func} window {window} outside [{WINDOW_MIN}, {WINDOW_MAX}]")
        return window

    def call(self, func: str) -> Node:
        self.expect("(")
        arg = self.expr()
        arg2: Node | None = None
        window: int | None = None
        if func in BINARY_WINDOW_FUNCS:
            self.expect(",")
            arg2 = self.expr()
            self.expect(",")
            window = self.window(func)
        elif func in BINARY_PARAMETER_FUNCS:
            self.expect(",")
            kind, value = self.take()
            if kind != "num":
                raise ParseError(f"{func} exponent must be a numeric literal, got {value!r}")
            arg2 = Const(float(value))
        elif FUNCS[func]:
            self.expect(",")
            window = self.window(func)
        self.expect(")")
        return Call(func, arg, window, arg2)


def parse(expression: str) -> Node:
    """Parse a factor expression into a tree. The ONLY entry point for strings.

    Raises `ParseError` for anything outside the grammar: unknown identifiers
    or functions, quotes, attribute access, subscripts, non-literal windows,
    expressions that reference no market field, or expressions above the node
    cap.
    """
    tokens = _tokenize(expression)
    if not tokens:
        raise ParseError("empty expression")
    parser = _Parser(tokens)
    node = parser.expr()
    if parser.peek() is not None:
        raise ParseError(f"trailing input from token {parser.peek()[1]!r}")
    if node_count(node) > MAX_EXPRESSION_NODES:
        raise ParseError(f"expression exceeds {MAX_EXPRESSION_NODES} nodes")
    if not features_used(node):
        raise ParseError("expression references no market field")
    return node


def unparse(node: Node) -> str:
    """The tree as a fully parenthesised expression that `parse` reads back."""
    if isinstance(node, Field_):
        return node.name
    if isinstance(node, Const):
        return repr(node.value)
    if isinstance(node, Unary):
        return f"-({unparse(node.operand)})"
    if isinstance(node, BinOp):
        return f"({unparse(node.left)} {node.op} {unparse(node.right)})"
    if isinstance(node, Call):
        if node.arg2 is not None:
            if node.window is None:
                return f"{node.func}({unparse(node.arg)}, {unparse(node.arg2)})"
            return f"{node.func}({unparse(node.arg)}, {unparse(node.arg2)}, {node.window})"
        if node.window is None:
            return f"{node.func}({unparse(node.arg)})"
        return f"{node.func}({unparse(node.arg)}, {node.window})"
    raise TypeError(f"not a Node: {node!r}")


# ---------------------------------------------------------------------------
# Canonical keys


def _fold_shifts(node: Call) -> Call:
    """shift(shift(x, a), b) is shift(x, a + b): one factor, however it is written."""
    while isinstance(node.arg, Call) and node.arg.func == "shift":
        node = Call("shift", node.arg.arg, node.window + node.arg.window)
    return node


def _key(node: Node, keep_parameters: bool) -> str:
    if isinstance(node, Call) and node.func == "shift":
        node = _fold_shifts(node)
    if isinstance(node, Field_):
        return f"F:{node.name}"
    if isinstance(node, Const):
        return f"C:{node.value!r}" if keep_parameters else "C"
    if isinstance(node, Unary):
        return f"(neg {_key(node.operand, keep_parameters)})"
    if isinstance(node, BinOp):
        left, right = _key(node.left, keep_parameters), _key(node.right, keep_parameters)
        if node.op in "+*":
            left, right = sorted((left, right))
        return f"({node.op} {left} {right})"
    if isinstance(node, Call):
        window = f" w={node.window}" if keep_parameters and node.window is not None else ""
        second = "" if node.arg2 is None else f" {_key(node.arg2, keep_parameters)}"
        return f"({node.func}{window} {_key(node.arg, keep_parameters)}{second})"
    raise TypeError(f"not a Node: {node!r}")


def identity_key(node: Node) -> str:
    """Same factor, same key: windows and constants kept, `+`/`*` order ignored."""
    return _key(node, keep_parameters=True)


def structural_key(node: Node) -> str:
    """Same shape, same key: windows and constants dropped as well."""
    return _key(node, keep_parameters=False)


def factor_id(node: Node) -> str:
    """Stable 16-hex-digit id of a factor, from its identity key."""
    return hashlib.sha256(identity_key(node).encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Originality (AlphaAgent Eq. 5-6)


def _subtree_sizes(node: Node) -> dict[str, int]:
    return {structural_key(sub): node_count(sub) for sub in walk(node)}


def subtree_similarity(a: Node, b: Node) -> int:
    """Eq. 5: node count of the largest subtree shape two trees share."""
    sizes = _subtree_sizes(a)
    shared = sizes.keys() & _subtree_sizes(b).keys()
    return max((sizes[key] for key in shared), default=0)


def zoo_similarity(node: Node, zoo: Sequence[tuple[str, Node]]) -> tuple[int, float, str]:
    """Eq. 6: the candidate's largest subtree shared with any zoo member.

    Returns (shared node count, that count as a share of the candidate, name
    of the closest member). A share of 1.0 means the whole candidate is a
    piece of something already in the zoo.
    """
    best_raw, best_name = 0, ""
    for name, member in zoo:
        raw = subtree_similarity(node, member)
        if raw > best_raw:
            best_raw, best_name = raw, name
    return best_raw, best_raw / node_count(node), best_name
