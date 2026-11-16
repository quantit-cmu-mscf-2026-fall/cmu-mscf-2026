"""Scoped memo for pure functions of frozen replay inputs.

Inside `scope()` (one registered evaluation or portfolio study opens it) a value is
keyed by the *identity* of the input objects it was computed from plus plain values
(a session, an end date). An entry keeps weak references to those objects and is used
only while each one is still the very same object, so a panel rebuilt with any other
frame (`dataclasses.replace`), a deep copy, or a reused `id` never hits. Never keyed on
a session alone.

Bound objects must not be mutated in place inside the scope. That is checked, not
assumed: replay re-hashes all of its inputs on completion outside the memo
and refuses a changed input, so no result built on a
stale entry is ever reported. Outside a scope, or with `ENABLED = False` (the tests'
reference path), nothing is cached.
"""

from __future__ import annotations

import functools
import weakref
from contextlib import contextmanager

ENABLED = True
_state = {"entries": None, "owners": {}}


@contextmanager
def scope():
    """Open a memo scope unless one is open already (nested scopes share it)."""
    if _state["entries"] is not None or not ENABLED:
        yield
        return
    _state["entries"], _state["owners"] = {}, {}
    try:
        yield
    finally:
        _state["entries"], _state["owners"] = None, {}


def scoped(function):
    @functools.wraps(function)
    def run(*args, **kwargs):
        with scope():
            return function(*args, **kwargs)

    return run


@contextmanager
def bypass():
    """Compute without the memo (replay's completion check re-hashes honestly)."""
    saved, _state["entries"] = _state["entries"], None
    try:
        yield
    finally:
        _state["entries"] = saved


def active():
    return _state["entries"] is not None


def retain(kind, owner):
    """Keep `kind` entries for one owner at a time: a candidate's formations serve its
    three cost scenarios, then are freed when the next program's first one is built."""
    entries = _state["entries"]
    if entries is None or _state["owners"].get(kind) is owner:
        return
    for key in [key for key in entries if key[0] == kind]:
        del entries[key]
    _state["owners"][kind] = owner


def _ref(value):
    if value is None:
        return lambda: None
    try:
        return weakref.ref(value)
    except TypeError:  # not weak-referenceable: keep it alive instead
        return lambda: value


def cached(kind, objects, values, compute):
    """`compute()` memoized on (kind, values, identity of each object) inside a scope."""
    entries = _state["entries"]
    if entries is None:
        return compute()
    key = (kind, values, tuple(map(id, objects)))
    try:
        entry = entries.get(key)
    except TypeError:  # an unhashable argument: let the computation judge it
        return compute()
    if entry is not None and all(r() is o for r, o in zip(entry[0], objects, strict=True)):
        return entry[1]
    result = compute()
    entries[key] = ([_ref(obj) for obj in objects], result)
    return result


def panel_cached(kind, spec, panel, values, compute):
    """`cached` keyed on a spec, every panel input object and plain values."""
    if _state["entries"] is None:
        return compute()
    try:
        objects, names = panel_key(panel)
    except AttributeError:  # not a panel: the computation refuses it
        return compute()
    return cached(kind, (spec, *objects), (*values, *names), compute)


def panel_key(panel):
    """(objects, values) naming every input a panel computation can read."""
    fields = sorted((panel.fields or {}).items())
    availability = sorted((panel.availability or {}).items())
    objects = (
        panel.universe,
        panel.half_spread,
        panel.sic,
        panel.calendar,
        panel.company,
        panel.fundamentals,
        panel.sizing_prices,
        panel.provenance,
        *(frame for _, frame in fields),
        *(frame for _, frame in availability),
    )
    names = (tuple(n for n, _ in fields), tuple(n for n, _ in availability))
    return objects, (panel.data_version, *names)
