"""Operations around the tick: alerting and status aggregation.

Each module here may export `SECTIONS` (config section name -> schema, validated like a
stage's) and `after_tick(ctx, report)`, which the runner's caller passes as a hook; it runs
at the end of every tick, also when the run is stopped. A new module is picked up without
editing the core.
"""

from __future__ import annotations

import importlib
import pkgutil


def discover(package=None):
    """(config sections, after-tick hooks) of every module in the package."""
    package = package or importlib.import_module(__name__)
    sections, hooks = {}, []
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda info: info.name):
        module = importlib.import_module(f"{package.__name__}.{info.name}")
        for name, schema in getattr(module, "SECTIONS", {}).items():
            if name in sections:
                raise ValueError(f"ops config section [{name}] is declared twice")
            sections[name] = schema
        if hasattr(module, "after_tick"):
            hooks.append(module.after_tick)
    return sections, hooks
