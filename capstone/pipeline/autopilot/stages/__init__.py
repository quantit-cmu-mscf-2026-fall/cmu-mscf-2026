"""Stage modules, one per owning unit: each exports `STAGES`.

A unit edits only its own module; `discover` finds every module in this package, so
adding a stage never touches the core.
"""

from __future__ import annotations

import importlib
import pkgutil


def discover(package=None):
    """Every `STAGES` entry of every module in the package, in module-name order."""
    package = package or importlib.import_module(__name__)
    found = []
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda info: info.name):
        module = importlib.import_module(f"{package.__name__}.{info.name}")
        found.extend(getattr(module, "STAGES", ()))
    return found
