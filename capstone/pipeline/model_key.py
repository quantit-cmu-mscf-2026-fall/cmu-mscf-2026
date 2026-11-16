"""Whether the model API key is in an environment, never its value.

The model client reads the key from `API_KEY_ENV`. The autopilot package never names that
variable; its read-only `preflight` asks this module, outside the package, whether the
variable is SET or UNSET. Nothing here returns, prints or logs the value.
"""

from __future__ import annotations

from collections.abc import Mapping

API_KEY_ENV = "ANTHROPIC_API_KEY"


def key_state(env: Mapping[str, str]) -> str:
    """SET when `API_KEY_ENV` holds a non-blank value in `env`, else UNSET."""
    value = env.get(API_KEY_ENV)
    return "SET" if isinstance(value, str) and value.strip() else "UNSET"
