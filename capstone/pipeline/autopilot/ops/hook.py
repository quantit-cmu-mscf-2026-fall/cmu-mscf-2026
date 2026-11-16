"""The operations hook the runner calls at the end of every tick (also when stopped).

`after_tick(ctx, report)`, in order:

1. alert (`alert`): `<run>/ALERT.json` / `.md` for every new STOP (and blocker, unless
   `[alert] on_blocked = false`), then the notifier;
2. warnings (`early_warnings`): the run's typed early warnings, deduplicated after
   acknowledgement;
3. status (`status`): `<run>/autopilot/ops/status.json` / `.md`.

File updates are atomic and resumable after a kill. A warning callback that completes
before its acknowledgement is saved can be retried, so notifier callbacks must handle
warning keys idempotently; notification delivery is not exactly once. What goes wrong
is a typed problem in
`report.problems` (`"stage": null`, codes in `common.REASONS`), never a traceback.

The notifier is `module:function` named in the config (or injected as
`services["notifier"]`); it is imported and called with a copy of the alert. It is code the
run's owner chose, so treat the config file like code. The default is no notifier.
"""

from __future__ import annotations

from capstone.pipeline.autopilot.config import Key
from capstone.pipeline.autopilot.ops import alert, early_warnings, status
from capstone.pipeline.autopilot.ops.common import problem

SECTIONS = {
    "alert": {
        "notifier": Key("str", None, "optional hook `module:function(alert: dict)`"),
        "on_blocked": Key("bool", True, "also alert when a tick ends blocked (not only STOP)"),
    },
}


def after_tick(ctx, report):
    alert.run(ctx, report)
    early_warnings.deliver(ctx, report)  # retries until acknowledgement is persisted
    try:
        status.write(ctx.config, list(report.states), journal=ctx.journal)
    except Exception as error:
        report.problems.append(problem("ops_status_failed", f"{type(error).__name__}: {error}"))
