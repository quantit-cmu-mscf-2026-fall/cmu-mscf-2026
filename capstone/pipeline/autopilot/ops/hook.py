"""The operations hook the runner calls at the end of every tick (also when stopped).

`after_tick(ctx, report)`, in order:

1. alert (`alert`): `<run>/ALERT.json` / `.md` for every new STOP (and blocker, unless
   `[alert] on_blocked = false`), then the notifier;
2. status (`status`): `<run>/autopilot/ops/status.json` / `.md`.

Each step is idempotent and resumable after a kill (its files are replaced atomically and
its progress is recorded before it moves on); what goes wrong is a typed problem in
`report.problems` (`"stage": null`, codes in `common.REASONS`), never a traceback.

The notifier is `module:function` named in the config (or injected as
`services["notifier"]`); it is imported and called with a copy of the alert. It is code the
run's owner chose, so treat the config file like code. The default is no notifier.
"""

from __future__ import annotations

from capstone.pipeline.autopilot.config import Key
from capstone.pipeline.autopilot.ops import alert, status
from capstone.pipeline.autopilot.ops.common import problem

SECTIONS = {
    "alert": {
        "notifier": Key("str", None, "optional hook `module:function(alert: dict)`"),
        "on_blocked": Key("bool", True, "also alert when a tick ends blocked (not only STOP)"),
    },
}


def after_tick(ctx, report):
    alert.run(ctx, report)
    try:
        status.write(ctx.config, list(report.states), journal=ctx.journal)
    except Exception as error:
        report.problems.append(problem("ops_status_failed", f"{type(error).__name__}: {error}"))
