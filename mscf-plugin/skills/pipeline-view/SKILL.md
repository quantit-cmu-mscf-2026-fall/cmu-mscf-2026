---
name: pipeline-view
description: Opens the live pipeline lab for cmu-mscf-2026 in a browser - a science room where each running experiment is a sample tube moving through ten instruments (hypothesis to incubation), changing colour at each step, with rejects in baskets under the gates - plus a ledger tab with trial count, funnel, holdout reads and hygiene, or a one-file snapshot. Also how to make an experiment report its progress (mscf_progress.track, pipeline_run.py). Use when the user wants to see, watch, monitor or visualize the pipeline, running experiments, the funnel or the ledger.
---

# Pipeline lab

`scripts/pipeline_view.py` (stdlib only) serves `scripts/view.html` on 127.0.0.1 and answers
`/state.json` on every request; the page polls every 1.5 s. It reads only the progress files,
`runs.jsonl` and git metadata. It never writes to the repository and never touches market data.

## The two tabs

**Lab.** Each tracked run is a sample tube on the room's transport track. The prep bench holds
stations 1-5 (notebook, sample freezer, stirrer, thermal cycler, label printer), a lift carries
tubes up to the validation shelf for stations 6-10 (centrifuge, stress chamber, microscope,
sealed glovebox, incubator), and finished runs slide into the results cabinet.

- An instrument animates only while a run is working at it; its display shows progress.
- When a tube reaches its next station a reagent drop falls in and the new colour diffuses
  through the old one (beige → brown → amber → orange → teal → blue → violet → magenta →
  gold; the wall shelf shows the order). From the ledger station on it carries a barcode,
  crystals grow through the gates, the holdout seals it and incubation turns it gold.
- Rejected runs tip into the wire basket under the gate that rejected them; crashed runs and
  runs with no heartbeat for 15 s (on hold) go to the basket under their station. A basket
  draws at most its 5 latest tubes; its label counts rejects from the ledger over a rolling
  window (`--reject-window-hours`, default 24).
- Three tubes fit in front of an instrument. More show as a `+N` badge (hover lists names), and
  the whiteboard names the busiest station. Runs that call `run.wait()` are shown as waiting
  (faded, with a "N waiting" tag) rather than working.
- The room's read-outs come from the ledger: whiteboard trial count, people, families and data
  version; the glovebox tag counts holdout reads (red above one); the cabinet counts shipped
  runs (`stage:incubation` + `verdict:pass`).
- Runs leave the room 30 minutes after they finish. The table under the scene lists every run.

**Ledger.** Trial count, funnel by `stage:` tag, trials over time, runs per experiment, holdout
reads and hygiene (runs missing a seed, `params.data_version` or metrics).

## Making a run show up

Wrap the experiment script body (preferred: it reports each station):

```python
from mscf_progress import track          # in scripts/; pipeline_run.py puts it on PYTHONPATH

with track("momentum-15-1") as run:
    run.stage("data");     daily = sd.load(...)
    run.stage("signal");   ...
    run.stage("backtest")
    run.wait("queued for a CPU")         # optional: waiting, not working
    for i, fold in enumerate(folds):
        run.progress((i + 1) / len(folds), f"fold {i + 1}")
    run.stage("ledger");   runlog.log_run("momentum-15-1", ..., tags=["stage:screen", "verdict:pass"])
```

or launch an unmodified script through the wrapper, which reports start, exit and whatever the
ledger records:

```bash
python "${CLAUDE_PLUGIN_ROOT}/skills/pipeline-view/scripts/pipeline_run.py" --name momentum-15-1 -- python experiments/momentum.py
```

Station ids come from `stations.json`. Once the ledger has an entry with the run's name and a
`stage:` tag, the run moves to that gate; `verdict:fail` drops it into the basket. Use the same
name in `track`/`--name` and `log_run`. Progress files live in `$MSCF_PROGRESS_DIR` (default
`~/.cache/mscf-pipeline/progress`), outside the repository. They hold the run name, id, pid,
user, timestamps, status and exit code, and per event the station, progress and a short note
(the exception class name on a crash, the script's file name under the wrapper, or the text you
pass to `run.stage(...)`): no exception messages, command arguments, conversation content, data
or results are recorded, and nothing is synced. If the directory cannot be written, tracking switches itself
off and the experiment carries on.

## Live (Claude Code on the user's machine)

Start it in the background from the repository root, then give the user the URL:

```bash
python "${CLAUDE_PLUGIN_ROOT}/skills/pipeline-view/scripts/pipeline_view.py" --repo . --port 8765
```

- `--ledger-dir DIR` for a scratch ledger (default `$CAPSTONE_LEDGER_DIR`, then `experiments/`);
  `--progress-dir DIR` for a non-default progress directory; `--reject-window-hours 168` for a
  weekly basket count.
- `http://127.0.0.1:8765/#ledger` opens on the ledger tab.
- Stop it with the background task's kill, not by editing anything.

## Snapshot (cloud sessions, sharing with the team)

A browser cannot reach a server inside a cloud session. Write one self-contained page instead
and send or publish it:

```bash
python "${CLAUDE_PLUGIN_ROOT}/skills/pipeline-view/scripts/pipeline_view.py" --repo . --snapshot ~/.cache/mscf-pipeline/pipeline-lab.html
```

A snapshot carries names and metrics from the private ledger: share it privately and never
commit it to the public repository. The viewer refuses a `--snapshot` path inside the repo. `--json` prints the state for scripting.

## Changing the room

The visual rules (palette, sample colours, type, motion, what to avoid) are in
`references/art-direction.md`; read it before drawing anything new. The room is drawn by
`scripts/build_scene.py` into `scripts/lab_scene.svg`: edit the generator, rerun it and commit
both (`tests/test_mscf_plugin_viewer.py` checks they match and that every id the page uses exists).

## Reading it for the user

Lead with what needs action: more than one holdout read, crashed or held runs, a station with a
long waiting queue, untagged runs, hygiene gaps. Then the trial count, since it feeds every
correction. If the Trials tile says `error`, the ledger has an unreadable line: tell the user and
give them `python -m capstone.runlog stats`. A run counts as a holdout read when it carries a
`holdout` tag, a `stage:final` tag, a `holdout_start` param, or an `end`/`holdout_end` after
2020-12-31.
