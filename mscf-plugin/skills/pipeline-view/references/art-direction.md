# Art direction: the pipeline lab

The design system for anything drawn in the pipeline view. Follow it when adding an instrument,
a state or a new view, so the room stays one place rather than a collage.

## The idea

A working science room, seen from the front. Each run is a physical sample that changes as it
is processed. Numbers live on objects in the room (whiteboard, glovebox tag, cabinet, basket
labels) instead of in dashboard tiles. One thing moves at a time where it means something: a
tube travelling, an instrument working, a colour reacting.

## Palette

| Role | Hex | Use |
|---|---|---|
| Wall tile | `#dde6e3` / grout `#cfdad6` | background; never a pure white or cream |
| Bench epoxy | `#2d3538` / edge `#3d474b` | the black resin bench top |
| Cabinet | `#e9edeb` / line `#c9d1ce` | under-bench doors |
| Enamel | `#f5f7f6` / shade `#dfe5e3` | instrument bodies |
| Steel | `#a9b4b8` / dark `#7e8a8f` | shelf, lift, baskets |
| Lab teal | `#2f6f73` | instrument bands, whiteboard ink |
| Ink | `#1f2a2c` | name plates, text |
| Display | `#1f2b24` with `#9be3b4` text | instrument read-outs only |
| Amber | `#f2c14e` | an instrument at work, gold samples |
| Reject red | `#d4583f` | rejects, crashes, the holdout seal above one read |
| Waiting orange | `#e08a3c` | waiting tags and lamps |

Sample colours, one per station, in order (also on the wall shelf):
`#cfc6b4 #a98a62 #c9a14a #d07a3a #3f9c8f #2e86a8 #5a7fd6 #7a5fd0 #a14fb5 #e8b931`.
Rejected samples cloud to `#a7a49a`, held ones to `#b9c0c2`. Status is never colour alone: each
carries a badge (✕, ‖, ✓) and a basket label.

## Type

- Overpass (600, 800) for every label, plate and the whiteboard: technical signage, not a
  default UI face.
- IBM Plex Mono 500 only inside instrument displays, where a monospaced read-out is what the
  real object would show.
- Both are SIL OFL fonts. They are not bundled here (no binary assets); if `scripts/assets/` holds
  the three woff2 files the page embeds them, otherwise it falls back to the system sans and
  monospace faces. Sentence case;
  no tracked-out capitals.

## Motion

- An instrument animates only while a run works at it; idle rooms are still.
- Travel eases with `cubic-bezier(.45,.05,.3,1)`; duration grows with distance (500 ms + 2.2 ms
  per pixel, capped at 5.2 s). The lift moves with the tube.
- The colour change is the focal moment: drop (0.5 s), diffusion front with a turbulence filter
  (1.3 s), fizz and a brief glow. Nothing else in the room should compete with it.
- `prefers-reduced-motion`: no travel or loops; tubes jump and colours switch instantly.
- Use the browser's own animation (CSS, Web Animations); no animation libraries in the repo.

## Adding an instrument or a state

1. Add the station to `stations.json` (id, label, detail, `ledger_stage` for gates).
2. Draw it in `build_scene.py` `instrument()`: a front view of real equipment, parts that move
   tagged `class="a-<verb>"`, a display with id `lcd-<i>` if it has one. Keep it under the
   180 px height of its neighbours.
3. Add the keyframes under `.inst.active` in `view.html` and its read-out in `lcdText`.
4. Rerun `python build_scene.py`, check a snapshot by eye, and run `tests/test_mscf_plugin_viewer.py`.

## Avoid

- Culturally specific motifs (origami cranes, seal stamps, flags, costume decoration): this is a
  joint American and Korean team, and the room should belong to neither.
- Dashboard tiles or cards inside the scene; put a number on an object instead.
- Decorative loops that run when nothing is happening.
- Copyrighted characters, brand logos or real people.
