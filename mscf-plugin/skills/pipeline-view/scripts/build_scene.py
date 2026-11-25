# ruff: noqa: E501 -- SVG markup: one element per line reads better than wrapped attributes
"""Draws the static science-room scene (lab_scene.svg) for the pipeline view's production line.

    python build_scene.py            # rewrites lab_scene.svg next to this file

The room, the ten instruments and their animatable parts are static SVG; view.html moves the
sample tubes, switches instruments on (class "active"), updates displays and fills baskets.
Instrument i has id "inst-<i>", its display "lcd-<i>", its name plate lamp "lamp-<i>". Parts
with class "a-*" animate only while their instrument is active (see view.html CSS).
"""

from __future__ import annotations

from pathlib import Path

W, H = 1600, 930
LOWER_Y, UPPER_Y = 700, 420  # track heights: prep bench, validation shelf
XS = [250, 490, 730, 970, 1210]  # instrument centres; the upper tier runs right to left
LIFT_X = 1479
ARCHIVE = (20, UPPER_Y - 190)
HEIGHT = {0: 100, 1: 128, 2: 112, 3: 126, 4: 104, 5: 86, 6: 140, 7: 150, 8: 138, 9: 150}
NAMES = [
    "Hypothesis",
    "Load data",
    "Build signal",
    "Backtest",
    "Log to ledger",
    "Screen",
    "Robustness",
    "Incremental",
    "Holdout",
    "Incubation",
]

C = dict(
    wall="#dde6e3",
    tile="#cfdad6",
    floor="#bfc2b6",
    epoxy="#2d3538",
    epoxy_edge="#3d474b",
    cab="#e9edeb",
    cab_line="#c9d1ce",
    steel="#a9b4b8",
    steel_dk="#7e8a8f",
    enamel="#f5f7f6",
    enamel_sh="#dfe5e3",
    band="#2f6f73",
    lcd="#1f2b24",
    ink="#1f2a2c",
    muted="#5f6e70",
    amber="#f2c14e",
    reject="#d4583f",
)


def centre(i: int) -> tuple[int, int]:
    """Instrument centre x and the track height it sits on."""
    return (XS[i], LOWER_Y) if i < 5 else (XS[4 - (i - 5)], UPPER_Y)


def lcd(i, x, y, w, text):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="18" rx="2" fill="{C["lcd"]}"/>'
        f'<text id="lcd-{i}" x="{x + w - 6}" y="{y + 13}" class="lcd" text-anchor="end">{text}</text>'
    )


def instrument(i: int) -> str:
    cx, y = centre(i)
    base = y - 6
    e, sh, band, st, sd = C["enamel"], C["enamel_sh"], C["band"], C["steel"], C["steel_dk"]
    g: list[str] = []
    if i == 0:  # notebook stand: the hypothesis is written down (pen line draws while active)
        g += [
            f'<rect x="{cx - 70}" y="{base - 96}" width="78" height="92" rx="3" fill="#f8f4e8" stroke="#cdbf9c" transform="rotate(-4 {cx - 30} {base - 50})"/>',
            *[
                f'<line x1="{cx - 62}" x2="{cx - 4}" y1="{base - 80 + k * 10}" y2="{base - 82 + k * 10}" stroke="#b9c6d6"/>'
                for k in range(7)
            ],
            f'<path class="a-write" pathLength="100" d="M{cx - 58} {base - 46} l12 -10 10 6 14 -18" stroke="{band}" stroke-width="1.8" fill="none"/>',
            f'<g class="a-pen"><path d="M{cx - 24} {base - 64} l22 -30 4 3 -22 30z" fill="{C["ink"]}"/></g>',
            f'<rect x="{cx + 20}" y="{base - 30}" width="52" height="26" rx="3" fill="{sh}"/>',
            *[
                f'<path d="M{cx + 28 + k * 12} {base - 30} v-50 l3 -10 3 10 v50" fill="{e}" stroke="{sd}"/>'
                for k in range(4)
            ],
            lcd(i, cx + 18, base - 20, 56, ""),
        ]
    if i == 1:  # sample freezer: frozen data comes out (vapour while open)
        g += [
            f'<rect x="{cx - 66}" y="{base - 128}" width="132" height="124" rx="6" fill="{e}" stroke="{sh}"/>',
            f'<rect x="{cx - 56}" y="{base - 118}" width="96" height="104" rx="3" fill="#cfe3ea" stroke="{st}"/>',
            *[
                f'<line x1="{cx - 56}" x2="{cx + 40}" y1="{base - 92 + k * 26}" y2="{base - 92 + k * 26}" stroke="{st}"/>'
                for k in range(3)
            ],
            *[
                f'<rect class="a-rack" style="--d:{(r * 8 + k) * 0.07:.2f}s" x="{cx - 48 + k * 11}" y="{base - 112 + r * 26}" width="6" height="16" rx="2" fill="#cfc6b4"/>'
                for r in range(4)
                for k in range(8)
            ],
            *[
                f'<path class="a-vapor" style="--d:{k * 0.5}s" d="M{cx - 40 + k * 26} {base - 118} q-6 -10 0 -18 t0 -18" stroke="#ffffff" stroke-width="5" stroke-linecap="round" fill="none" opacity="0"/>'
                for k in range(4)
            ],
            f'<rect x="{cx + 46}" y="{base - 100}" width="5" height="40" rx="2" fill="{sd}"/>',
            lcd(i, cx + 40, base - 124, 26, "-80"),
        ]
    if i == 2:  # stirrer hotplate: the signal is mixed (stir bar spins, vortex turns, bubbles)
        g += [
            f'<rect x="{cx - 64}" y="{base - 34}" width="128" height="30" rx="5" fill="{e}" stroke="{sh}"/>',
            f'<circle cx="{cx - 44}" cy="{base - 19}" r="7" fill="{sh}" stroke="{sd}"/><circle cx="{cx + 44}" cy="{base - 19}" r="7" fill="{sh}" stroke="{sd}"/>',
            f'<path d="M{cx - 34} {base - 112} h68 v74 a6 6 0 0 1 -6 6 h-56 a6 6 0 0 1 -6 -6z" fill="#fff" fill-opacity=".5" stroke="#8b9a9e" stroke-width="1.4"/>',
            f'<path d="M{cx - 33} {base - 74} q17 -8 34 0 t33 0 v36 a6 6 0 0 1 -6 6 h-56 a6 6 0 0 1 -6 -6z" fill="#c9a14a" opacity=".85"/>',
            f'<g class="a-swirl" style="transform-origin:{cx}px {base - 56}px"><path d="M{cx - 20} {base - 56} q20 -14 40 0 q-20 14 -40 0" stroke="#fff4d6" stroke-width="1.4" fill="none" opacity=".85"/></g>',
            f'<rect class="a-stir" style="transform-origin:{cx}px {base - 40}px" x="{cx - 10}" y="{base - 42}" width="20" height="4" rx="2" fill="{C["ink"]}"/>',
            *[
                f'<circle class="a-bub" style="--d:{k * 0.45}s" cx="{cx - 18 + k * 12}" cy="{base - 40}" r="2" fill="#fff8e1" opacity="0"/>'
                for k in range(4)
            ],
            lcd(i, cx - 22, base - 30, 44, ""),
        ]
    if i == 3:  # thermal cycler: the backtest runs its passes (wells cycle hot and cool)
        g += [
            f'<rect x="{cx - 70}" y="{base - 92}" width="140" height="88" rx="8" fill="{e}" stroke="{sh}"/>',
            f'<path d="M{cx - 64} {base - 92} q64 -34 128 0z" fill="{sh}" stroke="{sd}"/>',
            *[
                f'<path class="a-heat" style="--d:{k * 0.4}s" d="M{cx - 30 + k * 30} {base - 112} q-5 -8 0 -16 t0 -16" stroke="#e9a36b" stroke-width="2" fill="none" opacity="0"/>'
                for k in range(3)
            ],
            f'<rect x="{cx - 70}" y="{base - 48}" width="140" height="6" fill="{band}"/>',
            lcd(i, cx - 54, base - 78, 108, ""),
            *[
                f'<circle class="a-well" style="--d:{k * 0.18:.2f}s" cx="{cx - 49 + k * 14}" cy="{base - 24}" r="3.4" fill="{sh}"/>'
                for k in range(8)
            ],
        ]
    if i == 4:  # label printer and scanner: the ledger number goes on before anyone looks
        g += [
            f'<rect x="{cx - 66}" y="{base - 70}" width="96" height="66" rx="6" fill="{e}" stroke="{sh}"/>',
            f'<g class="a-feed"><path d="M{cx - 40} {base - 52} h44 v22 h-44z" fill="#fbfbf7" stroke="#c9cfcc"/>'
            + "".join(
                f'<line x1="{cx - 34 + k * 4}" x2="{cx - 34 + k * 4}" y1="{base - 47}" y2="{base - 37}" stroke="{C["ink"]}" stroke-width="{1 + (k % 3) * 0.6:.1f}"/>'
                for k in range(9)
            )
            + "</g>",
            f'<rect x="{cx - 56}" y="{base - 60}" width="76" height="8" rx="2" fill="{C["ink"]}"/>',
            f'<rect x="{cx + 36}" y="{base - 104}" width="22" height="100" rx="3" fill="{sd}"/>',
            f'<path class="a-beam" style="transform-origin:{cx + 38}px {base - 86}px" d="M{cx + 38} {base - 86} l-40 34" stroke="{C["reject"]}" stroke-width="1.8" opacity="0"/>',
            lcd(i, cx - 56, base - 24, 76, ""),
        ]
    if i == 5:  # centrifuge: spin the candidates and see what separates
        g += [
            f'<g class="a-shake" style="transform-origin:{cx}px {base}px">',
            f'<rect x="{cx - 70}" y="{base - 70}" width="140" height="66" rx="10" fill="{e}" stroke="{sh}"/>',
            f'<ellipse cx="{cx}" cy="{base - 70}" rx="62" ry="16" fill="{sh}" stroke="{sd}"/>',
            f'<ellipse class="a-rotor" cx="{cx}" cy="{base - 72}" rx="44" ry="10" fill="#c7d3d6" stroke="{C["ink"]}" stroke-width="2" stroke-dasharray="10 12"/>',
            f'<rect x="{cx - 70}" y="{base - 36}" width="140" height="5" fill="{band}"/>',
            lcd(i, cx - 44, base - 26, 88, ""),
            "</g>",
        ]
    if i == 6:  # environment chamber: stress under other costs, lags and sub-periods (mist drifts)
        g += [
            f'<rect x="{cx - 68}" y="{base - 140}" width="136" height="136" rx="6" fill="{e}" stroke="{sh}"/>',
            f'<clipPath id="chamber-clip"><rect x="{cx - 58}" y="{base - 128}" width="92" height="112" rx="4"/></clipPath>',
            f'<rect x="{cx - 58}" y="{base - 128}" width="92" height="112" rx="4" fill="#d9ecef" stroke="{st}"/>',
            '<g clip-path="url(#chamber-clip)"><g class="a-fog">'
            + "".join(
                f'<path d="M{cx - 150 + k * 60} {base - 110 + r * 30} q15 -8 30 0 t30 0" stroke="#ffffff" stroke-width="6" stroke-linecap="round" fill="none" opacity=".75"/>'
                for k in range(6)
                for r in range(3)
            )
            + "</g></g>",
            *[
                f'<line x1="{cx - 58}" x2="{cx + 34}" y1="{base - 100 + k * 28}" y2="{base - 100 + k * 28}" stroke="{st}"/>'
                for k in range(3)
            ],
            lcd(i, cx + 38, base - 128, 26, ""),
            f'<text x="{cx + 51}" y="{base - 98}" class="micro" text-anchor="middle">fold</text>',
            f'<circle cx="{cx + 51}" cy="{base - 76}" r="6" fill="{sh}" stroke="{sd}"/>',
        ]
    if i == 7:  # microscope: what it adds beyond known factors (lamp pulses, specimen glows)
        g += [
            f'<path class="a-light" d="M{cx - 6} {base - 46} l-18 30 h48 l-18 -30z" fill="#fff2b3" opacity="0"/>',
            f'<rect x="{cx - 52}" y="{base - 16}" width="104" height="12" rx="4" fill="{C["ink"]}"/>',
            f'<path d="M{cx + 22} {base - 16} v-60 q0 -24 -24 -34" stroke="{C["ink"]}" stroke-width="14" fill="none" stroke-linecap="round"/>',
            f'<rect x="{cx - 34}" y="{base - 48}" width="56" height="6" rx="2" fill="{sd}"/>',
            f'<g transform="rotate(-28 {cx - 10} {base - 110})"><rect x="{cx - 18}" y="{base - 138}" width="16" height="56" rx="4" fill="{e}" stroke="{sd}"/><rect x="{cx - 22}" y="{base - 146}" width="24" height="10" rx="3" fill="{C["ink"]}"/></g>',
            f'<rect x="{cx - 6}" y="{base - 80}" width="12" height="22" rx="2" fill="{sd}"/>',
            f'<circle class="a-knob" style="transform-origin:{cx + 22}px {base - 38}px" cx="{cx + 22}" cy="{base - 38}" r="9" fill="{sh}" stroke="{sd}" stroke-dasharray="4 3"/>',
            f'<g transform="translate({cx - 66} {base - 150})"><circle r="24" fill="#f4f8f6" stroke="{sd}" stroke-width="2"/>'
            + "".join(
                f'<circle class="a-cell" style="--d:{k * 0.3:.1f}s" cx="{(-12 + (k * 7) % 24)}" cy="{(-10 + (k * 11) % 20)}" r="{2 + k % 3}" fill="#7a5fd0" opacity=".25"/>'
                for k in range(7)
            )
            + "</g>",
        ]
    if i == 8:  # sealed glovebox: the holdout is opened once (gloves work while active)
        g += [
            f'<path d="M{cx - 74} {base - 4} v-110 l14 -24 h120 l14 24 v110z" fill="#e7eef0" stroke="{st}" stroke-width="2"/>',
            f'<path d="M{cx - 64} {base - 112} l10 -18 h108 l10 18" fill="none" stroke="{st}"/>',
            f'<circle cx="{cx - 30}" cy="{base - 58}" r="17" fill="{C["ink"]}"/><circle cx="{cx + 30}" cy="{base - 58}" r="17" fill="{C["ink"]}"/>',
            f'<path class="a-glove" style="transform-origin:{cx - 30}px {base - 58}px" d="M{cx - 30} {base - 58} q6 22 -4 34" stroke="#3b4a4d" stroke-width="10" fill="none" stroke-linecap="round"/>',
            f'<path class="a-glove r" style="transform-origin:{cx + 30}px {base - 58}px" d="M{cx + 30} {base - 58} q-6 22 4 34" stroke="#3b4a4d" stroke-width="10" fill="none" stroke-linecap="round"/>',
            f'<g transform="translate({cx - 30} {base - 126})"><rect id="seal-bg" width="60" height="14" rx="2" fill="{C["band"]}"/><text id="seal-8" x="30" y="10.5" class="micro w" text-anchor="middle">opened 0×</text></g>',
        ]
    if i == 9:  # incubator: what survives gets paper capital and time (warm light breathes)
        g += [
            f'<rect x="{cx - 68}" y="{base - 150}" width="136" height="146" rx="8" fill="{e}" stroke="{sh}"/>',
            f'<rect x="{cx - 58}" y="{base - 138}" width="98" height="122" rx="4" fill="#fff3d6" stroke="{st}"/>',
            f'<rect class="a-warm" x="{cx - 58}" y="{base - 138}" width="98" height="122" rx="4" fill="{C["amber"]}" opacity=".12"/>',
            *[
                f'<line x1="{cx - 58}" x2="{cx + 40}" y1="{base - 98 + k * 40}" y2="{base - 98 + k * 40}" stroke="{st}"/>'
                for k in range(2)
            ],
            *[
                f'<path d="M{cx - 46 + k * 26} {base - 100} l6 -22 h8 l6 22z" fill="#fbe9b6" stroke="#d9b45a"/>'
                for k in range(3)
            ],
            *[
                f'<circle class="a-bub" style="--d:{k * 0.6:.1f}s" cx="{cx - 36 + k * 26}" cy="{base - 104}" r="1.8" fill="#fff" opacity="0"/>'
                for k in range(3)
            ],
            lcd(i, cx + 42, base - 138, 24, "37°"),
        ]
    plate = (
        f'<g transform="translate({cx - 66} {base - HEIGHT[i] - 36})"><rect width="132" height="26" rx="4" fill="{C["ink"]}"/>'
        f'<text x="10" y="17.5" class="pnum">{i + 1}</text><text x="28" y="17.5" class="pname">{NAMES[i]}</text>'
        f'<circle id="lamp-{i}" cx="121" cy="13" r="3.8" fill="#9aa5a8"/></g>'
    )
    return f'<g id="inst-{i}" class="inst">{"".join(g)}{plate}</g>'


def room() -> str:
    s = [f'<rect width="{W}" height="{H}" fill="{C["wall"]}"/>']
    s += [
        f'<line x1="0" x2="{W}" y1="{y}" y2="{y}" stroke="{C["tile"]}"/>'
        for y in range(20, LOWER_Y + 30, 26)
    ]
    s += [
        f'<line y1="0" y2="{LOWER_Y + 30}" x1="{x}" x2="{x}" stroke="{C["tile"]}" opacity=".35"/>'
        for x in range(0, W, 52)
    ]
    # window
    s.append(
        '<rect x="1150" y="34" width="250" height="150" rx="4" fill="#38456e"/><rect x="1150" y="124" width="250" height="60" fill="#e7a76f" opacity=".55"/>'
        '<path d="M1150 166 h18 v-14 h12 v-10 h10 v24 h16 v-18 h14 v14 h22 v-26 h10 v-6 h8 v32 h20 v-12 h16 v16 h24 v-22 h12 v22 h30 v-10 h20 v28 h-250z" fill="#28335a"/>'
        + "".join(
            f'<rect x="{1166 + k * 38}" y="{156 + (k % 3) * 4}" width="3" height="4" fill="{C["amber"]}"/>'
            for k in range(6)
        )
        + '<rect x="1144" y="28" width="262" height="162" rx="6" fill="none" stroke="#f1f4f3" stroke-width="10"/><line x1="1275" x2="1275" y1="30" y2="188" stroke="#f1f4f3" stroke-width="6"/>'
    )
    # whiteboard: the ledger lives in the room
    s.append(
        '<g transform="translate(176 40)"><rect width="320" height="150" rx="4" fill="#fbfcfc" stroke="#b9c3c0" stroke-width="3"/>'
        '<text x="20" y="34" class="board">trials logged</text><text id="wb-trials" x="20" y="94" class="boardbig">0</text>'
        '<text id="wb-sub" x="20" y="120" class="board s"></text><text id="wb-busy" x="20" y="138" class="board s warn"></text>'
        '<path d="M190 76 l18 -14 16 8 22 -26 22 6" stroke="#2f6f73" stroke-width="2.4" fill="none" stroke-linecap="round"/>'
        '<rect x="40" y="150" width="70" height="7" rx="2" fill="#9aa5a8"/><rect x="220" y="148" width="14" height="6" rx="2" fill="#d4583f"/><rect x="240" y="148" width="14" height="6" rx="2" fill="#2f6f73"/></g>'
    )
    # clock (hands set by view.html to the viewer's time)
    s.append(
        '<g transform="translate(580 106)"><circle r="34" fill="#fbfcfc" stroke="#7e8a8f" stroke-width="4"/>'
        + "".join(
            f'<line x1="0" y1="-27" x2="0" y2="-22" stroke="#1f2a2c" stroke-width="2" transform="rotate({k * 30})"/>'
            for k in range(12)
        )
        + '<line id="clock-h" x1="0" y1="0" x2="0" y2="-18" stroke="#1f2a2c" stroke-width="3"/><line id="clock-m" x1="0" y1="0" x2="0" y2="-26" stroke="#1f2a2c" stroke-width="2"/><circle r="2.5" fill="#1f2a2c"/></g>'
    )
    # reagent shelf: the colours a sample goes through, in order
    liq = [
        "#cfc6b4",
        "#a98a62",
        "#c9a14a",
        "#d07a3a",
        "#3f9c8f",
        "#2e86a8",
        "#5a7fd6",
        "#7a5fd0",
        "#a14fb5",
        "#e8b931",
    ]
    s.append(
        '<rect x="680" y="160" width="420" height="8" fill="#b8c2c6"/><rect x="694" y="168" width="6" height="16" fill="#9aa5a8"/><rect x="1080" y="168" width="6" height="16" fill="#9aa5a8"/>'
        + "".join(
            f'<g transform="translate({704 + k * 40} 160)"><rect x="-12" y="{-40 - (k % 3) * 8}" width="24" height="{40 + (k % 3) * 8}" rx="4" fill="{liq[k]}" opacity=".9"/><rect x="-6" y="{-48 - (k % 3) * 8}" width="12" height="8" rx="2" fill="#3f4a4d"/><text x="0" y="{-12 - (k % 3) * 2}" class="micro" text-anchor="middle">{k + 1}</text></g>'
            for k in range(10)
        )
    )
    # shelf, bench, cabinets, floor
    s.append(
        f'<rect x="120" y="{UPPER_Y}" width="1280" height="12" fill="{C["steel"]}"/><rect x="120" y="{UPPER_Y + 12}" width="1280" height="3" fill="{C["steel_dk"]}"/>'
    )
    s += [
        f'<path d="M{x} {UPPER_Y + 15} v40 l-26 -40z" fill="{C["steel_dk"]}" opacity=".8"/>'
        for x in (170, 610, 850, 1090, 1380)
    ]
    s.append(
        f'<rect x="60" y="{LOWER_Y}" width="1460" height="22" fill="{C["epoxy"]}"/><rect x="60" y="{LOWER_Y}" width="1460" height="4" fill="{C["epoxy_edge"]}"/>'
    )
    s.append(f'<rect x="60" y="{LOWER_Y + 22}" width="1460" height="150" fill="{C["cab"]}"/>')
    s += [
        f'<rect x="{x}" y="{LOWER_Y + 30}" width="112" height="134" rx="3" fill="none" stroke="{C["cab_line"]}"/><rect x="{x + 46}" y="{LOWER_Y + 40}" width="20" height="4" rx="2" fill="{C["steel_dk"]}"/>'
        for x in range(78, 1500, 120)
    ]
    s.append(
        f'<rect x="0" y="{LOWER_Y + 172}" width="{W}" height="{H - LOWER_Y - 172}" fill="{C["floor"]}"/>'
    )
    # transport track on both tiers and the lift between them
    s.append(
        f'<rect x="110" y="{LOWER_Y - 4}" width="1400" height="6" rx="3" fill="#56666b"/><rect x="140" y="{UPPER_Y - 4}" width="1370" height="6" rx="3" fill="#56666b"/>'
    )
    s.append(
        f'<line class="track-flow lower" x1="120" x2="1500" y1="{LOWER_Y - 1}" y2="{LOWER_Y - 1}" stroke="#8b9a9e" stroke-dasharray="4 6"/>'
        f'<line class="track-flow upper" x1="1500" x2="150" y1="{UPPER_Y - 1}" y2="{UPPER_Y - 1}" stroke="#8b9a9e" stroke-dasharray="4 6"/>'
    )
    s.append(
        f'<rect x="{LIFT_X - 27}" y="{UPPER_Y - 10}" width="8" height="{LOWER_Y - UPPER_Y + 14}" fill="{C["steel_dk"]}"/><rect x="{LIFT_X + 19}" y="{UPPER_Y - 10}" width="8" height="{LOWER_Y - UPPER_Y + 14}" fill="{C["steel_dk"]}"/>'
        f'<rect x="{LIFT_X - 33}" y="{UPPER_Y - 22}" width="66" height="14" rx="3" fill="{C["ink"]}"/><text x="{LIFT_X}" y="{UPPER_Y - 12}" class="micro w" text-anchor="middle">lift</text>'
        f'<rect id="lift-car" x="{LIFT_X - 21}" y="{LOWER_Y - 4}" width="42" height="6" rx="2" fill="#56666b"/>'
    )
    # results archive: shipped runs end up here (count and glowing tubes set by view.html)
    ax, ay = ARCHIVE
    s.append(
        f'<rect x="{ax}" y="{ay}" width="96" height="190" rx="6" fill="{C["enamel"]}" stroke="{C["enamel_sh"]}"/><rect x="{ax + 8}" y="{ay + 10}" width="80" height="150" rx="3" fill="#2b3640"/>'
        + "".join(
            f'<line x1="{ax + 8}" x2="{ax + 88}" y1="{ay + 60 + k * 50}" y2="{ay + 60 + k * 50}" stroke="#56666b"/>'
            for k in range(2)
        )
        + f'<g id="archive-slots"></g><text id="archive-n" x="{ax + 48}" y="{ay + 176}" class="arch" text-anchor="middle">shipped 0</text>'
    )
    return "".join(s)


def baskets() -> str:
    """One wire basket per station: under the shelf for the gates, on the cabinet front for prep stations."""
    out = []
    for i in range(10):
        cx, y = centre(i)
        top = y + 60
        wires = "".join(
            f'<line x1="{cx - 42 + k * 10.5:.1f}" y1="{top + 2}" x2="{cx - 38 + k * 9.5:.1f}" y2="{top + 34}" stroke="{C["steel_dk"]}" stroke-width=".8" opacity=".7"/>'
            for k in range(9)
        )
        out.append(
            f'<g id="basket-{i}" class="basket" visibility="hidden"><path d="M{cx - 46} {top} l6 36 h80 l6 -36" fill="#ffffff" fill-opacity=".25" stroke="{C["steel_dk"]}" stroke-width="1.6"/>{wires}'
            f'<path d="M{cx - 46} {top} h92" stroke="{C["steel_dk"]}" stroke-width="2.4"/><g id="basket-tubes-{i}"></g>'
            f'<text id="basket-label-{i}" x="{cx}" y="{top + 50}" class="blabel" text-anchor="middle"></text></g>'
        )
    return "".join(out)


def scene() -> str:
    body = room() + "".join(instrument(i) for i in range(10)) + baskets()
    return (
        f'<svg id="lab" viewBox="0 0 {W} {H}" role="img" aria-label="Pipeline as a science room: sample tubes move between ten instruments">'
        '<defs><filter id="diffuse" x="-50%" y="-50%" width="200%" height="200%"><feTurbulence type="fractalNoise" baseFrequency=".09" numOctaves="2" seed="3" result="n"/>'
        '<feDisplacementMap in="SourceGraphic" in2="n" scale="5"/></filter></defs>'
        f'{body}<g id="tubes"></g></svg>'
    )


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "lab_scene.svg"
    out.write_text(scene(), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
