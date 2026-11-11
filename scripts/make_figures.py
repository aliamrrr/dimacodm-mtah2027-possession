"""Write the three SVG figures of the documentation: python scripts/make_figures.py docs/figures.

The drawings are schematic: the dots, the token matrix and the curves are illustrative, they are
not data of any match. The files are plain ASCII SVG and need no dependency.
"""

import random
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/figures")

INK = "#111111"
MUTED = "#555555"
RULE = "#444444"
FAINT = "#c9ccd1"
BLUE = "#1f4e9c"
ORANGE = "#d9651e"
GREY = "#8a8f98"
GOLD = "#e0a800"
PALE_BLUE = "#dbe5f4"
PALE_GREY = "#ececec"
SERIF = "font-family=\"'Times New Roman', Times, serif\""
PLUS_MINUS = "&#177;"
ELEMENT_OF = "&#8712;"
Y_HAT = "&#375;"
TIMES = "&#215;"

# Every label is drawn this much larger than its nominal size. Set by each figure.
scale = 1.0


def head(w: int, h: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" aria-label="{title}">',
        f"<title>{title}</title>",
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0,1 L10,5 L0,9 z" '
        f'fill="{INK}"/></marker>',
        "</defs>",
        f'<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>',
    ]


def text(x, y, content, size=13, fill=INK, anchor="middle", style="normal", weight="normal") -> str:
    return (
        f'<text {SERIF} x="{x}" y="{y}" font-size="{size * scale:.1f}" fill="{fill}" '
        f'text-anchor="{anchor}" font-style="{style}" font-weight="{weight}">{content}</text>'
    )


def sub(base: str, index: str, weight: str = "normal", style: str = "italic") -> str:
    """base with a subscript, for example y with t."""
    return (
        f'<tspan font-style="{style}" font-weight="{weight}">{base}</tspan>'
        f'<tspan dy="3" font-size="70%" font-style="italic">{index}</tspan><tspan dy="-3"></tspan>'
    )


def sup(value: str) -> str:
    return f'<tspan dy="-6" font-size="70%">{value}</tspan><tspan dy="6"></tspan>'


def label(x, y, main, *minor) -> list[str]:
    out = [text(x, y, main, 13)]
    for i, line_text in enumerate(minor):
        out.append(text(x, y + (i + 1) * 16.5 * scale, line_text, 11, MUTED))
    return out


def line(x1, y1, x2, y2, color=INK, width=1.1, dash="", arrow=False) -> str:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    extra += ' marker-end="url(#arrow)"' if arrow else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="{width}"{extra}/>'
    )


def rect(x, y, w, h, fill="#ffffff", stroke=RULE, width=1.0, dash="", rx=0) -> str:
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{width}"{extra}/>'
    )


def circle(x, y, r, fill, stroke="none", width=0.0) -> str:
    return (
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="{stroke}" '
        f'stroke-width="{width}"/>'
    )


def panel(x1, x2, title) -> list[str]:
    """Italic panel title with a bracket under it."""
    y = 34
    return [
        text((x1 + x2) / 2, 23, title, 13, style="italic"),
        f'<path d="M{x1},{y + 7} L{x1},{y} L{x2},{y} L{x2},{y + 7}" fill="none" '
        f'stroke="{RULE}" stroke-width="0.9"/>',
    ]


def pitch(x, y, w, h, players, ball=None, ball_lost=False, r=2.6) -> list[str]:
    """A small pitch. players: (u, v, team) with u, v in [0, 1]."""
    out = [
        rect(x, y, w, h, "#f6f6f4", GREY, 0.9),
        line(x + w / 2, y, x + w / 2, y + h, GREY, 0.9),
        f'<circle cx="{x + w / 2}" cy="{y + h / 2}" r="{h * 0.14:.1f}" fill="none" '
        f'stroke="{GREY}" stroke-width="0.9"/>',
        rect(x, y + h * 0.25, w * 0.13, h * 0.5, "none", GREY, 0.9),
        rect(x + w * 0.87, y + h * 0.25, w * 0.13, h * 0.5, "none", GREY, 0.9),
    ]
    for u, v, team in players:
        out.append(circle(x + u * w, y + v * h, r, BLUE if team == 0 else ORANGE))
    if ball is not None:
        bx, by = x + ball[0] * w, y + ball[1] * h
        if ball_lost:
            out.append(circle(bx, by, r + 1.6, "none", GOLD, 1.4))
        else:
            out.append(circle(bx, by, r, GOLD, INK, 0.9))
    return out


def save(name: str, parts: list[str]) -> None:
    body = "\n".join([*parts, "</svg>"]) + "\n"
    (OUT / name).write_text(body, encoding="ascii", newline="\n")
    print("wrote", name)


def formation(rng: random.Random) -> list[tuple[float, float, int]]:
    home = [(0.06, 0.5), (0.2, 0.2), (0.19, 0.42), (0.2, 0.62), (0.22, 0.83), (0.36, 0.3),
            (0.34, 0.55), (0.4, 0.75), (0.5, 0.2), (0.52, 0.5), (0.5, 0.8)]  # fmt: skip
    away = [(0.93, 0.5), (0.74, 0.18), (0.76, 0.4), (0.75, 0.6), (0.73, 0.82), (0.6, 0.32),
            (0.62, 0.52), (0.6, 0.72), (0.47, 0.36), (0.45, 0.64), (0.55, 0.9)]  # fmt: skip
    jitter = lambda value: min(0.97, max(0.03, value + rng.uniform(-0.03, 0.03)))  # noqa: E731
    return [(jitter(u), jitter(v), 0) for u, v in home] + [
        (jitter(u), jitter(v), 1) for u, v in away
    ]


def heat(value: float) -> str:
    """Diverging blue / white / red colour for a value in [-1, 1]."""
    low, high = (33, 76, 156), (178, 34, 44)
    target = high if value > 0 else low
    mix = min(1.0, abs(value))
    rgb = [round(255 + (channel - 255) * mix) for channel in target]
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


# ------------------------------------------------------------------ figure 1: the architecture
def architecture() -> None:
    global scale
    scale = 1.35
    rng = random.Random(7)
    w, h = 1012, 362
    p = head(w, h, "Architecture of the model, from the tracking to the label of each frame")
    p += panel(14, 446, "(a) Per-frame set encoder")
    p += panel(456, 682, "(b) Temporal model")
    p += panel(692, 998, "(c) Heads and structured decoding")
    top, bottom = 66, 266
    mid = (top + bottom) / 2
    ly = bottom + 34

    # (a) pitch, token matrix, set transformer, frame vector
    players = formation(rng)
    p += pitch(16, mid - 31, 96, 62, players, ball=(0.36, 0.33), r=2.8)
    p += label(64, ly, "Tracking", "46 slots + ball")
    p.append(line(116, mid, 135, mid, arrow=True))

    rows, cols, cw = 25, 13, 4.8
    ch = (bottom - top) / rows
    mx = 139
    empty = set(rng.sample(range(rows - 1), 6))
    for i in range(rows):
        for j in range(cols):
            value = 0.0 if i in empty else rng.uniform(-1, 1) * (0.9 if j < 9 else 0.5)
            p.append(
                f'<rect x="{mx + j * cw:.1f}" y="{top + i * ch:.1f}" width="{cw}" '
                f'height="{ch + 0.2:.1f}" fill="{heat(value)}"/>'
            )
    p.append(rect(mx, top, cols * cw, bottom - top, "none", RULE, 0.9))
    matrix_label = (
        f"{sub('X', 't', 'bold', 'normal')} {ELEMENT_OF} "
        f'<tspan font-style="italic">R</tspan>{sup(f"47{TIMES}13")}'
    )
    p += label(mx + cols * cw / 2, ly, matrix_label, "masked tokens")
    p.append(line(mx + cols * cw + 5, mid, 230, mid, arrow=True))

    bx, bw = 234, 112
    p.append(rect(bx, top - 4, bw, bottom - top + 8, "#ffffff", RULE, 0.9, dash="3 2"))
    blocks = [("LayerNorm", "#ffffff"), ("Masked MHSA", PALE_BLUE), ("LayerNorm", "#ffffff"),
              ("MLP", PALE_GREY)]  # fmt: skip
    for i, (name, fill) in enumerate(blocks):
        y = top + 7 + i * 49
        p.append(rect(bx + 8, y, bw - 16, 39, fill, RULE, 0.9))
        p.append(text(bx + bw / 2, y + 24.5, name, 10))
    p += label(bx + bw / 2, ly, "Set Transformer", f"{TIMES}2, 4 heads, d = 64")
    p.append(line(bx + bw + 5, mid, 392, mid, arrow=True))

    zx = 397
    for i in range(rows):
        p.append(
            f'<rect x="{zx}" y="{top + i * ch:.1f}" width="14" height="{ch + 0.2:.1f}" '
            f'fill="{heat(rng.uniform(-1, 1))}"/>'
        )
    p.append(rect(zx, top, 14, bottom - top, "none", RULE, 0.9))
    vector_label = (
        f"{sub('z', 't', 'bold', 'normal')} {ELEMENT_OF} "
        f'<tspan font-style="italic">R</tspan>{sup("128")}'
    )
    p += label(zx + 7, ly, vector_label, "team pooling")
    p.append(line(zx + 20, mid, 458, mid, arrow=True))

    # (b) dilated convolutions: the cone of one output frame, first four dilations
    n, x0, step = 31, 464, 5.6
    ys = [bottom, bottom - 50, bottom - 100, bottom - 150, bottom - 200]
    centre = n // 2
    cones = [set(), set(), set(), set(), {centre}]
    for level, d in ((4, 8), (3, 4), (2, 2), (1, 1)):
        for i in cones[level]:
            cones[level - 1].update({i - d, i, i + d})
    for level in range(5):
        for i in range(n):
            if i not in cones[level]:
                p.append(circle(x0 + i * step, ys[level], 1.0, FAINT))
    for level, d in ((4, 8), (3, 4), (2, 2), (1, 1)):
        for i in sorted(cones[level]):
            for j in (i - d, i, i + d):
                p.append(line(x0 + i * step, ys[level], x0 + j * step, ys[level - 1], BLUE, 1.0))
    for level in range(5):
        for i in sorted(cones[level]):
            p.append(circle(x0 + i * step, ys[level], 2.3, BLUE))
    for level, d in ((4, 8), (3, 4), (2, 2), (1, 1)):
        p.append(text(x0 + n * step, ys[level] + 5, f"d = {d}", 10.5, MUTED, "start", "italic"))
    p += label(x0 + (n - 1) * step / 2 + 14, ly, "Dilated residual TCN",
               "9 blocks, d = 1, ..., 256",
               f"{PLUS_MINUS}511 frames (about {PLUS_MINUS}20 s)")  # fmt: skip
    p.append(line(678, mid, 696, mid, arrow=True))

    # (c) heads, trellis, labels
    hx, hw = 700, 86
    p.append(text(hx + hw / 2, top + 3, f"p({sub('y', 't')})", 11.5, style="italic"))
    by, bh = top + 12, 60
    p.append(rect(hx, by, hw, bh, BLUE, "none", 0))
    p.append(f'<polygon points="{hx},{by} {hx + 21},{by} {hx + 16},{by + bh} {hx},{by + bh}" '
             f'fill="{GREY}"/>')  # fmt: skip
    p.append(f'<polygon points="{hx + 52},{by} {hx + 67},{by} {hx + 71},{by + bh} '
             f'{hx + 58},{by + bh}" fill="{ORANGE}"/>')  # fmt: skip
    p.append(text(hx + hw / 2, bottom - 56, "p(team change)", 11.5, style="italic"))
    cy = bottom
    p.append(
        f'<path d="M{hx},{cy} L{hx + 10},{cy} C{hx + 15},{cy} {hx + 15},{cy - 32} {hx + 18},{cy - 32} '
        f"C{hx + 21},{cy - 32} {hx + 21},{cy} {hx + 27},{cy} L{hx + 46},{cy} "
        f"C{hx + 51},{cy} {hx + 51},{cy - 44} {hx + 54},{cy - 44} "
        f"C{hx + 57},{cy - 44} {hx + 57},{cy - 6} {hx + 61},{cy - 6} "
        f"C{hx + 64},{cy - 6} {hx + 64},{cy - 37} {hx + 67},{cy - 37} "
        f'C{hx + 70},{cy - 37} {hx + 70},{cy} {hx + 76},{cy} L{hx + hw},{cy} Z" '
        f'fill="{PALE_BLUE}" stroke="{BLUE}" stroke-width="1.1"/>'
    )
    p += label(hx + hw / 2, ly, "Output heads", "+ 2 auxiliary heads")
    p.append(line(hx + hw + 4, mid, hx + hw + 22, mid, arrow=True))

    tx, tstep = 852, 15
    state_y = [top + 10, top + 70, top + 130, top + 190]
    state_color = [BLUE, ORANGE, GREY, GREY]
    allowed = [(0, 0), (1, 1), (2, 2), (3, 3), (0, 1), (1, 0), (0, 2), (1, 3), (2, 1), (2, 0),
               (3, 0), (3, 1)]  # fmt: skip
    for name, y in zip(("H", "A", "D|H", "D|A"), state_y, strict=True):
        p.append(text(tx - 8, y + 4.5, name, 10.5, MUTED, "end"))
    for col in range(5):
        for a, b in allowed:
            p.append(line(tx + col * tstep, state_y[a], tx + (col + 1) * tstep, state_y[b],
                          FAINT, 0.8))  # fmt: skip
    path = [0, 0, 2, 1, 1, 0]
    for col in range(5):
        p.append(line(tx + col * tstep, state_y[path[col]], tx + (col + 1) * tstep,
                      state_y[path[col + 1]], INK, 2.0))  # fmt: skip
    for col in range(6):
        for s in range(4):
            on_path = path[col] == s
            p.append(circle(tx + col * tstep, state_y[s], 3.0 if on_path else 1.9,
                            state_color[s] if on_path else FAINT))  # fmt: skip
    p += label(tx + 2.5 * tstep - 14, ly, "4-state Viterbi", "time-varying costs")
    p.append(line(tx + 5 * tstep + 7, mid, tx + 5 * tstep + 26, mid, arrow=True))

    sx = 960
    y = top
    for state, share in zip(path, (0.2, 0.2, 0.16, 0.14, 0.14, 0.16), strict=True):
        height = share * (bottom - top)
        p.append(f'<rect x="{sx}" y="{y:.1f}" width="18" height="{height + 0.3:.1f}" '
                 f'fill="{state_color[state]}"/>')  # fmt: skip
        y += height
    p += label(sx + 9, ly, sub(Y_HAT, "t"), "H / A / D")
    save("pipeline.svg", p)


# ------------------------------------------------------------------ figure 2: the decoder
def decoder() -> None:
    global scale
    scale = 1.2
    w, h = 830, 284
    p = head(w, h, "The four states of the decoder and the cost of each move")
    p += panel(18, 440, "(a) States and allowed moves")
    p += panel(470, 812, "(b) Cost of each move")
    xl, xr, yt, yb, r = 120, 350, 96, 220, 24
    # change of team
    p.append(line(xl - 6, yt + r + 2, xl - 6, yb - r - 3, arrow=True))
    p.append(line(xl + 6, yb - r - 2, xl + 6, yt + r + 3, arrow=True))
    p.append(text(xl - 14, (yt + yb) / 2 + 4, "change of team", 11, MUTED, "end", "italic"))
    # stoppage (solid) and restart by the same team (dashed)
    for y, sign in ((yt, -1), (yb, 1)):
        p.append(line(xl + r + 2, y + sign * 8, xr - r - 3, y + sign * 8, arrow=True))
        p.append(line(xr - r - 2, y - sign * 8, xl + r + 3, y - sign * 8, dash="4 3", arrow=True))
    p.append(text((xl + xr) / 2, yt - 15, "stoppage", 11, MUTED, style="italic"))
    p.append(text((xl + xr) / 2, yb + 25, "stoppage", 11, MUTED, style="italic"))
    # restart by the other team
    p.append(line(xr - r + 2, yt + r - 6, xl + r + 1, yb - r + 4, arrow=True))
    p.append(line(xr - r + 2, yb - r + 6, xl + r + 1, yt + r - 4, arrow=True))
    p.append(rect((xl + xr) / 2 + 30, (yt + yb) / 2 - 10, 70, 18, "#ffffff", "none", 0))
    p.append(text((xl + xr) / 2 + 65, (yt + yb) / 2 + 3, "restart", 11, MUTED, style="italic"))
    for x, y, name, color in ((xl, yt, "H", BLUE), (xl, yb, "A", ORANGE), (xr, yt, "D|H", GREY),
                              (xr, yb, "D|A", GREY)):  # fmt: skip
        p.append(circle(x, y, r, color))
        p.append(text(x, y + 4.5, name, 13, "#ffffff", weight="bold"))
    p.append(text(xr, yt - r - 10, "dead ball, home had it", 11, MUTED))
    p.append(text(xr, yb + r + 19, "dead ball, away had it", 11, MUTED))

    rows = [("change of team", "5", ""), ("stoppage", "5", ""),
            ("restart by the other team", "30", ""),
            ("restart by the same team", "31", "4 3")]  # fmt: skip
    ty, right = 76, 804
    p.append(line(478, ty - 20, right, ty - 20, RULE, 1.0))
    for i, (name, cost, dash) in enumerate(rows):
        y = ty + i * 28
        p.append(line(482, y - 4, 512, y - 4, INK, 1.1, dash=dash, arrow=True))
        p.append(text(522, y, name, 12.5, anchor="start"))
        p.append(text(right - 4, y, cost, 12.5, anchor="end"))
    p.append(line(478, ty + 3 * 28 + 12, right, ty + 3 * 28 + 12, RULE, 1.0))
    notes = ["Staying in a state is free.", "A restart is cheaper while the players accelerate.",
             "A change of team is cheaper where the model detects one.",
             "Costs are searched on held-out games only."]  # fmt: skip
    for i, note in enumerate(notes):
        p.append(text(478, ty + 3 * 28 + 36 + i * 19, note, 11, MUTED, "start"))
    save("decoder.svg", p)


# ------------------------------------------------------------------ figure 3: training data
def training() -> None:
    global scale
    scale = 1.35
    rng = random.Random(11)
    w, h = 1030, 380
    p = head(w, h, "Where the training data comes from and what the provided match is used for")
    p += panel(14, 216, "(a) Public data")
    p += panel(232, 624, "(b) Degradation, redrawn for every window")
    p += panel(640, 1016, "(c) Training and selection")

    sources = [("Metrica Sports", "3 games, event labels"),
               ("IDSSE / DFL", "7 games, per-frame flags"),
               ("SkillCorner", "20 games, training only")]  # fmt: skip
    for i, (name, detail) in enumerate(sources):
        y = 56 + i * 68
        p.append(rect(20, y, 190, 58, "#ffffff", RULE, 0.9))
        p.append(text(115, y + 24, name, 13))
        p.append(text(115, y + 46, detail, 11, MUTED))

    pw, ph, py = 170, 112, 72
    x1, x2 = 240, 446
    mid = py + ph / 2
    p.append(line(214, mid, x1 - 5, mid, arrow=True))
    players = formation(rng)
    p += pitch(x1, py, pw, ph, players, ball=(0.36, 0.33), r=3.6)
    kept = [pl for i, pl in enumerate(players) if i not in (2, 7, 9, 13, 16, 20)]
    noisy = [(u + rng.uniform(-0.015, 0.015), v + rng.uniform(-0.02, 0.02), t) for u, v, t in kept]
    p += pitch(x2, py, pw, ph, noisy, ball=(0.36, 0.33), ball_lost=True, r=3.6)
    p.append(line(x2 + 0.36 * pw, py + 0.33 * ph - 7, x2 + 0.36 * pw, py - 3, GOLD, 1.1))
    p.append(text(x2 + 0.36 * pw, py - 8, "ball lost", 10.5, "#8a6800"))
    p.append(line(x1 + pw + 6, mid, x2 - 6, mid, arrow=True))
    p += label(x1 + pw / 2, py + ph + 28, "Clean tracking", "every player, ball seen")
    p += label(x2 + pw / 2, py + ph + 28, "Degraded tracking", "anonymous slots, gaps, lost ball")
    p.append(line(x2 + pw + 6, mid, 644, mid, arrow=True))

    gx, gy, cw, ch = 650, 62, 22, 15
    for row in range(6):
        for col in range(5):
            held = row < 5 and col == row
            p.append(rect(gx + col * cw, gy + row * (ch + 5), cw - 3, ch,
                          ORANGE if held else BLUE, "none", 0))  # fmt: skip
        p.append(text(gx + 5 * cw + 3, gy + row * (ch + 5) + 12,
                      f"fold {row}" if row < 5 else "all games", 10, MUTED, "start"))  # fmt: skip
    p += label(gx + 82, py + ph + 28, "6 models", "held-out games in orange")
    bx, bw = 842, 174
    for y, name, detail in ((56, "Weights", "pinned by sha256"),
                            (124, "Decoder settings", "from held-out predictions")):  # fmt: skip
        p.append(rect(bx, y, bw, 58, "#ffffff", RULE, 0.9))
        p.append(text(bx + bw / 2, y + 24, name, 13))
        p.append(text(bx + bw / 2, y + 46, detail, 10.5, MUTED))
    p.append(line(820, 108, bx - 4, 92, arrow=True))
    p.append(line(820, 128, bx - 4, 150, arrow=True))

    my = 296
    p.append(rect(20, my, 604, 68, "#ffffff", ORANGE, 1.2, dash="5 3"))
    p.append(text(322, my + 28, "The provided match (FC Barcelona vs Athletic Club) is never used "
                  "to train or to tune", 12.5))  # fmt: skip
    p.append(text(322, my + 51, "its tracking is only described, without any label, to set the "
                  "degradation ranges", 11, MUTED))  # fmt: skip
    ax = x2 + pw / 2
    p.append(line(ax, my - 2, ax, py + ph + 62, RULE, 1.0, dash="3 3", arrow=True))
    p.append(text(ax + 8, my - 12, "label-free description", 10.5, MUTED, "start", "italic"))
    p.append(rect(712, my, 304, 68, "#ffffff", RULE, 0.9))
    p.append(text(864, my + 28, "Prediction on the provided match", 12.5))
    p.append(text(864, my + 51, "public 5 minutes: checks only", 11, MUTED))
    p.append(line(628, my + 34, 708, my + 34, arrow=True))
    p.append(line(bx + bw / 2, 186, bx + bw / 2, my - 4, arrow=True))
    save("training.svg", p)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    architecture()
    decoder()
    training()
