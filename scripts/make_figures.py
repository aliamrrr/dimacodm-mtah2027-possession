"""Write the three SVG figures of the documentation: python scripts/make_figures.py docs/figures."""

import sys
from pathlib import Path

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1f2933"
MUTED = "#52606d"
LINE = "#52606d"
GREY = ("#f5f7fa", "#9aa5b1")
BLUE = ("#e1ecfb", "#3b6fb6")
GREEN = ("#e3f5e8", "#2f855a")
AMBER = ("#fdf1d8", "#b7791f")
RED = ("#fde8e8", "#c53030")
FONT = 'font-family="Arial, Helvetica, sans-serif"'


def head(w: int, h: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" aria-label="{title}">',
        f"<title>{title}</title>",
        "<defs>",
        '<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{LINE}"/>'
        "</marker>",
        "</defs>",
        f'<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff"/>',
    ]


def text(x, y, lines, size=13, weight="normal", fill=INK, anchor="middle", lh=None) -> str:
    lh = lh or round(size * 1.3)
    if isinstance(lines, str):
        lines = [lines]
    y0 = y - (len(lines) - 1) * lh / 2
    spans = "".join(
        f'<tspan x="{x}" y="{y0 + i * lh:.1f}">{line}</tspan>' for i, line in enumerate(lines)
    )
    return (
        f'<text {FONT} font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'text-anchor="{anchor}" dominant-baseline="middle">{spans}</text>'
    )


def box(x, y, w, h, title, lines=(), color=GREY, dashed=False) -> list[str]:
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{color[0]}" '
        f'stroke="{color[1]}" stroke-width="1.5"{dash}/>'
    ]
    cx = x + w / 2
    if lines:
        n = len(lines)
        block = 18 + 6 + n * 15
        top = y + (h - block) / 2
        out.append(text(cx, top + 9, title, 13.5, "bold"))
        out.append(text(cx, top + 18 + 6 + n * 15 / 2, list(lines), 11.5, fill=MUTED, lh=15))
    else:
        out.append(text(cx, y + h / 2, title, 13.5, "bold"))
    return out


def arrow(x1, y1, x2, y2, both=False, dashed=False) -> str:
    start = ' marker-start="url(#arrow)"' if both else ""
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{LINE}" stroke-width="1.6"'
        f'{dash} marker-end="url(#arrow)"{start}/>'
    )


def path(d) -> str:
    return (
        f'<path d="{d}" fill="none" stroke="{LINE}" stroke-width="1.6" marker-end="url(#arrow)"/>'
    )


def save(name: str, parts: list[str]) -> None:
    body = "\n".join([*parts, "</svg>"]) + "\n"
    body.encode("ascii")
    (OUT / name).write_text(body, encoding="ascii", newline="\n")
    print("wrote", name)


# ---------------------------------------------------------------- figure 1: inference pipeline
W, H = 980, 330
p = head(W, H, "From the tracking of the match to the possession file")
bw, bh, gap = 205, 96, 40
xs = [20 + i * (bw + gap) for i in range(4)]
y1, y2 = 44, 204
p.append(text(20, 22, "For every frame", 12, "bold", MUTED, "start"))
p += box(
    xs[0],
    y1,
    bw,
    bh,
    "Tracking of the match",
    ["players and ball positions", "25 frames per second", "ball often missing"],
    GREY,
)
p += box(
    xs[1],
    y1,
    bw,
    bh,
    "Frame description",
    ["one token per tracked player", "one token for the ball", "30 global features"],
    GREY,
)
p += box(
    xs[2],
    y1,
    bw,
    bh,
    "Set transformer",
    ["2 attention layers", "looks at all players at once", "any number, any order"],
    BLUE,
)
p += box(
    xs[3],
    y1,
    bw,
    bh,
    "Temporal convolution",
    ["9 dilated blocks", "20 s of context", "before and after the frame"],
    BLUE,
)
for i in range(3):
    p.append(arrow(xs[i] + bw + 3, y1 + bh / 2, xs[i + 1] - 4, y1 + bh / 2))
# down and back to the left for the second row
p.append(path(f"M{xs[3] + bw / 2},{y1 + bh + 3} L{xs[3] + bw / 2},{y2 - 5}"))
p.append(text(20, y2 - 22, "For the whole match", 12, "bold", MUTED, "start"))
p += box(
    xs[3],
    y2,
    bw,
    bh,
    "Ensemble",
    ["6 models, each run twice", "(teams as is, teams swapped)", "scores are averaged"],
    BLUE,
)
p += box(
    xs[2],
    y2,
    bw,
    bh,
    "Viterbi decoder",
    ["4 states, one half at a time", "turns noisy scores into", "clean possessions"],
    AMBER,
)
p += box(
    xs[1],
    y2,
    bw,
    bh,
    "DIMACodm.csv",
    ["one label per frame", "H, A, D or X", "199,319 rows"],
    GREEN,
)
p.append(arrow(xs[3] - 3, y2 + bh / 2, xs[2] + bw + 4, y2 + bh / 2))
p.append(arrow(xs[2] - 3, y2 + bh / 2, xs[1] + bw + 4, y2 + bh / 2))
p.append(
    text(
        xs[0] + bw / 2,
        y2 + bh / 2,
        ["numpy only, no GPU", "about 15 minutes", "same bytes on any machine"],
        12,
        fill=MUTED,
        lh=17,
    )
)
save("pipeline.svg", p)

# ---------------------------------------------------------------- figure 2: the decoder
W, H = 940, 400
p = head(W, H, "The four states of the decoder and the cost of each move")
sw, sh = 190, 70
xl, xr, yt, yb = 170, 580, 50, 280
p += box(xl, yt, sw, sh, "H", ["home team has the ball"], BLUE)
p += box(xr, yt, sw, sh, "A", ["away team has the ball"], BLUE)
p += box(xl, yb, sw, sh, "D after H", ["dead ball, home had it"], GREY)
p += box(xr, yb, sw, sh, "D after A", ["dead ball, away had it"], GREY)
# change of team
p.append(arrow(xl + sw + 4, yt + sh / 2, xr - 5, yt + sh / 2, both=True))
p.append(text((xl + sw + xr) / 2, yt + sh / 2 - 14, "change of team: 5", 12.5, "bold"))
p.append(
    text(
        (xl + sw + xr) / 2, yt + sh / 2 + 15, "cheaper where a change is detected", 11.5, fill=MUTED
    )
)
# stoppage and same-team restart, left and right
for x0, side in ((xl, -1), (xr, 1)):
    xa = x0 + sw / 2 - 22
    xb = x0 + sw / 2 + 22
    p.append(arrow(xa, yt + sh + 4, xa, yb - 5))
    p.append(arrow(xb, yb - 4, xb, yt + sh + 5))
    tx = xa - 12 if side < 0 else xb + 12
    anchor = "end" if side < 0 else "start"
    ym = (yt + sh + yb) / 2
    p.append(text(tx, ym - 22, "down, stoppage: 5", 12.5, "bold", anchor=anchor))
    p.append(
        text(tx, ym + 14, ["up, restart by the", "same team: 31"], 12.5, "bold", anchor=anchor)
    )
# restart by the other team (diagonals)
p.append(arrow(xl + sw + 4, yb + 8, xr - 5, yt + sh - 2))
p.append(arrow(xr - 4, yb + 8, xl + sw + 5, yt + sh - 2))
p.append(
    f'<rect x="{(xl + sw + xr) / 2 - 92}" y="{(yt + sh + yb) / 2 - 2}" width="184" '
    f'height="40" fill="#ffffff"/>'
)
p.append(text((xl + sw + xr) / 2, (yt + sh + yb) / 2 + 10, "restart, other team: 30", 12.5, "bold"))
p.append(
    text(
        (xl + sw + xr) / 2,
        (yt + sh + yb) / 2 + 28,
        "cheaper while players accelerate",
        11.5,
        fill=MUTED,
    )
)
p.append(
    text(
        W / 2,
        H - 16,
        "Staying in a state is free. A move is made only when the evidence outweighs its cost.",
        12,
        fill=MUTED,
    )
)
save("decoder.svg", p)

# ---------------------------------------------------------------- figure 3: training
W, H = 940, 400
p = head(W, H, "Where the training data comes from and what the provided match is used for")
sx, sw, sh = 20, 210, 62
ys = [40, 116, 192]
p.append(text(sx, 20, "Public data about other games", 12, "bold", MUTED, "start"))
p += box(sx, ys[0], sw, sh, "Metrica Sports", ["3 games, labels from events"], GREY)
p += box(sx, ys[1], sw, sh, "IDSSE / DFL", ["7 games, labels on every frame"], GREY)
p += box(sx, ys[2], sw, sh, "SkillCorner", ["20 games, training only"], GREY)
dx, dw = 280, 200
p += box(
    dx,
    86,
    dw,
    122,
    "Degrade the tracking",
    [
        "anonymous track slots",
        "gaps and identity swaps",
        "ball lost for up to 30 s",
        "ball moving while dead",
    ],
    AMBER,
)
for y in ys:
    p.append(arrow(sx + sw + 3, y + sh / 2, dx - 5, 147 + (y - ys[1]) * 0.35))
tx, tw = 530, 190
p += box(
    tx,
    86,
    tw,
    122,
    "Train 6 models",
    [
        "5 folds by game",
        "+ 1 model on all games",
        "each benchmark game is",
        "scored by a model that",
        "never saw it",
    ],
    BLUE,
)
p.append(arrow(dx + dw + 3, 147, tx - 5, 147))
ox, ow = 770, 150
p += box(ox, 56, ow, 70, "Weights", ["models/dimacodm", "pinned by sha256"], GREEN)
p += box(ox, 168, ow, 70, "Decoder settings", ["searched on held-out", "predictions"], GREEN)
p.append(arrow(tx + tw + 3, 130, ox - 5, 96))
p.append(arrow(tx + tw + 3, 164, ox - 5, 200))
# the provided match
my = 300
p.append(text(sx, my - 18, "The provided match", 12, "bold", MUTED, "start"))
p += box(
    sx,
    my,
    480,
    76,
    "FC Barcelona vs Athletic Club",
    [
        "never used to train a model or to choose a setting",
        "its tracking was only described, without labels, to set the degradation",
    ],
    RED,
    dashed=True,
)
p += box(
    ox - 190,
    my,
    ow + 190,
    76,
    "Prediction and checks",
    ["the trained system is applied to the match", "public 5 minutes: checks only"],
    GREY,
)
p.append(arrow(sx + 480 + 3, my + 38, ox - 190 - 5, my + 38))
p.append(path(f"M{ox + ow / 2},{238 + 3} L{ox + ow / 2},{my - 5}"))
p.append(
    path(f"M{sx + 380},{my - 4} L{sx + 380},{250} L{dx + dw / 2},{250} L{dx + dw / 2},{208 + 5}")
)
p.append(text(sx + 388, 268, "label-free description", 11.5, fill=MUTED, anchor="start"))
save("training.svg", p)
