"""H/A/D timeline of the ground truth and any number of submissions over a frame range."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mtah.evaluation import load_gt, resolve_gt
from mtah.labels import A, D, H, X
from mtah.paths import OUTPUTS_DIR, SYNTHETIC_DIR
from mtah.submission import read_submission

COLORS = {H: "#1f4e9c", A: "#c8102e", D: "#9a9a9a", X: "#ffffff"}


def plot_timeline(
    csvs: list[Path],
    start: int,
    end: int,
    gt: str | None,
    data_root: Path | None = None,
    out: Path | None = None,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as error:  # pragma: no cover
        raise SystemExit("matplotlib missing: uv sync --extra viz") from error

    rows: list[tuple[str, np.ndarray]] = []
    if gt:
        root = data_root or (SYNTHETIC_DIR if gt.startswith("synthetic") else None)
        rows.append((f"truth ({gt})", load_gt(resolve_gt(gt, root))))
    for path in csvs:
        name = path.parent.parent.name if path.name == "submission.csv" else path.stem
        rows.append((name, read_submission(path)))

    fig, ax = plt.subplots(figsize=(16, 0.6 * len(rows) + 1.2))
    for y, (_, codes) in enumerate(rows):
        window = codes[start : end + 1]
        breaks = np.flatnonzero(window[1:] != window[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(window)]))
        for i in range(len(bounds) - 1):
            a, b = bounds[i], bounds[i + 1]
            ax.broken_barh([(start + a, b - a)], (y - 0.4, 0.8), color=COLORS[int(window[a])])
    ax.set_yticks(range(len(rows)), [name for name, _ in rows])
    ax.invert_yaxis()
    ticks = np.linspace(start, end, 9).astype(int)
    ax.set_xticks(ticks, [f"{t}\n{t // 1500:02d}:{(t // 25) % 60:02d}" for t in ticks])
    ax.set_xlim(start, end)
    ax.legend(
        handles=[Patch(color=COLORS[k], label=v) for k, v in ((H, "H"), (A, "A"), (D, "D"))],
        loc="upper right",
        ncol=3,
        bbox_to_anchor=(1, 1.35),
    )
    ax.set_title(f"frames {start}-{end} (frame / mm:ss of the video)", loc="left")
    fig.tight_layout()
    target = out or OUTPUTS_DIR / f"timeline_{start}_{end}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=110)
    plt.close(fig)
    return target
