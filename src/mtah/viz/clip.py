"""Short annotated video around a frame: projected tracking + every label row, for error analysis.

Typical use: take a `missed`/`spurious` transition from the official report
(outputs/official_board/reports/<name>_transitions.csv) and watch what fooled the model.
Never commit or share the result: it is derived from the licensed video.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mtah.data.match import MatchData
from mtah.evaluation import load_gt, resolve_gt
from mtah.labels import LABELS
from mtah.paths import OUTPUTS_DIR, SYNTHETIC_DIR
from mtah.submission import read_submission

BGR = {"H": (156, 78, 31), "A": (46, 16, 200), "D": (154, 154, 154), "X": (255, 255, 255)}


def make_clip(
    frame: int,
    csvs: list[Path],
    seconds: float,
    gt: str | None,
    data_root: Path | None = None,
    out: Path | None = None,
    scale: float = 0.5,
) -> Path:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover
        raise SystemExit("opencv missing: uv sync --extra viz") from error

    match = MatchData(data_root)
    video = match.video_path()
    if not video.is_file():
        raise FileNotFoundError(f"{video} not found")
    rows: list[tuple[str, np.ndarray]] = []
    if gt:
        root = match.root if gt.startswith("synthetic") else (data_root or SYNTHETIC_DIR)
        rows.append(("GT", load_gt(resolve_gt(gt, root))))
    rows += [
        (p.parent.parent.name if p.name == "submission.csv" else p.stem, read_submission(p))
        for p in csvs
    ]

    fps = match.spec.fps
    half = int(seconds * fps / 2)
    first, last = max(0, frame - half), min(match.spec.n_frames - 1, frame + half)
    tr, hom = match.tracking, match.homography
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) * scale)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * scale)
    target = out or OUTPUTS_DIR / f"clip_{frame}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))
    for f in range(first, last + 1):
        ok, image = capture.read()
        if not ok:
            break
        image = cv2.resize(image, (width, height))
        if hom.valid[f]:
            m = hom.matrices[f]
            for slot in range(tr.n_slots):
                x, y = tr.players[f, slot]
                if np.isnan(x):
                    continue
                p = m @ np.array([x, y, 1.0])
                color = BGR["H"] if tr.slot_team[slot] == 0 else BGR["A"]
                cv2.circle(image, (int(p[0] / p[2] * scale), int(p[1] / p[2] * scale)), 6, color, 2)
            bx, by = tr.ball[f]
            if not np.isnan(bx):
                p = m @ np.array([bx, by, 1.0])
                cv2.circle(
                    image,
                    (int(p[0] / p[2] * scale), int(p[1] / p[2] * scale)),
                    5,
                    (0, 255, 255),
                    -1,
                )
        for i, (name, codes) in enumerate(rows):
            label = LABELS[int(codes[f])]
            cv2.rectangle(image, (8, 8 + 26 * i), (200, 30 + 26 * i), BGR[label], -1)
            cv2.putText(
                image,
                f"{name}: {label}",
                (14, 26 + 26 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            image,
            f"frame {f}",
            (width - 170, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(image)
    writer.release()
    capture.release()
    return target
