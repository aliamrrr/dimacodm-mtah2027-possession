"""Per-frame homography: normalised pitch coordinates (0-1) <-> image pixels.

CSV columns: frame, h00..h22 (row-major 3x3), reprojection_error, valid. Where ``valid`` is 0 the
matrix is NaN. Lower reprojection_error is better. Tactical: valid on 100 % of frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

MATRIX_COLUMNS = [f"h{r}{c}" for r in range(3) for c in range(3)]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Homography:
    matrices: FloatArray  # (n, 3, 3), NaN where invalid
    reprojection_error: FloatArray  # (n,)
    valid: npt.NDArray[np.bool_]  # (n,)

    @property
    def n_frames(self) -> int:
        return int(self.matrices.shape[0])

    def inverse(self) -> FloatArray:
        """(n, 3, 3) image -> pitch matrices (NaN where invalid)."""
        out = np.full_like(self.matrices, np.nan)
        out[self.valid] = np.linalg.inv(self.matrices[self.valid])
        return out

    def camera_target(self, image_size: tuple[int, int] = (1920, 1080)) -> FloatArray:
        """(n, 2) pitch point (normalised) shown at the centre of the image on each frame."""
        u, v = image_size[0] / 2.0, image_size[1] / 2.0
        inv = self.inverse()
        point = inv @ np.array([u, v, 1.0])
        return point[:, :2] / point[:, 2:3]


def load_homography(path: Path, n_frames: int | None = None) -> Homography:
    table = pd.read_csv(path)
    frames = table["frame"].to_numpy()
    n = int(n_frames if n_frames is not None else frames.max() + 1)
    matrices = np.full((n, 3, 3), np.nan)
    error = np.full(n, np.nan)
    valid = np.zeros(n, dtype=bool)
    keep = (frames >= 0) & (frames < n)
    matrices[frames[keep]] = (
        table[MATRIX_COLUMNS].to_numpy(dtype=np.float64)[keep].reshape(-1, 3, 3)
    )
    error[frames[keep]] = table["reprojection_error"].to_numpy(dtype=np.float64)[keep]
    valid[frames[keep]] = table["valid"].to_numpy()[keep].astype(bool)
    valid &= ~np.isnan(matrices).any(axis=(1, 2))
    return Homography(matrices, error, valid)


def project(
    matrix: FloatArray, x: npt.ArrayLike, y: npt.ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Pitch (normalised) -> pixels for one 3x3 matrix. Works on scalars or arrays."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    points = np.stack([x, y, np.ones_like(x)], axis=-1) @ np.asarray(matrix).T
    return points[..., 0] / points[..., 2], points[..., 1] / points[..., 2]


def unproject(
    matrix: FloatArray, u: npt.ArrayLike, v: npt.ArrayLike
) -> tuple[FloatArray, FloatArray]:
    """Pixels -> pitch (normalised) for one 3x3 matrix (ground-plane assumption)."""
    return project(np.linalg.inv(np.asarray(matrix)), u, v)
