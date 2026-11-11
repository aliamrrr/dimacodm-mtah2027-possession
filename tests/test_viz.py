import numpy as np
import pytest

from mtah.submission import write_submission


def test_timeline_png(mini_bundle, tmp_path):
    pytest.importorskip("matplotlib")
    from mtah.viz.timeline import plot_timeline

    csv = write_submission(tmp_path / "s.csv", np.zeros(36_000, dtype=np.int8))
    png = plot_timeline([csv], 1500, 4000, "synthetic", mini_bundle, out=tmp_path / "t.png")
    assert png.stat().st_size > 1000


def test_clip(mini_bundle, tmp_path):
    cv2 = pytest.importorskip("cv2")
    from mtah.viz.clip import make_clip

    video = mini_bundle / "tactical" / "BAR-ATH_tactical.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), 25, (320, 180))
    for _ in range(1800):
        writer.write(np.zeros((180, 320, 3), dtype=np.uint8))
    writer.release()
    try:
        csv = write_submission(tmp_path / "s.csv", np.zeros(36_000, dtype=np.int8))
        clip = make_clip(
            1600, [csv], 4.0, "synthetic", mini_bundle, out=tmp_path / "c.mp4", scale=1.0
        )
        assert clip.stat().st_size > 0
    finally:
        video.unlink()
