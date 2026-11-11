"""dummy == the organisers' starter/baseline.py, byte for byte, on any file in the official format.

On the real data, `mtah data check-loader` closes the loop against the published baseline CSVs.
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

from mtah.paths import VENDOR_DIR
from mtah.solutions import get
from mtah.submission import write_submission


@pytest.fixture(scope="module")
def baseline():
    sys.path.insert(0, str(VENDOR_DIR))
    spec = importlib.util.spec_from_file_location(
        "official_baseline", VENDOR_DIR / "starter" / "baseline.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("window", [0, 25, 50, 125])
def test_dummy_reproduces_official_baseline(baseline, mini_match, tmp_path, window):
    meta = mini_match.asset_path("atd_meta")
    raw = mini_match.asset_path("atd_raw")
    codes = baseline.nearest_player_labels(
        str(raw),
        baseline.team_of_each_track(str(meta)),
        mini_match.spec.n_frames,
        mini_match.spec.pitch,
    )
    theirs = baseline.write_submission(str(tmp_path / "theirs.csv"), baseline.smooth(codes, window))
    ours = write_submission(tmp_path / "ours.csv", get("dummy")(window=window).predict(mini_match))

    def digest(data):
        return hashlib.sha256(data).hexdigest()

    # The official writer opens its file in text mode, so on Windows it writes CRLF. Ours always
    # writes LF, the bytes of the published baseline CSVs: only the official side is normalised.
    official_bytes = Path(theirs).read_bytes().replace(b"\r\n", b"\n")
    assert digest(official_bytes) == digest(Path(ours).read_bytes())
