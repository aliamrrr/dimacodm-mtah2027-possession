"""The vendored scorer is the organisers' file, byte for byte, and behaves as documented."""

import subprocess
import sys

import numpy as np

from mtah import official
from mtah.paths import PUBLIC_GT, REPO_ROOT


def test_vendored_files_match_manifest():
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "sync_official.py"), "--check"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_public_gt_scores_perfectly_against_itself():
    pm = official.metrics()
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    result = pm.evaluate(gt, gt.copy())
    assert result["spell_f1_mean"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["n_gt_spells"] == 17  # documented in DATA.md


def test_public_gt_shape_matches_documentation():
    pm = official.metrics()
    gt = pm.label_codes(pm.read_label_csv(PUBLIC_GT))
    assert len(gt) == 199_319
    graded = np.flatnonzero(gt != pm.X)
    assert graded[0] == 11_185 and graded[-1] == 18_684


def test_reference_outputs_recorded():
    refs = official.manifest()["reference_outputs"]
    assert set(refs) == {
        "baseline_raw",
        "baseline_smooth25",
        "baseline_smooth50",
        "baseline_smooth125",
    }


def test_official_spec_halves():
    from mtah.spec import MatchSpec

    spec = MatchSpec.official()
    assert spec.n_frames == 199_319
    assert spec.halves == ((1, 11_185, 90_651), (2, 110_808, 190_197))
    assert spec.n_graded == 158_857
