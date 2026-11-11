#!/usr/bin/env bash
# Train `dimacodm` on a GPU machine, from a clone of this repository (docs/TRAINING.md).
#
#   FOLDS="0" bash src/mtah/solutions/dimacodm/train_gpu.sh     # one fold per machine, six machines
#   bash src/mtah/solutions/dimacodm/train_gpu.sh               # or the six folds one after the other
#
# The recipe of the published weights: 20 000 steps, spell-balanced loss, boundary head, the
# SkillCorner games as extra training data, +/- 20 s of context, and the held-out games scored on
# the real-like-moving benchmark.
#
# Only public data about OTHER games is used: the machine downloads it itself. Nothing from the
# BAR-ATH bundle is needed here, and nothing from it must be uploaded.
set -euo pipefail
cd "$(dirname "$0")/../../../.."

STEPS="${STEPS:-20000}"
BATCH="${BATCH:-16}"
CENTRE="${CENTRE:-768}"
LR="${LR:-5e-4}"
WORKERS="${WORKERS:-10}"  # each data worker holds every training game: about 2.6 GB of RAM each
CONTEXT="${CONTEXT:-1}"
FOLDS="${FOLDS:-0 1 2 3 4 5}"  # 5: every game, one more member of the ensemble
OUT="${OUT:-models/dimacodm}"
PROFILE="${PROFILE:-real-like-moving}"  # benchmark the held-out games are scored on
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export UV_PYTHON_PREFERENCE=only-managed
mkdir -p logs

command -v uv >/dev/null || pip install --quiet uv==0.11.32
uv sync --locked --group train
uv run python -c "import torch; assert torch.cuda.is_available(), 'no GPU'; torch.zeros(1).cuda(); print(torch.cuda.get_device_name(0))"

# Public data of other games -> clean training caches (downloads about 3 GB), then the benchmark
# bundles of the games these folds hold out (the only ones scored here).
uv run python -c "from mtah.solutions.dimacodm.bench import extra_games
from mtah.solutions.dimacodm.deep_data import load_clean
from mtah.solutions.dimacodm.deep_train import ALL_GAMES
for name in ALL_GAMES + extra_games(): load_clean(name)" 2>&1 | tee logs/clean.log
HELD=$(uv run python -c "from mtah.solutions.dimacodm.deep_train import FOLDS
print(' '.join('-g ' + g for k in map(int, '$FOLDS'.split()) if k < len(FOLDS) for g in FOLDS[k]))")
if [ -n "$HELD" ]; then
  # shellcheck disable=SC2086
  uv run python -m mtah.solutions.dimacodm.bench --profile "$PROFILE" $HELD 2>&1 | tee logs/bench.log
fi

for k in $FOLDS; do
  uv run python -m mtah.solutions.dimacodm.deep_train --device cuda --fold "$k" --steps "$STEPS" \
    --batch "$BATCH" --centre "$CENTRE" --lr "$LR" --workers "$WORKERS" --extra --boundary \
    --spell-alpha 0.5 --context "$CONTEXT" --profile "$PROFILE" --out "$OUT" 2>&1 | tee "logs/fold$k.log"
done

tar -czf ali_deep_models.tar.gz "$OUT"/*.npz "$OUT"/*.json "$OUT"/oof "$OUT"/oof_conv logs 2>/dev/null \
  || tar -czf ali_deep_models.tar.gz "$OUT"/*.npz "$OUT"/*.json logs
echo "done: ali_deep_models.tar.gz"
touch DONE
