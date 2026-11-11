# Training

The weights in `models/ali_deep/` were produced by the code in this repository, on public data
about other games only. Nothing from the provided match is needed to train, and nothing from it
was ever sent to the training machines.

Training is not needed to regenerate the submission: the published weights are in the repository
and the prediction from them is byte-reproducible. This page documents how they were made.

## Data

Downloaded by the code on first use, each file checked against a pinned hash
(`src/mtah/solutions/ali/bench.py`). About 3 GB.

| source | games | licence | role |
|---|---|---|---|
| Metrica Sports sample data, commit `e706dd5` (github.com/metrica-sports/sample-data) | 3 | "please acknowledge the source" | training and benchmark |
| IDSSE / DFL, Bassek et al., Scientific Data 2025 (doi 10.6084/m9.figshare.28196177) | 7 | CC-BY 4.0 | training and benchmark |
| SkillCorner open data, commit `4340d27` (github.com/SkillCorner/opendata) | 20 | MIT | training only |

## Recipe

`src/mtah/solutions/ali/train_gpu.sh` runs it end to end on a machine with one GPU:

```bash
git clone https://github.com/aliamrrr/dimacodm-mtah2027-possession.git
cd dimacodm-mtah2027-possession
FOLDS="0" bash src/mtah/solutions/ali/train_gpu.sh      # one fold; folds 0 to 5 make the ensemble
```

- Folds 0 to 4 each hold two of the ten benchmark games out, train on the other eight and on the
  SkillCorner games, and write out-of-fold predictions for the two held-out games. Fold 5 trains
  on all ten games and holds nothing out.
- 20 000 steps of 16 windows. A window is 1 792 frames; only its 768 central frames are
  supervised, so every supervised frame has its full 20 s of context on both sides.
- AdamW, learning rate 5e-4 with warm-up then cosine decay, weight decay 0.01, gradient clipping
  at 1, mixed precision.
- Loss: state cross-entropy weighted by (possession duration) ^ -0.5, plus 0.3 x possessor
  pointer, 0.5 x ball position, and the team-change head.
- Every window is degraded at random (`deep_data.Randomization`), flipped along either axis of the
  pitch, and has its two teams exchanged half of the time.

One fold takes 30 to 50 minutes on a single consumer GPU (RTX 3090, 4090 or similar) with 8 to
12 data workers. The folds were trained on rented GPU machines, in parallel. Each data worker
holds every training game in memory, about 2.6 GB each.

## After training

```bash
uv run python -m mtah.solutions.ali.bench --profile real-like-moving      # benchmark bundles
uv run python -m mtah.solutions.ali.deep_tune --convention dfl --objective balanced \
    --profile real-like-moving \
    --start models/ali_deep/decoder_search_start.json                     # decoder, out-of-fold
uv run python -m mtah.solutions.ali.deep_weights write                    # pin the weights
```

`deep_tune` searches the decoder parameters by coordinate ascent on the out-of-fold predictions
that training writes under `models/ali_deep/oof_conv/` (not committed: they are large and can be
regenerated). Each game is decoded exactly as the solution decodes a match and scored with the
official scorer against its own labels. The search starts from the decoder of an earlier model
(`decoder_search_start.json`). Its result, `models/ali_deep/decoder_dfl_balanced_moving.json`, is
what `configs/ali_deep.yaml` holds.

## What is and is not reproducible

- Prediction from the published weights: bit for bit, on any machine (numpy, float64, decoder
  inputs rounded to 1e-4).
- Training: the recipe and the seeds are fixed, but GPU arithmetic is not deterministic to the
  last bit. Running it again gives weights that are close, not identical. The submission is
  therefore defined by the published weights, which are pinned by sha256 in
  `src/mtah/solutions/ali/deep_weights.json`.
