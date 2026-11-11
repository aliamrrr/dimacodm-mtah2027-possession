# models/: trained weights

`ali_deep/` holds the six fold models of the submitted solution, as numpy archives
(`fold0.npz` ... `fold5.npz`, about 3 MB each), their training summaries (`fold*.json`: recipe,
held-out games, score curve) and the decoder search (`decoder_dfl_balanced_moving.json`, its
result, and `decoder_search_start.json`, where it started from).

- The weights were trained on public data about OTHER games only (docs/TRAINING.md). Nothing in
  them comes from BAR-ATH.
- Each file is pinned by sha256 in `src/mtah/solutions/ali/deep_weights.json`. The solution
  refuses to run with a file that does not match, so a prediction always comes from these weights.
- Training on a GPU is not bit-reproducible: running the recipe again gives close weights, not the
  same bytes. The prediction from the published weights is bit-reproducible (numpy, float64).
