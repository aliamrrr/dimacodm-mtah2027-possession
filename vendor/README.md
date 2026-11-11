# vendor/official/: exact copy of the organisers' repository

Source: https://github.com/metrica-sports/more-than-a-hack-2027 (MIT licence, see
`official/LICENSE`). The commit copied and the sha256 of every file are in
`official/MANIFEST.json`.

These files are never modified. The scorer that decides the ranking is this one; our code calls it
as it is (`mtah.official`). `python scripts/sync_official.py --check` and
`tests/test_official_integrity.py` fail if a single byte changes.

| file | why it is here |
|---|---|
| `evaluation/possession_metrics.py` | spellF1, macroF1, validation: the official definition |
| `evaluation/run_evaluation.py` | the official board (`mtah official-board`) |
| `evaluation/validate_submission.py` | the exact validation of the scorer (`mtah validate`) |
| `evaluation/gt/*_public*` | the public ground truth (5 minutes), for pipeline checks only |
| `games.py` | constants of the match, halves, asset paths |
| `starter/baseline.py` | the baseline: our tests prove that `dummy` rebuilds it byte for byte |
| `docs/*.md` | the rules and the documentation, as they were when this copy was made |
