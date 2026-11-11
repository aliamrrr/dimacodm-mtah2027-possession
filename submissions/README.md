# submissions/: runs (never committed)

Every `uv run mtah run <solution>` writes

```
submissions/<solution>/<date-time-commit>/submission.csv
submissions/<solution>/<date-time-commit>/manifest.json   git commit, config, data fingerprint, sha256
submissions/<solution>/latest/                            copy of the most recent run
```

`uv run mtah reproduce submissions/<solution>/latest/manifest.json` replays a run from its
manifest and compares the bytes.
