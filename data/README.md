# data/: the official bundle (never committed)

Unpack the bundle received after enrolment here, keeping the original names:

```
data/
  tactical/
    BAR-ATH_tactical.mp4
    BAR-ATH_tactical_atd.xml
    BAR-ATH_tactical_atd.txt
    BAR-ATH_tactical_homography.csv
    BAR-ATH_tactical_smart_tagging.xml
  broadcast/
    BAR-ATH_broadcast.mp4
    BAR-ATH_broadcast_atd.xml
    BAR-ATH_broadcast_atd.txt
    BAR-ATH_broadcast_homography.csv
```

Somewhere else on the disk? Set `MTAH_DATA_DIR=/path/to/bundle` (environment variable, or a
`.env` file copied from `.env.example`).

Then, in this order:

```bash
uv run mtah data check          # files present, alignment, figures published in DATA.md
uv run mtah data check-loader   # our loader rebuilds the four official baselines byte for byte
```

The submitted solution reads the tactical tracking only (`BAR-ATH_tactical_atd.xml` and
`BAR-ATH_tactical_atd.txt`). Everything in this folder is ignored by git: the data is licensed for
the competition only.
