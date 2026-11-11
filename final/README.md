# final/: the submission file

`uv run mtah final ali_deep` writes two files here:

- `DIMACodm.csv`, the possession file to upload (ignored by git);
- `SUBMISSION.json`, its sha256, the commit and configuration that produced it, and the outcome of
  the determinism check (two runs, tracking cache on then off, same bytes). It is committed.

To check a regenerated file, compare its sha256 with the one in `SUBMISSION.json`.
