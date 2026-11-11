"""The weights of dimacodm: files under models/dimacodm/, pinned by sha256 in deep_weights.json.

    python -m mtah.solutions.dimacodm.deep_weights write   # after training: pin models/dimacodm

The six fold models (about 2.7 MB each) are committed with the code. deep_weights.json records the
sha256 of each one, and DimacodmSolution only loads files that match it, so a prediction always
comes from the published weights (rule 3 of the challenge: the repository regenerates the file).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from mtah.paths import REPO_ROOT

app = typer.Typer(add_completion=False)
MANIFEST = Path(__file__).with_name("deep_weights.json")
MODELS = REPO_ROOT / "models" / "dimacodm"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest() -> dict[str, Any] | None:
    return json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.is_file() else None


def verified(folder: Path) -> list[Path]:
    """The manifest's model files, each checked against its sha256 (sorted). Another folder than
    models/dimacodm (tests, experiments), or no manifest yet: every fold*.npz there, unchecked."""
    pinned = manifest()
    if pinned is None or folder.resolve() != MODELS.resolve():
        return sorted(folder.glob("fold*.npz"))
    paths = []
    for name, entry in sorted(pinned["files"].items()):
        path = folder / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing: it is part of the repository")
        if sha256(path) != entry["sha256"]:
            raise ValueError(
                f"{path}: sha256 differs from deep_weights.json (not the published file)"
            )
        paths.append(path)
    return paths


@app.command()
def write(folder: Annotated[Path, typer.Option()] = MODELS) -> None:
    """Pin every fold*.npz of `folder` (after training)."""
    files = {
        p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
        for p in sorted(folder.glob("fold*.npz"))
    }
    if not files:
        raise typer.BadParameter(f"no fold*.npz under {folder}")
    text = json.dumps({"files": files}, indent=2) + "\n"
    MANIFEST.write_bytes(text.encode("utf-8"))
    typer.echo(f"pinned {len(files)} files in {MANIFEST.name}")


if __name__ == "__main__":  # pragma: no cover
    app()
