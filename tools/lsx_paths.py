#!/usr/bin/env python3
"""Where a model lives, resolved from the registry instead of guessed.

The store is ORG-NESTED and split in two:

    source      <LSX_SOURCE_ROOT>/<org>/<model>            the download
    artifact    <LSX_ARTIFACT_ROOT>/<org>/<model>/<profile> the converted tree

`models/<org>/<model>.json` is the only thing that knows the org, so every
Python tool that needs a model path asks THIS file, not `os.path.join(root,
"Some-Model")` — a join without the org names a directory that never exists.

The profile directory IS the artifact: there is no `logestix/` segment any more
(that was the pre-split layout, where conversion wrote inside the checkpoint).
`artifact_dir()` still accepts an old tree so a reference run captured against
one can be reproduced, but it never produces one.
"""
import glob
import json
import os
import pathlib

DEFAULT_SOURCE_ROOT = "/mnt/fast_storage/lsx_model"


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent


def source_root() -> pathlib.Path:
    """LSX_SOURCE_ROOT, then the pre-split LSX_MODEL_ROOT, then the default."""
    for name in ("LSX_SOURCE_ROOT", "LSX_MODEL_ROOT"):
        value = os.environ.get(name)
        if value:
            return pathlib.Path(value)
    return pathlib.Path(DEFAULT_SOURCE_ROOT)


def artifact_root():
    """LSX_ARTIFACT_ROOT, or None: the root has no default by design."""
    value = os.environ.get("LSX_ARTIFACT_ROOT")
    return pathlib.Path(value) if value else None


def rows():
    for path in sorted(glob.glob(str(repo_root() / "models" / "*" / "*.json"))):
        with open(path) as f:
            yield json.load(f)


def row_for(model_or_path):
    """The registry row for a model_type or for a store directory name."""
    if not str(model_or_path):
        return None
    name = pathlib.Path(str(model_or_path)).name
    for row in rows():
        if row["model_type"] == model_or_path or row["vault"]["dir"] == name:
            return row
    return None


def source_dir(model_or_path) -> pathlib.Path:
    """`<source root>/<org>/<model>` for a model_type or store directory name."""
    row = row_for(model_or_path)
    if row:
        return source_root() / row["vault"]["org"] / row["vault"]["dir"]
    return pathlib.Path(str(model_or_path))


def artifact_dir(model_or_path, profile=None) -> pathlib.Path:
    """The converted tree for a model_type, a store name or an existing path.

    An argument that already IS an artifact (or an old `logestix/` tree) is
    returned as it is: a caller comparing against a tree on another disk must
    keep working. An empty argument is refused rather than resolving to `.` —
    a relative cwd is never a model, and returning it would turn a caller's
    typo into a plausible-looking path.
    """
    text = str(model_or_path)
    if not text:
        raise ValueError("empty model path: refuse to resolve '.' as a model")
    given = pathlib.Path(text)
    if (given / "artifact.fb").exists():
        return given
    if (given / "logestix" / "artifact.fb").exists():
        return given / "logestix"
    row = row_for(text)
    root = artifact_root()
    if row and root:
        chosen = profile or row.get("preferred_profile") or "nvfp4"
        return root / row["vault"]["org"] / row["vault"]["dir"] / chosen
    return given
