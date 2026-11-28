"""Unit tests for tools/lsx_paths.py — the one resolver Python tools use.

Host-only, no CUDA, no repo weights: every case builds a synthetic store in a
temp directory and pins the env vars so the real store is never consulted.

What is under test is the shape of a path, not the presence of a model:

  * the store is ORG-NESTED, and the org comes from `models/<org>/<model>.json`,
    so a resolver that joins a bare name onto a root is wrong even when the
    directory happens to exist;
  * an artifact is `<root>/<org>/<model>/<profile>` — the profile directory IS
    the artifact, with no `logestix/` segment;
  * an argument that already is a tree is returned as it is, so a reference run
    against a copy on another disk keeps working;
  * an empty argument is refused instead of resolving to `.`.

Run: python3 tools/test_lsx_paths.py  (or via ctest test_lsx_paths)
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import lsx_paths  # noqa: E402


class StoreCase(unittest.TestCase):
    """A temp source store, a temp artifact store, and two registry rows."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.source = base / "src"
        self.artifact = base / "art"
        (self.source / "Qwen" / "Qwen3.5-9B").mkdir(parents=True)
        (self.source / "microsoft" / "phi-4").mkdir(parents=True)
        self.art_phi = self.artifact / "microsoft" / "phi-4" / "nvfp4"
        self.art_phi.mkdir(parents=True)
        (self.art_phi / "artifact.fb").write_bytes(b"x")

        # Two rows, read from a synthetic registry so the case does not depend on
        # what is committed: the resolver only ever needs vault.org / vault.dir.
        self.rows = [
            {"model_type": "qwen3.5-9b",
             "vault": {"org": "Qwen", "dir": "Qwen3.5-9B"},
             "preferred_profile": "nvfp4"},
            {"model_type": "phi-4",
             "vault": {"org": "microsoft", "dir": "phi-4"},
             "preferred_profile": "nvfp4"},
        ]
        self.env = mock.patch.dict(os.environ, {
            "LSX_SOURCE_ROOT": str(self.source),
            "LSX_ARTIFACT_ROOT": str(self.artifact),
        })
        self.env.start()
        self.rows_patch = mock.patch.object(lsx_paths, "rows",
                                            lambda: iter(self.rows))
        self.rows_patch.start()

    def tearDown(self):
        self.rows_patch.stop()
        self.env.stop()
        self.tmp.cleanup()

    # ── no default, in either language ────────────────────────────────────
    def test_an_unset_source_root_is_not_invented(self):
        # The C++ side has no compiled-in checkpoint root; neither may this
        # helper. An invented root points at `<root>/<org>/<model>` for a root
        # that holds nothing, so every model reads as "not downloaded".
        with mock.patch.dict(os.environ, {}, clear=False):
            for name in ("LSX_SOURCE_ROOT", "LSX_MODEL_ROOT"):
                os.environ.pop(name, None)
            self.assertIsNone(lsx_paths.source_root())
            with self.assertRaises(lsx_paths.NotConfigured) as raised:
                lsx_paths.source_dir("qwen3.5-9b")
            self.assertIn("LSX_SOURCE_ROOT", str(raised.exception))
            self.assertIn("no default", str(raised.exception))
            # The artifact side answers None rather than raising: a caller can
            # report it without an exception, and ArtifactRoot has no default
            # either.
            os.environ.pop("LSX_ARTIFACT_ROOT", None)
            self.assertIsNone(lsx_paths.artifact_root())

    # ── source ────────────────────────────────────────────────────────────
    def test_source_dir_includes_the_org(self):
        self.assertEqual(lsx_paths.source_dir("qwen3.5-9b"),
                         self.source / "Qwen" / "Qwen3.5-9B")

    def test_source_dir_accepts_the_store_spelling(self):
        self.assertEqual(lsx_paths.source_dir("Qwen3.5-9B"),
                         self.source / "Qwen" / "Qwen3.5-9B")

    def test_source_root_env_wins(self):
        self.assertEqual(lsx_paths.source_root(), self.source)

    # ── artifact ──────────────────────────────────────────────────────────
    def test_artifact_dir_is_org_model_profile(self):
        self.assertEqual(lsx_paths.artifact_dir("qwen3.5-9b"),
                         self.artifact / "Qwen" / "Qwen3.5-9B" / "nvfp4")

    def test_artifact_dir_has_no_logestix_segment(self):
        got = lsx_paths.artifact_dir("phi-4")
        self.assertNotIn("logestix", got.parts)

    def test_artifact_dir_honours_an_explicit_profile(self):
        self.assertEqual(lsx_paths.artifact_dir("phi-4", "bf16"),
                         self.artifact / "microsoft" / "phi-4" / "bf16")

    def test_an_existing_tree_is_returned_as_it_is(self):
        # A reference run against a copy elsewhere must keep working, and the
        # tree that already carries artifact.fb is that copy.
        self.assertEqual(lsx_paths.artifact_dir(str(self.art_phi)), self.art_phi)

    def test_without_an_artifact_root_nothing_is_invented(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            del os.environ["LSX_ARTIFACT_ROOT"]
            # No root: the registry cannot place it, so the argument is echoed
            # back rather than pointed at a guessed store.
            self.assertEqual(lsx_paths.artifact_dir("qwen3.5-9b"),
                             Path("qwen3.5-9b"))

    # ── refusals ──────────────────────────────────────────────────────────
    def test_an_empty_path_is_refused(self):
        # '.' is a relative cwd, never a model: answering it would turn a typo
        # into a plausible path and the caller's existence check into a lie.
        with self.assertRaises(ValueError):
            lsx_paths.artifact_dir("")

    def test_an_unknown_name_is_not_given_an_org(self):
        got = lsx_paths.artifact_dir("not-a-model")
        self.assertEqual(got, Path("not-a-model"))
        self.assertIsNone(lsx_paths.row_for("not-a-model"))

    # ── the registry itself ───────────────────────────────────────────────
    def test_the_committed_registry_is_org_nested(self):
        # The real rows, not the synthetic ones: file path and vault block must
        # agree, which is the rule the generator enforces on its side.
        self.rows_patch.stop()
        try:
            seen = 0
            for row in lsx_paths.rows():
                path = (ROOT / "models" / row["vault"]["org"] /
                        (row["vault"]["dir"] + ".json"))
                self.assertTrue(path.is_file(),
                                "%s: registry row not at %s"
                                % (row["model_type"], path))
                seen += 1
            self.assertGreater(seen, 40)   # an empty scan would pass silently
        finally:
            self.rows_patch.start()


if __name__ == "__main__":
    unittest.main(verbosity=2)
