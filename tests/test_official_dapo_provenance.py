"""Reject silent upstream substitutions, including misuse of patch exceptions."""

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.verify_official_dapo import verify_repositories


def sha(data):
    return hashlib.sha256(data).hexdigest()


class OfficialSourceProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project = Path(self.tmp.name)
        self.root = self.project / "upstream"
        self.tool = "verl/tools/schemas.py"
        self.core = "verl/trainer/ppo/core_algos.py"
        for relative, content in [(self.tool, b"patched"), (self.core, b"official")]:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        patch = self.project / "runtime.patch"
        patch.write_text(f"--- a/{self.tool}\n+++ b/{self.tool}\n")
        self.manifest = {
            "repositories": {
                "verl": {
                    "commit": "pinned",
                    "files": {
                        self.tool: sha(b"original"),
                        self.core: sha(b"official"),
                    },
                }
            },
            "reviewed_runtime_patches": [
                {
                    "repository": "verl",
                    "base_commit": "pinned",
                    "path": "runtime.patch",
                    "sha256": sha(patch.read_bytes()),
                    "files": {
                        self.tool: {
                            "upstream_sha256": sha(b"original"),
                            "patched_sha256": sha(b"patched"),
                        }
                    },
                }
            ],
        }

    def verify(self, manifest=None):
        return verify_repositories(manifest or self.manifest, {"verl": self.root}, self.project)

    def test_reviewed_runtime_patch_preserves_baseline_and_passes(self):
        before = copy.deepcopy(self.manifest)
        repos, patches = self.verify()
        self.assertEqual(repos["verl"]["reviewed_runtime_patch_files"], 1)
        self.assertEqual(len(patches), 1)
        self.assertEqual(self.manifest, before)

    def test_unpatched_runtime_is_rejected(self):
        (self.root / self.tool).write_bytes(b"original")
        with self.assertRaisesRegex(RuntimeError, "differs"):
            self.verify()

    def test_changed_algorithm_is_rejected(self):
        (self.root / self.core).write_bytes(b"substitute")
        with self.assertRaisesRegex(RuntimeError, "differs"):
            self.verify()

    def test_algorithm_cannot_use_runtime_patch_exception(self):
        patch = self.project / "runtime.patch"
        patch.write_text(f"--- a/{self.core}\n+++ b/{self.core}\n")
        entry = self.manifest["reviewed_runtime_patches"][0]
        entry["sha256"] = sha(patch.read_bytes())
        entry["files"] = {
            self.core: {
                "upstream_sha256": sha(b"official"),
                "patched_sha256": sha(b"substitute"),
            }
        }
        with self.assertRaisesRegex(RuntimeError, "algorithm sources"):
            self.verify()

    def test_changed_patch_is_rejected(self):
        (self.project / "runtime.patch").write_bytes(b"different patch")
        with self.assertRaisesRegex(RuntimeError, "patch content changed"):
            self.verify()

    def test_recipe_exception_cannot_admit_an_arbitrary_trainer_patch(self):
        relative = "dapo/dapo_ray_trainer.py"
        source = self.root / relative
        source.parent.mkdir(parents=True)
        source.write_bytes(b"replacement trainer")
        patch = self.project / "recipe.patch"
        patch.write_text(f"--- a/{relative}\n+++ b/{relative}\n")
        manifest = {
            "repositories": {
                "verl-recipe": {"commit": "pinned", "files": {relative: sha(b"official")}}
            },
            "reviewed_runtime_patches": [
                {
                    "repository": "verl-recipe",
                    "base_commit": "pinned",
                    "path": "recipe.patch",
                    "sha256": sha(patch.read_bytes()),
                    "files": {
                        relative: {
                            "upstream_sha256": sha(b"official"),
                            "patched_sha256": sha(b"replacement trainer"),
                        }
                    },
                }
            ],
        }
        with self.assertRaisesRegex(RuntimeError, "reviewed action-mask compatibility"):
            verify_repositories(manifest, {"verl-recipe": self.root}, self.project)

    def test_replaced_baseline_is_rejected(self):
        self.manifest["reviewed_runtime_patches"][0]["files"][self.tool]["upstream_sha256"] = "new"
        with self.assertRaisesRegex(RuntimeError, "original upstream fingerprint"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
