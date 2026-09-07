"""Patch only progress/checkpoint control in isolated copies of pinned upstream.

This does not provide a trainer or a replacement loss. The generated patches
modify the actual upstream classes used by the accompanying control-flow probe.
"""

import argparse
import ast
import difflib
import hashlib
import json
import shutil
from pathlib import Path

VERL_FILE = Path("verl/trainer/ppo/ray_trainer.py")
RECIPE_FILE = Path("dapo/dapo_ray_trainer.py")
VERL_SHA = "c930623489f724e9abbf1dab41ed617da1be2f0d60485d4c0ee5354c145dd809"
RECIPE_SHA = "ce4a091c7987088694b9fddfd57c2988439f2ad2df580e5debf113b538e91cc0"

PROGRESS_METHODS = """    def _save_checkpoint(self):
        # This hook runs only after an actual update, before the base class
        # publishes the latest-checkpoint tracker. No pending filtered batch is
        # checkpointed as an optimizer update.
        folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        os.makedirs(folder, exist_ok=True)
        progress = {
            "version": 1,
            "completed_updates": self.global_steps,
            "generated_batches": self.gen_steps,
            "data_epoch": self.data_epoch,
            "batches_per_epoch": len(self.train_dataloader),
        }
        path = os.path.join(folder, "dapo_progress.json")
        with open(path + ".tmp", "w") as stream:
            json.dump(progress, stream)
        os.replace(path + ".tmp", path)
        super()._save_checkpoint()

    def _load_checkpoint(self):
        folder = super()._load_checkpoint()
        if not folder:
            self.gen_steps = 0
            self.data_epoch = 0
            return
        data_path = os.path.join(folder, "data.pt")
        if not os.path.exists(data_path):
            raise ValueError("DAPO continuation requires the saved data.pt sampler state")
        data_state = torch.load(data_path, map_location="cpu", weights_only=False)
        n = len(self.train_dataloader)
        # This state schema is the installed StatefulDataLoader's num_workers=0
        # path. Reject unsupported schemas rather than infer a cursor from U.
        k = data_state.get("_num_yielded")
        finished = data_state.get("_iterator_finished")
        if n <= 0 or type(k) is not int or not 0 <= k <= n or type(finished) is not bool:
            raise ValueError("Unsupported DAPO dataloader checkpoint cursor schema")
        if finished and k != n:
            raise ValueError("An exhausted DAPO iterator must have consumed its epoch")
        path = os.path.join(folder, "dapo_progress.json")
        if os.path.exists(path):
            with open(path) as stream:
                progress = json.load(stream)
            required = ("version", "completed_updates", "generated_batches",
                        "data_epoch", "batches_per_epoch")
            if any(type(progress.get(key)) is not int for key in required):
                raise ValueError("Invalid DAPO checkpoint progress schema")
            if (progress["version"] != 1 or progress["completed_updates"] != self.global_steps
                    or progress["batches_per_epoch"] != n or progress["data_epoch"] < 0
                    or progress["generated_batches"] < self.global_steps
                    or progress["generated_batches"] != progress["data_epoch"] * n + k):
                raise ValueError("DAPO progress disagrees with the checkpoint step or data cursor")
            self.gen_steps = progress["generated_batches"]
            self.data_epoch = progress["data_epoch"] + int(finished)
        else:
            # Legacy checkpoints lack the epoch count. Accept only a provably
            # first-epoch position: each completed update consumed at most M
            # batches, U*M < N, and the saved cursor lies in [U, U*M].
            filters = self.config.algorithm.filter_groups
            maximum = filters.max_num_gen_batches if filters.enable else 1
            if (maximum <= 0 or self.global_steps * maximum >= n or finished
                    or not self.global_steps <= k <= self.global_steps * maximum):
                raise ValueError("Legacy DAPO checkpoint has ambiguous generation/epoch progress; "
                                 "recover it from verified generation records before resuming")
            self.gen_steps = k
            self.data_epoch = 0

"""


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree(root):
    return {
        str(p.relative_to(root)): sha(p)
        for p in root.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
    }


def once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def patch_verl(before):
    start = before.index(
        "            steps_per_epoch = len(self.train_dataloader)",
        before.index("    def _load_checkpoint("),
    )
    end = before.index(
        '\n        else:\n            print(f"Warning: No dataloader state found', start
    )
    after = (
        before[:start]
        + """            # Restore the sampler RNG at real and apparent epoch boundaries.
            # DAPO optimizer updates do not count filtered generation batches.
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)"""
        + before[end:]
    )
    return once(
        after,
        '            print(f"Warning: No dataloader state found at {dataloader_local_path}, '
        'will start from scratch")\n',
        '            print(f"Warning: No dataloader state found at {dataloader_local_path}, '
        'will start from scratch")\n'
        "        return global_step_folder\n",
    )


def patch_recipe(before):
    after = once(before, "import os\n", "import json\nimport os\n")
    after = once(
        after,
        "from copy import deepcopy\n",
        "from copy import deepcopy\nfrom itertools import count\n",
    )
    after = once(after, "    def fit(self):\n", PROGRESS_METHODS + "    def fit(self):\n")
    after = once(
        after,
        "        self.gen_steps = 0\n        self.max_steps_duration",
        "        self.gen_steps = 0\n        self.data_epoch = 0\n        self.max_steps_duration",
    )
    after = once(
        after,
        "        self._load_checkpoint()\n        self.checkpoint_manager.update_weights()",
        """        self._load_checkpoint()
        if self.global_steps >= self.total_training_steps:
            print(f"Target already reached: {self.global_steps} completed updates")
            return
        if len(self.train_dataloader) <= 0:
            raise ValueError("DAPO requires a nonempty training dataloader")
        self.checkpoint_manager.update_weights()""",
    )
    after = once(
        after,
        "        self.global_steps += 1\n        self.gen_steps += 1\n",
        "        self.global_steps += 1\n",
    )
    after = once(
        after,
        """        current_epoch = self.global_steps // len(self.train_dataloader)

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
""",
        """        # An explicit update target takes precedence over an epoch budget,
        # exactly as it does when constructing the optimizer's step schedule.
        epochs = (count(self.data_epoch) if self.config.trainer.total_training_steps is not None
                  else range(self.data_epoch, self.config.trainer.total_epochs))
        for epoch in epochs:
            self.data_epoch = epoch
            for batch_dict in self.train_dataloader:
                self.gen_steps += 1
""",
    )
    after = once(after, "                                self.gen_steps += 1\n", "")
    after = once(
        after,
        '                metrics["train/num_gen_batches"] = num_gen_batches\n',
        '                metrics["train/num_gen_batches"] = num_gen_batches\n'
        '                metrics["training/generated_batches_total"] = self.gen_steps\n'
        '                metrics["training/data_epoch"] = self.data_epoch\n',
    )
    start = after.index(
        "                self.gen_steps += 1\n        # check if last step checkpint exists"
    )
    after = (
        after[:start]
        + """        # The counter points to the next update here. Exhausting an implicit
        # epoch budget is not successful completion and must not publish that
        # unexecuted step (or discard pending filtered batches in a checkpoint).
        self.global_steps -= 1
        if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
            self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
        progress_bar.close()
        raise RuntimeError(f"DAPO epoch budget exhausted after {self.global_steps} actual updates; "
                           f"target={self.total_training_steps}, "
                           f"generated_batches={self.gen_steps}. "
                           "Set an explicit total_training_steps to train to an update target.")
"""
    )
    # Protect the entire numerical update block, including mask handling,
    # rollout correction, advantage, actor RPC, and weight synchronization.
    for begin, end in [
        ("    def compute_kl_related_metrics", "    def _save_checkpoint"),
        (
            "                    self.checkpoint_manager.sleep_replicas()",
            "                # validate",
        ),
    ]:
        old_end = "    def fit" if end == "    def _save_checkpoint" else end
        assert (
            before[before.index(begin) : before.index(old_end)]
            == after[after.index(begin) : after.index(end)]
        )
    ast.parse(after)
    return after


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verl-source", type=Path, required=True)
    parser.add_argument("--recipe-source", type=Path, required=True)
    parser.add_argument("--verl-candidate", type=Path, required=True)
    parser.add_argument("--recipe-candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {
        "scope": "Isolated upstream progress-only candidate; "
        "no GPU acceptance or production admission yet."
    }
    for name, source, target, relative, expected, transform in [
        ("verl", args.verl_source, args.verl_candidate, VERL_FILE, VERL_SHA, patch_verl),
        (
            "recipe",
            args.recipe_source,
            args.recipe_candidate,
            RECIPE_FILE,
            RECIPE_SHA,
            patch_recipe,
        ),
    ]:
        assert sha(source / relative) == expected
        original = tree(source)
        before = (source / relative).read_text()
        after = transform(before)
        ast.parse(after)
        shutil.copytree(
            source, target, ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc")
        )
        (target / relative).write_text(after)
        copied = tree(target)
        assert original.keys() == copied.keys()
        changed = [n for n in original if original[n] != copied[n]]
        assert changed == [str(relative)] and tree(source) == original
        patch = "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="a/" + str(relative),
                tofile="b/" + str(relative),
            )
        )
        patch_path = args.output / f"{name}-progress.patch"
        patch_path.write_text(patch)
        report[name] = {
            "source": str(source),
            "candidate": str(target),
            "changed_files": changed,
            "files_checked": len(original),
            "before_sha256": expected,
            "after_sha256": sha(target / relative),
            "patch_sha256": sha(patch_path),
            "source_unchanged": True,
        }
    (args.output / "candidate-source.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
