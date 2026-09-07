"""Check the profile's batch interpolation against the installed OmegaConf runtime."""

import argparse
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument("--profile", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
config = OmegaConf.load(args.profile)
assert config.data.train_batch_size == config.data.gen_batch_size == 1
checks = []
for size in (1, 2, 4):
    config.data.train_batch_size = size
    assert config.data.gen_batch_size == size
    checks.append({"train_batch_size": size, "gen_batch_size": config.data.gen_batch_size})
recipe = Path(
    "/home/chen/vendor/dapo-official-20260906/verl-recipe-mask-v1/dapo/dapo_ray_trainer.py"
)
source = recipe.read_text(encoding="utf-8")
assert "batch = batch[:traj_bsz]" in source
report = {
    "passed": True,
    "interpolation_checks": checks,
    "profile_sha256": hashlib.sha256(args.profile.read_bytes()).hexdigest(),
    "recipe_sha256": hashlib.sha256(recipe.read_bytes()).hexdigest(),
    "scope": "Config interpolation and installed surplus-slicing source only; "
    "not a new DAPO execution or learning test. Run-02 already used gen_batch_size=1.",
}
args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report))
