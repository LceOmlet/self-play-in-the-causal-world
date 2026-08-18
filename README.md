# Self-Play in the Causal World

This repository contains a small reference environment for active causal
reasoning with hidden binary CPT worlds. A model receives opaque variable names,
may request batched hard interventions, and must finally return both directed
causal effects as numbers.

The current milestone is deliberately narrow:

- three exact candidate difficulty seeds;
- one exact observational/hard-do probability-law owner;
- an action-keyed outcome tape for reproducible policy comparisons;
- a truth-free renderer and strict two-effect JSON decoder;
- three frozen continuous terminal diagnostics.

It does **not** yet define a self-play algorithm, a scalar training reward, or a
final benchmark sampling mixture. Historical provider-specific pilot scripts
are intentionally outside the core package.

## Quick start

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
python -m ruff check .
python -m ruff format --check .
```

Render a truth-free task:

```python
from cpt_world import VisibleTask, factorial_layouts, render_initial_messages

task = VisibleTask(factorial_layouts()[0])
messages = render_initial_messages(task)
```

Construct the paired candidate cases:

```python
from cpt_world import build_candidate_episodes, factorial_layouts

# A pilot may use an explicitly preregistered subset of the full 72-way
# surface factorial. No hidden subsampling occurs inside the builder.
episodes = build_candidate_episodes(layouts=factorial_layouts()[:2])
```

## Frozen terminal diagnostics

For `N` episodes, every terminal answer contains two effects. The public schema
`cpt-world-terminal-diagnostics-v1` fixes:

1. `vector_rmse`: RMSE over all `2N` numeric components;
2. `active_mae`: MAE on the generator-certified active direction;
3. `inactive_mae`: MAE on the generator-certified zero-effect direction.

All three retain the full numeric error. There is no direction threshold,
clipping, confidence field, or discrete correctness conversion. The active
coordinate comes from the sealed seed certificate, never from the prediction.
See [the metric contract](docs/metric-contract.md) for exact formulas.

These values diagnose terminal estimation. They do not by themselves measure
intervention efficiency, experimental-design quality, or calibration, and they
are not used as a training reward.

## Design boundaries

The package uses concrete functions and immutable data records rather than a
plugin system or inheritance framework. Ownership is intentionally simple:

- `world.py` owns the exact causal law and sampling;
- `protocol.py` owns visible rendering and decoding;
- `metrics.py` owns the three diagnostics;
- `seeds.py` owns the exact candidate parameters.

New abstractions should be added only when two real implementations need them.
The current candidate status and remaining validation boundary are recorded in
[the seed environment contract](docs/seed-environment.md).
