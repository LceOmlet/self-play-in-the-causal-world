# DoLens

DoLens is a reference environment for active causal reasoning in hidden, finite
CPT worlds. A model sees an anonymized task, gathers selected
measurements through passive observation or batched hard interventions, and
returns one structured terminal answer. The hidden graph, CPT entries, internal
variable names, task truth, and scorer are never rendered to the model.

The current milestone provides the environment and task-generation pipeline.
It does **not** yet define a self-play algorithm, a scalar training reward, or a
final benchmark mixture.

## Paper

The ICLR 2026 manuscript presents the DoLens intuition, shared interaction
loop, five causal tasks, compatible-set counterfactual evaluator, exact
execution engine, and validation results:

- [compiled paper](paper/output/pdf/dolens-iclr.pdf)
- [LaTeX source](paper/main.tex)
- [evidence map](paper/evidence_map.md)
- [paper blueprint](paper/blueprint.md)

## Implemented task family

All tasks use the same `WorldSpec` sampler, renderer, interaction runtime, and
exact truth owners.

| Query | Terminal output |
| --- | --- |
| ATE | The numerical total effect between the named treatment and outcome. The generic sampler keeps both query endpoints readonly, so evidence must come from permitted indirect experiments or passive observations. |
| Individual counterfactual probability | One scalar probability for the same individual under a named counterfactual treatment, conditioned on that individual's factual treatment and observed outcome. A hidden compatible interval verifies the scalar. |
| Experimental decision | A deployment intervention that minimizes or maximizes the named outcome event. Experiment targets and the deployment decision are separated. |
| Backdoor adjustment | Every minimal valid adjustment set for the named treatment–outcome query. |
| Mediator set and order | The mediator variables and the direct path-order relations between them. |

Pinned real-world and motif seeds under `data/seeds/` are validation fixtures.
They are not a second sampler and do not define the generated task distribution.

## Model-visible interaction

The initial prompt exposes only:

- opaque variable labels and finite state domains;
- the causal query and required terminal JSON schema;
- legal hard-do targets and readable variables;
- the selected-measure width `M` and remaining scalar-observation budget.

The model may issue:

```json
{"type":"observe","measure":["RTG"],"batch_size":4}
```

or:

```json
{"type":"intervene","target":"ITJ","value":"state_1","measure":["RTG"],"batch_size":8}
```

Each batch contains IID samples from one fixed hidden world. Feedback reports
joint counts only for the requested variables. The count map is sparse:
omitted requested-variable assignments have count exactly zero.

## World and interaction-surface sampling

`WorldGrammar` declares the probability model rather than enumerating a fixed
grid. In the current implementation:

- node count is uniform over the configured `node_counts` support; the default
  support is every integer from 3 through 15;
- every node's domain size is uniform over `2..max_domain_size`, currently up to
  five states;
- a topological order is sampled, followed by a forward-edge subset and
  continuous float64 CPT parameters;
- legal query anchors are derived from the sampled world;
- task-family answerability remains an optional diagnostic and is not applied
  as a generation label, filter, or admission check;
- conditional on the eligible non-anchor intervention variables, the legal
  hard-do width `K` is uniform over `1..|eligible|`, and the subset is uniform
  conditional on that width;
- `M` is sampled independently and uniformly over `1..n`.

Thus `K` and `M` change the evidence surface without redefining whether the
underlying world/query instance has an answer.

## Probability engine

The probability layer has one semantic owner and two execution paths:

- `worldspec_interventional_distribution` retains the original
  full-joint enumeration as a reference implementation;
- selected marginals use variable elimination with the same CPT factors and
  hard-do mechanism replacement;
- interactive batches use ancestral sampling with a versioned, per-node
  action-keyed outcome tape;
- the tape remains invariant to surface renaming, requested-measure projection,
  batch splitting, and action interleaving.

For `n` nodes, maximum domain size `d`, selected width `m`, induced elimination
width `w`, and batch size `b`, the main costs are:

- old full-joint batch path: `O((n + b) d^n)` time and `O(d^n + d^m)` memory;
- ancestral batch path: `O(b n d)` time and
  `O(n + min(b, d^m))` working/output memory;
- selected marginal: `O(n d^(w+1) + d^m)` time.

On the included sparse binary-chain benchmark, the measured median speedups
were:

| Nodes | Joint states | Exact one-node marginal | 64-sample batch |
| ---: | ---: | ---: | ---: |
| 10 | 1,024 | 13.8x | 7.3x |
| 14 | 16,384 | 230.4x | 58.4x |
| 15 | 32,768 | 338.4x | 102.5x |

Reproduce the comparison with:

```bash
python scripts/benchmark_worldspec_acceleration.py --nodes 15 --batch-size 64
```

The benchmark uses a fixed `Fraction` fixture and first requires exact equality
with the full-joint reference. Generated float64 worlds use the shared CPT
validity tolerance instead.

## Quick start

```bash
python -m pip install -e ".[dev]"
python -m unittest discover -s tests -v
python -m ruff check .
python -m ruff format --check .
```

Run one generic episode:

```python
from cpt_world import Budget, OutcomeTape, WorldSpecEpisode

episode = WorldSpecEpisode(
    world,
    seed,
    OutcomeTape("preregistered-tape-key"),
    budget=Budget(max_observations=16),
)
messages = episode.initial_messages()
step = episode.step(
    '{"type":"intervene","target":"ITJ","value":"state_1","measure":["RTG"],"batch_size":8}'
)
```

Sample all five task types through the main pipeline:

```python
from cpt_world import WorldGrammar, iter_sampled_seeds

tasks = iter_sampled_seeds(
    WorldGrammar(),
    count=20,
    query_types=(
        "ate",
        "individual_counterfactual_probability",
        "backadj_minimal_sets",
        "best_intervention",
        "mediator_set",
    ),
)
```

For prompt-to-feedback examples, run:

```bash
python scripts/demo_worldspec_runtime.py
```

## Ownership boundaries

- `world_space.py`: world/task sampling, query anchors, and interaction masks;
- `query_truth.py`: exact hard-do laws and query truth;
- `world.py`: versioned action-keyed random tapes;
- `world_runtime.py`: command execution, selected measurements, and feedback;
- `rendering.py`: truth-free model-visible tasks;
- `task_scoring.py`: strict terminal parsing and task diagnostics;
- `seeds.py`: pinned validation-seed manifests.

The package deliberately uses concrete functions and immutable data records.
New abstractions should be introduced only when two real implementations need
them.
