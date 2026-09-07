# DoLens

DoLens is a reference environment for active causal reasoning in hidden, finite
CPT worlds. A model sees an anonymized task, gathers selected
measurements through passive observation or batched hard interventions, and
returns one structured terminal answer. The hidden graph, CPT entries, internal
variable names, task truth, and scorer are never rendered to the model.

The current milestone provides the environment, the shared task-generation and
interaction pipeline, the frozen
[Terminal-Quality Reward v10](docs/terminal-quality-reward-v10.md), the
[uniform task-family training mixture](docs/training-mixture-v1.md), and an
official verl DAPO post-training integration. It does not yet freeze a self-play
data-generation algorithm, difficulty bands, or a final benchmark aggregation
mixture.

The active correctness work is tracked in the
[2026-09-07 goal and remaining proof obligations](docs/rl-correctness-goal-20260907.md).
[Reproducible research and execution evidence](research/rl_correctness_20260907/README.md)
include successful checks and disqualified runs. The current training entry is
`scripts/run_official_dapo.sh`; historical TRL patches and scripts are retained
for provenance only. Long training remains stopped, and old checkpoints must
not initialize new training. An isolated audit is not evidence of learning gains.

## Paper

The ICLR 2026 manuscript presents the DoLens intuition, shared interaction
loop, five causal tasks, compatible-set counterfactual evaluator, certified
execution engine, and validation results:

- [compiled paper](paper/output/pdf/dolens-iclr.pdf)
- [LaTeX source](paper/main.tex)
- [evidence map](paper/evidence_map.md)
- [paper blueprint](paper/blueprint.md)

## Implemented task families

All tasks use the same `WorldSpec` sampler, renderer, interaction runtime, and
task-specific truth owners. The model never receives the graph or CPT tables.

| Task | What the model returns | What the task tests |
| --- | --- | --- |
| Categorical total effect (ATE) | The complete outcome-state effect vector for one ordered pair of treatment states. | Estimating an interventional distributional change when the treatment and outcome are readonly. |
| Individual counterfactual ROI | The lower and upper endpoints of the same individual's sharp counterfactual probability interval, conditioned on their factual treatment and outcome. | Reasoning over every causally sufficient mechanism compatible with the hidden CPT world, without selecting one hidden SCM. |
| Experimental decision | One final state of a readonly deployment variable. | Using experiments on other legal targets to minimize or maximize the requested outcome event. |
| Backdoor adjustment | One complete adjustment set, which may be empty. | Finding variables that satisfy the graphical backdoor criterion without conditioning on treatment descendants. |
| Mediator set and order | All mediators on directed treatment-to-outcome paths and the consecutive directed path edges. | Recovering the query-relevant causal pathway without reconstructing the full graph. |

Pinned real-world and motif seeds under `data/seeds/` are validation fixtures.
They are not a second sampler and do not define the generated task distribution.

## Reward and benchmark metrics

### Terminal-Quality Reward v10

Every legal terminal answer receives one continuous quality value between zero
and one. An unfinished episode or an illegal terminal answer receives zero. The
environment owns this value, and the tool adapter passes it through unchanged.
Experimental cost, query count, trajectory length, token usage, and wall-clock
time are recorded separately and never folded into terminal quality.
The official DAPO optimizer adds its own soft overlong penalty; its shaped
training reward must be distinguished from the environment's raw quality.

| Task | Error used by the reward | Meaning of the terminal quality |
| --- | --- | --- |
| Categorical total effect | Total-variation error between the predicted and true complete effect vectors. | Accuracy is calibrated against the error of replacing the causal effect with the corresponding observational conditional effect, with one fixed sampling-resolution allowance. Exact recovery receives one and larger vector error lowers quality continuously. |
| Individual counterfactual ROI | Mean absolute distance of the two predicted endpoints from their certified endpoint ranges. | Accuracy is calibrated against the observational counterfactual plug-in interval with the same fixed sampling-resolution allowance. Both endpoints contribute equally, and exact certified recovery receives one. |
| Experimental decision | Regret of the chosen deployment state, normalized by the full causal value range available in that world. | Accuracy is calibrated against the normalized regret of the state selected from observational conditionals. An optimal state receives one; increasingly costly decisions receive lower quality. |
| Backdoor adjustment | Number of variable additions or removals needed to reach the nearest graphically valid adjustment set. | Any valid set receives one. Every additional edit subtracts the same amount, fixed by the configured maximum graph size rather than by the current world. |
| Mediator set and order | Set disagreement for the mediators and directed-edge disagreement for their path order. | Terminal quality is the equal average of mediator F1 and path-order F1. |

For the first three tasks, the fixed sampling-resolution allowance is set once
from the public budget of 2,048 sample rows per observation-bandwidth unit. It
is not fitted per model or per episode. If the relevant observational
conditional is undefined, the scorer falls back to the task's bounded
absolute-error quality because that episode has no observational plug-in
answer to use as a calibration reference. When all deployment states have the
same causal value, every state is optimal and has zero regret.

The full versioned contract is
[Terminal-Quality Reward v10](docs/terminal-quality-reward-v10.md). Historical
reward documents remain in `docs/` only to explain earlier experiments; they
are not active training contracts.

### Benchmark reporting

Terminal quality is a training-compatible summary, not a replacement for the
underlying task metrics. A benchmark result should report every task family
separately using the following definitions:

| Task | Primary benchmark metrics | Direction |
| --- | --- | --- |
| Categorical total effect | Complete-vector RMSE and mean total-variation error; observational-shortcut total-variation error is reported beside them. | Lower is better. |
| Individual counterfactual ROI | Endpoint MAE, endpoint RMSE, exact endpoint-recovery rate, and predicted and certified interval widths. | Lower endpoint error and higher exact recovery are better. |
| Experimental decision | Optimal-action accuracy, raw causal regret, and normalized causal regret; concordant and observationally discordant strata are reported separately. | Higher accuracy and lower regret are better. |
| Backdoor adjustment | Nearest-valid-set edit distance, graphically valid-set rate, and mean v10 terminal quality. | Lower edit distance and higher valid-set rate are better. |
| Mediator set and order | Mediator precision, recall, and F1; path-order precision, recall, and F1; and joint exact-match rate. | Higher is better. |

Complete-vector RMSE gives every outcome-state component equal weight. Endpoint
errors measure distance to the certified endpoint ranges, so a solver's safe
numerical enclosure is not counted as model error. Raw regret is the causal
value lost by the selected action; normalized regret expresses that loss
relative to the complete causal value range in the same world.

Every task table must also include mean v10 terminal quality and valid-terminal
coverage. Counterfactual truth certification coverage, exact versus
epsilon-sharp certification counts, node-count histograms, task-answer
distributions, and observational-shortcut baselines are dataset diagnostics
and must be published beside model results. Query count, sampled rows, scalar
measurements, input and output tokens, and wall-clock time are resource
diagnostics rather than components of task accuracy. The current repository
freezes equal task-family mass for training only; until a benchmark aggregation
mixture is frozen, it does not define or endorse one overall five-task score.

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
  support is every integer from 8 through 16;
- every node's domain size is uniform from two states through the configured
  maximum, currently five states;
- a topological order is sampled uniformly; each node then draws its parent
  count uniformly from zero through one third of the world size, rounded down
  and capped by the available predecessors, followed by a uniform parent subset
  and continuous float64 CPT parameters; custom grammars below eight nodes
  retain the legacy maximum parent count of three;
- each conditional CPT samples a uniform undirected interaction graph on its
  parents, activates the pure categorical interaction blocks indexed by that
  graph's nonempty cliques, and combines the active orthogonal blocks with
  simplex-uniform energy shares;
- ET-V2 maps the combined score direction to probabilities with an exponential
  tilt, so an unrelated rare state cannot impose a global additive zero-wall
  on every effect; its bounded uniform amplitude has one unit of expected
  squared score energy; a binary-preserving contextual parent-pair scale
  removes geometric attenuation caused by additional parent states, parent
  configurations, or parents without selecting a distinguished contrast;
- legal query anchors are derived from the sampled world;
- all five task families draw the minimum adjustment-set size uniformly from
  zero through one third of the node count, rounded down, and resample
  same-size structures until a role with that value exists;
- task-family answerability remains an optional diagnostic and is not applied
  as a generation label, filter, or admission check;
- every non-anchor variable is available for hard intervention, and all
  variables can be measured jointly;
- the actual world and permissions must satisfy the constructive
  [population identification contract](docs/population-identification-v1.md),
  including positivity, causal minimality and a public source-ancestry prior;
- a budget width `M0` is drawn independently and uniformly from one through
  the node count, using the former measurement-width RNG draw;
- an exponent is sampled independently and uniformly from `11`, `12`, `13`,
  and `14`; the explicit scalar-observation budget remains `M0 * 2^exponent`.

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

The production paths avoid materializing the full joint state space for every
interactive query. Runtime is governed by the requested batch, the selected
measurements, and the induced elimination width; the full-joint path remains
only as a semantic reference for parity tests.

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

Run the end-to-end prompt, experiment, feedback, and terminal-scoring demo:

```bash
python scripts/demo_worldspec_runtime.py
```

Sample all five task types through the main pipeline:

```python
from cpt_world import TASK_FAMILY_QUERY_TYPES, WorldGrammar, iter_sampled_seeds

tasks = iter_sampled_seeds(
    WorldGrammar(),
    count=20,
    query_types=TASK_FAMILY_QUERY_TYPES,
)
```

This materializes exactly 20 task manifests from each family (100 total).
Training consumers may deterministically shuffle the stream; they must not
reweight the five families. Within `best_intervention`, each aligned five-slot
block contains one observationally concordant and four observationally
discordant optimal-action sets. A discordant task is admitted only when choosing
the observationally optimal state loses at least about 0.044 in absolute causal
outcome probability. This is the two-standard-error resolution of the minimum
`2^11 = 2048` per-bandwidth sample unit under an ideal two-action split.
The terminal score remains normalized causal regret; this admission condition
only removes observational reversals that are statistical near-ties.

## Official DAPO post-training

`scripts/run_official_dapo.sh` executes the unmodified official verl DAPO recipe.
`scripts/prepare_verl_cpt_data.py` materializes the existing balanced task stream
as standard parquet. See [source provenance, objective and verification
status](docs/official-dapo-integration-20260906.md). This entry point uses an
independent environment without TRL. The historical TRL launcher is retired,
and old RL checkpoints are disqualified as initialization and evaluation
evidence. Formal training remains stopped while algorithm and environment
correctness are addressed.

This integration does not freeze a self-play algorithm. A future
self-play contract must specify how new worlds or curricula are proposed, how
opponents or generators are updated, and how generated episodes enter the
training distribution.

## Ownership boundaries

- `world_space.py`: world/task sampling, query anchors, and interaction masks;
- `query_truth.py`: exact hard-do laws and query truth;
- `world.py`: versioned action-keyed random tapes;
- `world_runtime.py`: command execution, selected measurements, and feedback;
- `rendering.py`: truth-free model-visible tasks;
- `task_scoring.py`: strict terminal parsing and task diagnostics;
- `rewards.py`: frozen scalarization of owner-produced terminal diagnostics;
- `seeds.py`: pinned validation-seed manifests.

The package deliberately uses concrete functions and immutable data records.
New abstractions should be introduced only when two real implementations need
them.
