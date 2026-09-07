# Uniform Task-Family Training Mixture v1

## Frozen distribution

RL training assigns equal mass to the five task families admitted by
`TASK_FAMILY_QUERY_TYPES`:

For the official DAPO entry point, these are the input proposal weights.
Official dynamic filtering changes the distribution of accepted update groups;
it must not be reported as preserving a 20% accepted fraction per family.

| Task family | Training mass |
| --- | ---: |
| `ate` | 20% |
| `individual_counterfactual_probability` | 20% |
| `backadj_minimal_sets` | 20% |
| `best_intervention` | 20% |
| `mediator_set` | 20% |

For `iter_sampled_seeds(..., count=S,
query_types=TASK_FAMILY_QUERY_TYPES)`, the materialized dataset contains
exactly `S` episodes from every family, rather than five IID family draws per
seed. A training consumer may deterministically shuffle that dataset before
batching. It must not change the family weights.

`TASK_FAMILY_QUERY_TYPES` is an explicit admission list rather than an alias
for the query registry. Registering a future query therefore cannot silently
change this training distribution.

## Within-family sampling

The equal family weights do not replace or rebalance the existing conditional
samplers:

- node count is `n ~ Uniform{8, ..., 16}` under the default grammar;
- structural-eligibility retries keep the sampled node count fixed;
- all five task families draw their minimum adjustment-set size uniformly from
  `{0, ..., floor(n / 3)}` before same-size structural retries;
- all current families have two readonly query anchors, leaving `n - 2`
  non-anchor variables eligible for hard intervention;
- all non-anchor variables are legal intervention targets and the maximum
  joint measurement width is n, subject to the
  [population identification contract](population-identification-v1.md);
- the previous independent `M0 ~ Uniform{1, ..., n}` draw determines scalar
  budget `M0 * 2^e` only; opening permissions does not enlarge that budget.

Roles, state anchors, structures, mechanisms, and task truths retain their
existing family-conditional distributions. In particular, v1 does not
oversample rare answers, hard instances, nonempty adjustment families, or any
difficulty band.

## Boundary

This contract freezes the RL training task-family mixture only. It does not
define minibatch ordering, a curriculum, benchmark weights, optimizer
settings, rollout allocation, or an RL algorithm.
