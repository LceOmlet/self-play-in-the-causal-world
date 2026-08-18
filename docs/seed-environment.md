# Candidate QN seed environment

Status: **candidate**, not yet a final released benchmark distribution.

The environment contains three binary roles: two focal variables and one
isolated variable. A hidden world has exactly one positive focal edge, no hidden
confounding, and an isolated third variable. Forward and reverse worlds have the
same exact observational joint law, so interventions are necessary.

| Seed | Difficulty | `P(child=1 | parent=0)` | `P(child=1 | parent=1)` | Active effect |
| --- | --- | ---: | ---: | ---: |
| `QN-EASY` | easy | `1/10` | `9/10` | `4/5` |
| `QN-MEDIUM` | medium | `3/10` | `7/10` | `2/5` |
| `QN-HARD` | hard | `9/20` | `11/20` | `1/10` |

All probabilities are represented by `fractions.Fraction` until deterministic
sampling or terminal serialization.

## Reproducible interventions

The sampler does not consume one global random stream. Each tuple

```text
(tape key, intervention target, intervention value, arm-local sample index)
```

owns a deterministic SHA-256 draw. Consequently:

- splitting one batch into smaller batches preserves the combined observations;
- interleaving another intervention arm does not change either arm's outcomes;
- paired policies can be compared on the same potential-outcome tape;
- all difficulties, truths, and surface layouts within one replicate share that
  tape, so controlled comparisons are not confounded with a different random
  sample.

The sampler obtains all masses from the same exact hard-do distribution used by
the semantic tests; there is no second hand-written simulation law.

## Surface controls

The renderer exposes stable opaque three-letter identifiers that exclude
`A/B/C/X/Y/Z`. Role assignment, target-list order, and effect-line order are
fully crossed, producing 72 deterministic surface layouts for a public label
seed. This controls surface preference; it is not itself a claim that every
possible linguistic bias has been eliminated.

`SeedEpisode` is an internal audit carrier containing the hidden world. Model
adapters must consume only `VisibleTask` through `render_initial_messages`.
Opaque episode/tape identifiers are only join keys, not secrets or access-control
boundaries, and must never be appended to the prompt.

## Not frozen yet

- the number of repeats per seed, truth, and layout;
- the train/validation/test split;
- a scalar reinforcement-learning reward;
- a self-play opponent or curriculum;
- the final bridge from the formal CPT-world verifier to a released artifact.

Those decisions should be introduced only with their own executable contracts
and tests. They are not hidden inside the current candidate builder.
