# Optimization goal

Improve the fraction of generated individual-counterfactual CPT-World tasks
that receive an exact or epsilon-sharp certificate within the frozen endpoint
budget.

The optimization succeeds only when all of the following hold:

1. The aggregate closure rate on the frozen distribution cohort increases
   beyond measurement noise.
2. The aggregate closure rate on the formal regression cohort does not fall.
3. The optimized solver agrees with the original exact owner on every
   tractable parity case.
4. CPT-World generation, task and query semantics, the compatible Markovian
   SCM set, rewards, numerical tolerance, fallback behavior, and endpoint
   allowance remain unchanged.
5. Every accepted fast path is triggered by a variable-renaming-invariant and
   graph-isomorphism-invariant mathematical property that describes an entire
   instance class.

## Evidence rule

A hypothesis is admissible only when supported by both:

- an aggregate function-level profile or solver-wide structural count; and
- a semantics-preserving argument that applies to the full triggering class.

Individual seeds may demonstrate a symptom after a hypothesis exists. Seed
identities, finite failed-case lists, variable labels, topology names, exact
CPT fingerprints, and enumerated graph shapes cannot generate, tune, trigger,
or justify an optimization.

## Frozen baseline

- Distribution cohort: 23/30 certified, closure rate 0.7667.
- Formal regression cohort: 20/30 certified, closure rate 0.6667.
- Required improvement: at least one additional distribution-cohort world,
  with no formal-cohort regression and all semantic gates passing.
