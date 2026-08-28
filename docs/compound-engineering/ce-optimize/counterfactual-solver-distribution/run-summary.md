# Counterfactual solver closure optimization run

## Goal

Increase exact or epsilon-sharp individual-counterfactual closure on the frozen
distribution while preserving CPT-World generation, task/query/reward
semantics, the Markovian compatible-SCM set, the 0.002 endpoint tolerance, the
five-second per-endpoint allowance, and formal-regression closure. Every
optimization must apply to a renaming- and graph-isomorphism-invariant instance
class; finite-case and seed-specific optimization is forbidden.

## Baseline and final best

| Cohort | Baseline | Final best |
|---|---:|---:|
| Distribution | 23/30 | 23/30 |
| Formal regression | 20/30 | 20/30 |

No candidate produced a stable closure gain, so baseline remains the best and
all solver candidates were reverted.

## Tested hypotheses

1. Exact direct-only shared-parent Frechet mixture: exact, but only accelerated
   already-closed instances.
2. Pre-model constant-zero direct lower certificate: did not reduce unresolved
   count and regressed one formal sample.
3. Exact priced routing for layered cases rejected by explicit objective-size
   estimation: primary closure stayed unchanged; formal evidence was within
   the paired noise threshold.
4. Exact globally affine and one-sided terminal endpoint certificates: both
   cohorts stayed at baseline closure.

## Evidence-based next boundary

The current aggregate evidence rules out further work on these four fast-path
families as a justified route to higher closure. The next phase must first
classify unresolved endpoints by mathematical obstruction in aggregate:
affected-mechanism count, context width, upstream transport vertex product,
and the exact certification stage that fails. No additional solver change is
supported until that classification identifies a closure-limiting class.
