# Counterfactual solver closure optimization run

## Goal

Increase exact or epsilon-sharp individual-counterfactual closure on the frozen
distribution while preserving CPT-World generation, task/query/reward
semantics, the Markovian compatible-SCM set, the 0.001 conditional endpoint
tolerance, the five-second per-endpoint allowance, fallback behavior, and
formal-regression closure. Every optimization applies to a renaming- and
graph-isomorphism-invariant mathematical instance class.

## Result

| Cohort | Baseline | Retained solver | Repeats |
|---|---:|---:|---:|
| Frozen distribution | 23/30 | 24/30 | 3/3 |
| Formal regression | 20/30 | 20/30 | 3/3 |

The retained result contains 22 exact and two epsilon-sharp frozen endpoints;
formal regression contains 18 exact and two epsilon-sharp endpoints. The
related suite passed 132 tests with one existing skip, and the focused
counterfactual suite passed 26/26.

## Retained mechanism

Each twin-world elimination message has two fixed one-world marginals under
the supplied CPT-World. Their Frechet interval is therefore a valid bound on
the joint message for every compatible Markovian SCM. The aggregate profiler
found 90,431 stricter upper bounds and 16,428 positive lower bounds among
145,174 message cells. This strengthening alone stayed at 23/30.

For dynamic response blocks, every completed exact root pricing round also
provides the minimum reduced cost over every omitted deterministic response.
Correcting each block's normalization dual by that minimum converts the
restricted root LP bound into a rigorous full-master bound. Combining this
bound with the feasible primal certified one additional endpoint within the
unchanged conditional tolerance, raising frozen closure to 24/30.

The implementation never inspects seeds, labels, topology names, CPT
fingerprints, or a finite failure list. The added certificate is reported as
epsilon-sharp; incomplete pricing is never relabeled exact.

## Cost

The retained frozen runs took 359.32, 371.05, and 372.74 seconds in total, with
P95 per-task wall times of 146.09, 146.93, and 146.89 seconds. The result is a
closure improvement, not a global latency improvement.
