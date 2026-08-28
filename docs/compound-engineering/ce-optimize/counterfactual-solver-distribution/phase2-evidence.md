# Phase 2 aggregate evidence

All measurements below aggregate the frozen cohorts. No seed identities or
per-instance outcomes were exposed to hypothesis generation.

## Function-level profile

- Public individual-counterfactual solving: 458.8 seconds.
- SCIP optimization calls: 335.9 seconds.
- Direct-terminal dispatch: 325.2 seconds.
- Sparse-model construction: 81.5 seconds.
- Twin-probability circuit construction: 78.9 seconds.
- Pricing callbacks: 2.8 seconds.

The optimization target is therefore global solving and exact structural
dispatch. Pricing-loop micro-optimizations cannot plausibly move the primary
closure metric first.

## Endpoint profile

- Terminal lower/minimize: 24 calls, 284.4 seconds, 3 uncertified exits.
- Terminal upper/maximize: 20 calls, 6.4 seconds, 1 uncertified exit.
- Full-objective models: 8 builds, 40.2 seconds.
- Terminal lower models: 24 builds, 11.2 seconds.
- Terminal upper models: 20 builds, 2.8 seconds.

The terminal lower endpoint is the dominant solve bottleneck. The existing
constant-zero lower certificate is currently checked only after the direct
terminal dispatcher, so direct-terminal instances cannot benefit from it.

## Structural coverage

| Aggregate class | Distribution cohort | Formal regression cohort |
|---|---:|---:|
| Direct treatment parent of outcome | 22/30 | 15/30 |
| Only outcome is treatment-affected | 16/30 | 12/30 |
| Direct terminal with constant-zero lower | 4/30 | 2/30 |
| Outer interval already within 0.002 | 0/30 | 1/30 |

The first experiment targets the 16/30 complete class where only the outcome
is affected. In this class all other outcome parents have the same value in
both worlds. Conditioning on their shared configuration reduces the sharp
joint bound to a probability-weighted sum of contextwise Frechet bounds.
Different shared configurations use disjoint outcome-response contexts, so
all contextwise extrema are simultaneously attainable by one response-function
distribution. This gives an exact closed form without changing the compatible
SCM set.

