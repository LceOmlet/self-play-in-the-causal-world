---
status: accepted
implementation: implemented
---

# Use the full categorical treatment effect

ATE no longer samples one outcome category. It keeps the uniformly sampled
ordered treatment-state pair and returns the probability difference for every
outcome category:

\[
\tau_y=P(Y=y\mid do(X=x_1))-P(Y=y\mid do(X=x_0)),
\qquad y=0,\ldots,d_Y-1.
\]

This removes the projection onto one randomly selected category while retaining
the complete interventional distributional change. The vector sums to zero.
For binary outcomes it is `(-tau, tau)`, so it preserves the old scalar target
exactly. It adds no outcome ordering, utility, or target-event choice.

The terminal JSON maps every rendered `state_i` token to one effect component.
The scorer uses all components and Reward v2 uses full-vector L1 error. The
counterfactual and experimental-decision tasks keep their target outcome event,
because that event is part of their estimand rather than a projection artifact.

## Fixed-seed diagnostic

The default grammar was evaluated on ATE sampler seeds `0..399` without answer
filtering or resampling. The old statistic reconstructs the former uniformly
sampled outcome category; the new statistic is half the L1 norm of the complete
effect vector, i.e. the total-variation distance between the two interventional
outcome distributions.

| target statistic | q10 | median | q90 | fraction below 0.01 |
| --- | ---: | ---: | ---: | ---: |
| old absolute single-category effect | 0.00151 | 0.04178 | 0.35737 | 25.75% |
| new full-vector TV magnitude | 0.00665 | 0.11126 | 0.57490 | 12.75% |

For a zero-vector prediction, mean terminal reward falls from `0.93903` under
the old scalar projection to `0.89856` under Reward v2; median reward falls from
`0.97911` to `0.94437`. Binary tasks are unchanged exactly. For outcome domain
sizes three, four, and five, median target magnitude changes from
`0.03582/0.03189/0.03163` to `0.09356/0.12043/0.13306` respectively.
