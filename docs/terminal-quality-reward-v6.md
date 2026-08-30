# Terminal-Quality Reward v6

Superseded by [`terminal-quality-reward-v7`](terminal-quality-reward-v7.md).

All five task families pass their terminal quality unchanged to GRPO. There is
no additional training-time reward transformation.

For each numerical task with model error `E` and observational shortcut error
`B`, terminal quality is

\[
Q=\frac{B+s}{B+s+E},
\qquad
s=\frac{1}{\sqrt{2048}}.
\]

The fixed `2048` is the public number of sample rows per observation-bandwidth
unit. The implementation stores the IEEE-754 value of this irrational
resolution as an exact `Fraction`, so every subsequent scalarization operation
remains deterministic.

An exact answer receives one. For a fixed task, quality decreases continuously
with model error. As `B` approaches zero, an observational answer approaches
one continuously; when `B` is large relative to the sampling resolution, its
quality approaches one half. No numerical separability threshold is used. If
an observational conditional is undefined, the existing absolute-error
quality is used because no observational shortcut exists.

The task errors are:

- ATE: total-variation error of the complete categorical effect vector.
- Individual counterfactual ROI: mean certified error of the lower and upper
  endpoints.
- Best intervention: raw outcome-probability regret of the returned deployment
  state. Its shortcut error is the raw causal regret of the state selected by
  observational conditional probabilities.
- Backdoor adjustment: maximum-matching soft family F1.
- Mediator structure: the mean of mediator F1 and path-order F1.

Best intervention returns only the selected deployment state. Experimental
cost remains a separate diagnostic.
