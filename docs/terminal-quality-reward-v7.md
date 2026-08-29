# Terminal-Quality Reward v7

All five task families pass their terminal quality unchanged to GRPO. There is
no additional training-time reward transformation.

For each numerical task with model error `E` and observational shortcut error
`B`, terminal quality is

\[
Q=\frac{B+s}{B+s+E},
\qquad
s=\frac{1}{\sqrt{2048}}.
\]

The task errors are:

- ATE: total-variation error of the complete categorical effect vector.
- Individual counterfactual ROI: mean certified error of the lower and upper
  endpoints.
- Best intervention: regret of the returned state normalized by the complete
  candidate probability span. Its shortcut error is the causal regret of the
  observationally selected state divided by the same span. A zero span means
  that every state is tied, so both normalized regrets are zero.
- Backdoor adjustment: maximum-matching soft family F1.
- Mediator structure: the mean of mediator F1 and path-order F1.

Best intervention returns only the selected deployment state. Raw probability
regret, optimal-action accuracy, and experimental cost remain separate
diagnostics and do not determine terminal quality.
