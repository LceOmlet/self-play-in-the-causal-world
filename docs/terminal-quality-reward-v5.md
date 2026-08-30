# Terminal-Quality Reward v5

Superseded by [`terminal-quality-reward-v6`](terminal-quality-reward-v6.md).

`terminal-quality-v5` calibrates the three numerical causal tasks against the
corresponding oracle observational plug-in answer.  It changes neither the
hidden-world sampler nor the experiment protocol.

For a model error `E` and the error `B` of replacing the required causal law
with its observational conditional analogue, terminal quality is

\[
Q=\frac{B}{B+E}.
\]

An exact causal answer receives one.  The observational plug-in answer receives
one half whenever `B` exceeds the existing `1e-12` probability-law numerical
tolerance, and larger errors decrease continuously.  Below that tolerance,
the observational and causal terminal targets coincide and no terminal-only
score can distinguish their provenance; the task retains its previous
continuous absolute-error quality in that case.  The same fallback applies
when an observational conditional is undefined because its conditioning event
has zero mass.

## Task errors

- ATE: `E` is total-variation error between the predicted and true complete
  categorical treatment-effect vectors. `B` uses the vector obtained from
  `P(Y|X=x1)-P(Y|X=x0)`.
- Individual counterfactual ROI: `E` is the current mean certified endpoint
  error. `B` applies the same error to the Frechet interval obtained from the
  two observational conditional endpoint marginals.
- Best intervention: the model returns the outcome-event probability for every
  candidate deployment state. `E` is mean absolute error over all pairwise
  probability gaps. `B` replaces every hard-do probability by `P(Y|D=d)`.
  The predicted optimum, raw regret, and normalized regret remain diagnostics.

All five terminal qualities enter GRPO unchanged. Experimental cost remains
separate.
