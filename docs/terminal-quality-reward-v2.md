# Terminal-Quality Reward v2

Superseded by [`terminal-quality-reward-v3`](terminal-quality-reward-v3.md).

`terminal-quality-v2` changed only
the ATE component of v1: ATE now returns and scores the complete categorical
treatment-effect vector. Experimental observations consumed, query count,
turn count, and token count remain separate diagnostics.

## ATE

For outcome states \(y=0,\ldots,d_Y-1\), the truth is

\[
\tau_y=P(Y=y\mid do(X=x_1))-P(Y=y\mid do(X=x_0)).
\]

The model returns every component \(\hat\tau_y\). Both vectors are valid
differences of categorical distributions, so their components sum to zero and
their positive components sum to at most one. The raw scorer reports component
errors, full-vector L1 error, total-variation error, and mean squared component
error. The terminal reward is

\[
Q_{\mathrm{ATE}}=1-\frac{\lVert\hat\tau-\tau\rVert_1}{4}.
\]

For a binary outcome, \(\tau=(-a,a)\) and \(\hat\tau=(-\hat a,\hat a)\).
Therefore v2 reduces exactly to the v1 scalar reward
\(1-|\hat a-a|/2\).

## Other tasks

The individual-counterfactual, experimental-decision, backdoor-adjustment,
and mediator rewards are unchanged from
[`terminal-quality-v1`](terminal-quality-reward-v1.md).

`rewards.py` consumes only the exact diagnostics returned by
`score_terminal_answer`; it does not parse model text or recompute task truth.
