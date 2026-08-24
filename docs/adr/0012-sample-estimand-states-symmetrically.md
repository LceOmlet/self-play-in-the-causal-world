---
status: accepted
implementation: planned
---

# Sample estimand state anchors symmetrically

State anchors belong to the terminal estimand, not to the episode's experimental-action protocol.

For ATE and counterfactual-transition tasks with intervention variable \(X\) and outcome \(Y\):

1. sample \((x_{\mathrm{ref}},x_{\mathrm{cmp}})\) uniformly from the \(d_X(d_X-1)\) ordered pairs of distinct states of \(X\);
2. sample \(y^\star\) uniformly from the \(d_Y\) states of \(Y\).

The ATE target is the event-probability contrast

\[
P(Y=y^\star\mid do(X=x_{\mathrm{cmp}}))-
P(Y=y^\star\mid do(X=x_{\mathrm{ref}})).
\]

The counterfactual-transition target concerns

\[
P(Y_{x_{\mathrm{cmp}}}=y^\star,\;Y_{x_{\mathrm{ref}}}\ne y^\star).
\]

For an experimental-decision task, sample the target outcome state \(y^\star\) uniformly from all outcome states. The candidate intervention values remain the full state set of the decision variable. Backdoor-adjustment and mediator-structure tasks do not sample state anchors.

The state-anchor draw does not modify K, M, manipulability, readability, batch sizes, feedback laws, or budget. No draw is rejected using the resulting effect, direction, interval, or decision gap.

The current implementation's fixed `state_0`/`state_1` defaults are superseded and have not yet been migrated.
