# Terminal-Quality Reward v3

Superseded by [`terminal-quality-reward-v4`](terminal-quality-reward-v4.md).

`terminal-quality-v3` changes the individual-counterfactual terminal from one
compatible point to the two endpoints of its identified region. ATE keeps the
full-vector v2 reward. Experimental-decision, backdoor-adjustment, and mediator
rewards keep their v1 definitions. Experimental cost remains a separate
diagnostic.

## Individual counterfactual identified region

For the model prediction \([\hat L,\hat U]\), both endpoints must lie in
\([0,1]\) and \(\hat L\leq\hat U\). An exact truth owner returns the sharp
identified interval \([L,U]\), giving endpoint errors

\[
e_L=|\hat L-L|,\qquad e_U=|\hat U-U|.
\]

An `epsilon_sharp` truth owner returns safe outer endpoints \([L_o,U_o]\) and
a certified endpoint error \(\epsilon\). The unknown sharp endpoints therefore
lie in

\[
L\in[L_o,\min(1,L_o+\epsilon)],\qquad
U\in[\max(0,U_o-\epsilon),U_o].
\]

In this case, \(e_L\) and \(e_U\) are the distances from the predicted
endpoints to those two certified ranges. The scalar reward is

\[
Q_{\mathrm{CF}}=1-\frac{e_L+e_U}{2}.
\]

This is a continuous endpoint-accuracy reward. It has no direction threshold,
confidence field, interval-width normalization, or hard inside/outside test.

## Other tasks

- ATE: \(1-\lVert\hat\tau-\tau\rVert_1/4\), as in
  [`terminal-quality-reward-v2`](terminal-quality-reward-v2.md).
- Experimental decision: \(1-\mathrm{regret}\).
- Backdoor adjustment: maximum-matching soft family F1.
- Mediator structure: the mean of mediator F1 and path-order F1.
- An unfinished or illegal terminal answer: zero.

`rewards.py` consumes scorer-owned diagnostics only. It does not parse model
text or recompute task truth.
