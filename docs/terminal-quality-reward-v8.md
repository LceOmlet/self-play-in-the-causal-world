# Terminal-Quality Reward v8

Superseded by [`terminal-quality-reward-v9`](terminal-quality-reward-v9.md).

`terminal-quality-v8` keeps the v7 numerical rewards for ATE, individual
counterfactual ROI, and best intervention. It replaces the backdoor terminal
object and reward.

The model returns one complete `adjustment_set`. For every treatment state
`x`, the scorer computes the standardized observational outcome law

```text
Q_set(Y | x) = sum_z P(Y | X=x, set=z) P(set=z)
```

using every submitted variable. It compares this law with `P(Y | do(X=x))`
by total variation and averages across treatment states. Let that error be `E`;
let `B` be the same error for the empty set. Backdoor terminal quality is

```text
1                    if E = 0
0                    if B = 0 and E > 0
B / (B + E)          otherwise.
```

The reward contains no adjustment-family matching, structural F1, graph-overlap
term, or product with a structural score. A mediator or collider included in
the submission can worsen `E`; an irrelevant harmless covariate does not receive
an artificial penalty. When several graphically valid adjustment sets exist,
any one that exactly recovers the hard-do distributions receives one.

All five owner-produced terminal qualities enter GRPO unchanged.
