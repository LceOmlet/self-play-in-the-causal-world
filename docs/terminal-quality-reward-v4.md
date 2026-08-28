# Terminal-Quality Reward v4

Superseded by [`terminal-quality-reward-v5`](terminal-quality-reward-v5.md).

`terminal-quality-v4` changes only the experimental-decision component of v3.
It normalizes probability regret by the complete outcome-probability span of
the fixed deployment variable's candidate states. Experimental cost remains a
separate diagnostic.

## Experimental decision

For each candidate deployment state `d`, the truth owner computes

\[
p_d=P(Y=y^\star\mid do(D=d)),\qquad
S=p_{\max}-p_{\min}.
\]

The raw probability regret of the selected state is

\[
r=
\begin{cases}
p_{\max}-p_{\hat d},&\text{maximize},\\
p_{\hat d}-p_{\min},&\text{minimize}.
\end{cases}
\]

For `S > 0`, the terminal reward is

\[
Q_{\mathrm{decision}}=1-\frac{r}{S}.
\]

Thus every optimal state receives one, every worst state receives zero, and
intermediate states are scored by their relative deployment value. If `S = 0`,
all states are tied, normalized regret is zero, and every state receives one.
The denominator depends only on scorer-owned task truth. Raw regret, normalized
regret, and optimal-action success are all reported separately.

Best-intervention quality now passes unchanged into group-mean subtraction; the
ceiling-sensitive trainer utility is not applied a second time.

## Other tasks

ATE and individual counterfactual ROI retain their v3 definitions. Backdoor
adjustment and mediator structure retain their existing definitions. An
unfinished or illegal terminal answer still receives zero.

`rewards.py` consumes scorer-owned diagnostics only. It does not parse model
text or recompute task truth.
