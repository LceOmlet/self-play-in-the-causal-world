# Terminal-Quality Reward v1

`terminal-quality-v1` is the frozen scalar reward for the current RL milestone.
Every legal terminal answer receives one exact quality value (Q\in[0,1])
derived from the raw diagnostics owned by `task_scoring.py`. Experimental
observations consumed, query count, turn count, and token count are reported
separately and never alter (Q).

## Task rewards

For an ATE prediction \(\hat\tau\) and truth \(\tau\), both in \([-1,1]\),

\[
Q_{\mathrm{ATE}}=1-\frac{|\hat\tau-\tau|}{2}.
\]

For an individual-counterfactual prediction \(\hat q\) and hidden certified
compatible interval \([L,U]\),

\[
d=\max(L-\hat q,0,\hat q-U),\qquad Q_{\mathrm{CF}}=1-d.
\]

Every point in the certified scoring interval receives full quality. For an
`epsilon_sharp` result, that safe outer interval extends each unknown sharp
endpoint by at most `0.002`. The distance is not normalized by interval width
and no point inside the certified interval is preferred.

For a deployment decision with the scorer-owned exact probability regret,

\[
Q_{\mathrm{decision}}=1-\mathrm{regret}.
\]

Every tied zero-regret intervention receives full quality.

For a predicted backdoor-adjustment family \(\widehat{\mathcal A}\) and truth
family \(\mathcal A^\star\), define the Dice similarity between two adjustment
sets as

\[
s(A,B)=\frac{2|A\cap B|}{|A|+|B|},
\]

with \(s(\varnothing,\varnothing)=1\). Let \(M\) be a maximum-weight one-to-one
matching between the two families. Then

\[
Q_{\mathrm{backdoor}}
=\frac{2\sum_{(A,B)\in M}s(A,B)}
       {|\widehat{\mathcal A}|+|\mathcal A^\star|}.
\]

The value is one exactly when the predicted family contains all and only the
true inclusion-minimal adjustment sets. Exact family match remains the task
success metric.

For mediator-set F1 \(F_M\) and consecutive-path-edge F1 \(F_O\),

\[
Q_{\mathrm{mediator}}=\frac{F_M+F_O}{2}.
\]

Task success requires exact match of both the mediator set and the path-edge
set; the component F1 values remain separate diagnostics.

## Completion and failure boundary

- A model-caused trajectory that ends without a legal terminal answer receives
  quality zero.
- A recoverable intermediate protocol error does not alter the quality of a
  later legal terminal answer.
- An environment or verifier failure produces no model-training sample.
- Binary compatibility, zero-regret, and exact-match indicators are evaluation
  diagnostics rather than training rewards.

`rewards.py` performs only this scalarization. It consumes the exact mapping
returned by `score_terminal_answer`; it does not parse model text or recompute
truth. `WorldSpecEpisode` exposes both the raw terminal score and the resulting
quality, and RL adapters should use the latter without reconstructing either
artifact. The runtime preserves exact `Fraction` values; an RL adapter may
convert the final reward to `float` only at the trainer boundary.
