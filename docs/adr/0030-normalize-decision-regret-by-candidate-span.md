---
status: accepted
implementation: implemented
amends: 0023-use-ceiling-sensitive-advantage-utility
---

# Normalize decision regret by the candidate probability span

For the fixed deployment variable, score the selected state by exact regret
relative to the full range of candidate-state outcome probabilities:

\[
Q=1-\frac{\mathrm{regret}}{p_{\max}-p_{\min}}.
\]

When the span is zero, every candidate is tied and receives one. Raw regret and
optimal-action accuracy remain separate diagnostics. The task renderer,
deployment candidates, world sampler, K/M surface, and experiment protocol are
unchanged.

## Fixed-seed validation

The rule was tested on 2,520 generated best-intervention tasks, stratified over
every default node count from 8 through 16, plus the pinned Cancer and Survey
decision seeds. Under the previous absolute reward, a uniform random action had
median quality `0.9319` despite only `0.3226` exact optimal-action accuracy. The
normalized rule moved its median quality to `0.5000`.

The correlation between candidate probability span and random-policy quality
fell in magnitude from `0.9537` to `0.0476`. The normalized median remained
`0.5000` in every node-count stratum and under both minimize and maximize
objectives. Optimal state indices remained approximately uniform. No generated
task had zero span or a normalized value outside `[0,1]`; an independent
900-task numerical audit found a minimum positive span of about `1.07e-6`.

The pinned wrong action changed from `0.6500` to `0` on Cancer and from
`0.8980` to `0` on Survey. These checks support relative decision quality as
the semantic reward while preserving raw regret as the absolute-consequence
diagnostic.
