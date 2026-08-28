---
status: accepted
implementation: implemented
requires: 0015-bound-counterfactuals-over-full-world-completions
---

# Predict the individual-counterfactual identified region

The formal individual-counterfactual terminal predicts the sharp identified
region, not one arbitrary compatible point. For the rendered factual evidence
and counterfactual event,

\[
q=P(Y_{x_{cf}}=y^\star\mid Y_{x_f}=y_f),
\]

the model returns

```json
{"type":"answer","lower":0.2,"upper":0.8}
```

The two values estimate the minimum and maximum of \(q\) over every finite
Markovian mechanism completion compatible with the hidden CPT-World. Exact
truth uses absolute endpoint error. `epsilon_sharp` truth uses distance to each
endpoint's certified range. Reward v3 is one minus mean absolute endpoint
error.

The internal query ID remains `individual_counterfactual_probability` for
serialized-seed compatibility. This change does not alter the CPT generator,
compatible mechanism class, counterfactual solver, sampled roles, K/M surface,
or interaction budget.
