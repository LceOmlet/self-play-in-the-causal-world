# Population identification with readonly query anchors

The previous random permanent removal of non-anchor interventions admitted
worlds with identical legal experiment distributions and different required
answers. Training cannot recover information absent from every legal
transcript. The new contract keeps the two query anchors readonly, opens every
other single-variable hard intervention, and allows the full joint observation.
The scalar observation budget is retained from the previous random draw.

This is a population-identification guarantee under explicitly checked
conditions. It does not promise exact estimation from a finite transcript,
efficient full-distribution reconstruction, or improving RL checkpoints.

## Constructive proof without a faithfulness assumption

Let the readonly source and outcome be X and Y. The public task prior says X is
an ancestor of Y. The remaining variables are Z. Assume causal sufficiency, a
finite acyclic causal model, strictly positive CPT entries, and causal
minimality: every declared parent affects its child's CPT in some parent
context. The generator's actual CPTs are checked for positivity and minimality;
these are not merely almost-sure claims about ideal random numbers.

Write the observational joint law as `P(v) = product_i f_i(v_i | pa_i)`.
For each non-anchor i, a legal single-variable intervention fixes `V_i=v_i`.
For any full assignment v consistent with that intervention, truncated
factorization gives

```
P(v) / P(v_without_i | do(V_i=v_i)) = f_i(v_i | pa_i).
```

All denominators are positive. The intervention target need not be measured:
its value is known from the command, and all other variables can be measured
jointly. Thus every non-anchor CPT factor is recovered as a function of a full
assignment, without knowing its parents or graph in advance.

Remove the recovered factors from the observational law:

```
H(x,y,z) = P(x,y,z) / product_(i in Z) f_i(v_i | pa_i)
         = f_X(x | pa_X) * f_Y(y | pa_Y).
```

X is an ancestor of Y in an acyclic model, so Y is not a parent of X. For fixed
x,z, summing over y therefore yields

```
f_X(x | pa_X) = sum_y H(x,y,z)
f_Y(y | pa_Y) = H(x,y,z) / sum_y H(x,y,z).
```

Every factor is now recovered from legal experimental distributions. A variable
j is a parent of i precisely when changing j while holding every other argument
fixed changes recovered `f_i` somewhere. Causal minimality makes this criterion
agree with the declared graph. Consequently the DAG and all CPTs, and hence the
existing five task targets (including the certified counterfactual identified
region defined over that DAG/CPT world), are uniquely determined.

This proof uses only the legal single-target interventions. It does not assume
simultaneous interventions, observational faithfulness, knowledge of the
hidden topological order, or a finite candidate-world set. It also explains why
merely allowing more measurements cannot repair a missing intervention.

## Budget and public information

For the same task seed, keep the old independently drawn budget width `M0` and
exponent `e`. Store `B = M0 * 2**e` explicitly. The measurement width becomes n,
but B does not become `n * 2**e`. Each returned scalar is still charged by the
existing runtime. The 128-cell response bound still permits a joint assignment
as a batch of one and does not erase population information.

The prompt must disclose causal sufficiency, positivity, minimality and the
source-ancestor-of-outcome prior. It must not disclose the graph, parameters,
exact answer, sampled budget width or proof internals. Both rendering and
execution resolve the same explicit scalar budget.

## Acceptance boundary

Opening the permissions is necessary but is not the only check. An accepted
generated task must pass the actual-world contract validator. A zero-probability
CPT entry or an inactive declared edge fails closed; admitting it would break
this proof. Legacy manually assembled fixtures can retain their explicitly
specified permissions, but they cannot claim this contract.

Strictly positive finite distributions overlap. Therefore no finite observation
budget gives a uniform zero-error estimator over this entire world family.
Very weak dependencies and large joint state spaces can still make tasks
statistically or computationally difficult. Those limits must be distinguished
from the previous exact observational equivalence defect. Benchmark task errors
and matched-budget baselines remain required; checkpoint trust cannot be
inferred from raw training reward alone.
