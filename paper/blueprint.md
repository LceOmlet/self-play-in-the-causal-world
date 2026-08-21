# CPT-World paper blueprint

This file is a drafting aid. It follows the Supervisor-Skills benchmark-paper
logic: research gap, construction pipeline, evaluation framework, and empirical
validation.

## One-sentence thesis

CPT-World evaluates causal reasoning as a closed loop of choosing an experiment,
reading its selected evidence, and returning an exactly verifiable causal object
inside a hidden finite world.

## Introduction logic chain

1. **Motivating capability.** A causal reasoner should decide what evidence to
   collect before it answers.
2. **Current evaluation gap.** Static causal questions evaluate inference from a
   supplied description; broad scientific environments mix causal reasoning with
   navigation, language grounding, and domain scripts.
3. **Core problem.** Build an executable causal world that keeps the mechanism
   hidden, exposes a controlled experiment surface, and retains exact truth.
4. **Key challenge.** The same generator must support heterogeneous query types,
   selective observations, reproducible stochastic feedback, and counterfactual
   answers without an evaluator-selected cross-world coupling.
5. **Solution.** Finite CPT worlds provide an exact world law; K and M define the
   interaction surface; a common runtime executes observations and hard
   interventions; exact query owners verify terminal answers; compatible-SCM
   bounds define counterfactual legality.
6. **Evidence.** Five task modes run end to end, exact acceleration matches the
   full-joint oracle, action-keyed sampling passes invariance tests, and sparse
   15-node benchmarks obtain large speedups.

## Main-text structure

### 1. Introduction

- Running example: opaque variables, readonly ATE endpoints, one indirect
  intervention, selected joint counts, terminal numeric answer.
- Explain why this tests experimental causal reasoning.
- Explain the evaluator preference created by selecting one SCM coupling for a
  cross-world point answer.
- State three contributions in prose.

### 2. From causal questions to causal experiments

- Position CLadder and CauGym as static inference from supplied causal evidence.
- Position ABCI, DiffCBED, targeted indirect design as experiment-design methods.
- Position DiscoveryWorld, CausalGame, and CausaLab as interactive scientific or causal agency.
- Locate CPT-World at the intersection: LLM interface, hidden mechanism,
  controlled causal experiments, model-selected readouts, compatible-set
  counterfactual semantics, and exact replay.

### 3. The CPT-World episode

- Finite world `W=(G,Theta)`.
- Action `a=(do/observe, R, b)`.
- Feedback `C_R ~ Multinomial(b, P_W(R | do(...)))`.
- K and M.
- Truth-free renderer and structured terminal answer.

### 4. One world engine, five causal tasks

- Table: ATE, counterfactual bounds/value, experimental decision, backdoor sets,
  mediator set/order.
- Explain query-action separation per task.
- Explain compatible-SCM interval with the 0.2/0.8 example.

### 5. Exact, reproducible execution

- One probability semantics.
- Exact variable elimination.
- Ancestral batch sampling.
- Action-keyed outcome tape.
- Put derivations and pseudocode in appendix.

### 6. Validation

- Semantic equivalence and invariance tests.
- Five-mode prompt-to-verdict demonstration.
- Runtime table at 10, 14, 15 binary nodes.
- State what each result certifies.

### 7. Conclusion

- Restate experimental causal reasoning as the evaluation object.
- Connect exact verifier and compatible counterfactual sets to training/evaluation.

## Figures and tables

1. **Figure 1, motivated example and system loop.** Hidden world in the center;
   model sees an opaque prompt, chooses observation/intervention and readout,
   receives counts, and returns an answer to the exact verifier. A compact lower
   strip shows point-SCM coupling versus compatible interval.
2. **Table 1.** Benchmark-level comparison across world evidence, active probes,
   returned evidence, scored objects, counterfactual semantics, and verification.
3. **Table 2.** Five task modes and terminal objects.
4. **Table 3.** Exactness and invariance validation.
5. **Figure 2.** Log-scale acceleration plot for exact marginal and batch paths.

## Appendix structure

- A: Formal world and hard-do semantics.
- B: Exact query truth, including counterfactual compatible sets.
- C: Variable elimination and ancestral tape execution.
- D: Rendering and strict JSON protocol.
- E: Validation fixtures and benchmark setup.
- F: Full prompt example.
