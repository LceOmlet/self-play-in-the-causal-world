# CPT-World Active Causal Tasks

This context defines the task-level language for querying one hidden finite CPT world through selected batched experiments.

## Language

**Manipulability Width (K)**:
The number of visible non-anchor variables that are legal hard-intervention targets in one task seed.
_Avoid_: intervention budget, action budget

**Observation Bandwidth (M)**:
The seed-fixed maximum number of readable variables whose joint values may be requested in one experiment batch.
_Avoid_: context window, observation budget

**Passive Observation**:
A paid batch experiment that samples the hidden world's natural distribution without applying a hard intervention and returns only the requested joint readout.
_Avoid_: free initial data, automatic full joint

**Atomic Sample**:
One IID world draw consumed by either an intervention batch or a passive-observation batch, independent of how many variables are read from that draw.
_Avoid_: measured cell, variable observation

**Sparse Batch Feedback**:
The selected-measure joint count map contains only assignments observed at least once in the current batch. Every omitted assignment has count zero. This changes only the wire representation, not the sampled data or its probability law.
_Avoid_: truncated support, approximate feedback, missing observation

**Task Family**:
The query types admitted by the main seed-generation pipeline. Registration or fixture-only execution does not by itself make a query type part of the task family.
_Avoid_: query registry, fixture collection

**Counterfactual Transition Bounds**:
The sharp interval for the probability that the outcome is in a named target state under one treatment value and not in that state under the baseline value, leaving the CPT-unspecified cross-world coupling free.
_Avoid_: hidden-SCM answer, point counterfactual

**Counterfactual Answer Mode**:
One of two separately rendered terminal contracts for the same world, query, K, and M: `sharp_interval` asks for both exact endpoints; `compatible_value` asks for one value inside that interval and reports its continuous distance to the interval. Compatibility does not imply point identification.
_Avoid_: model-selected answer format, second counterfactual sampler, scalarized reward

**Objective Computability**:
The evaluator can derive one exact task answer from the sealed CPT-World semantics. This does not imply that the visible experiment surface determines that answer.
_Avoid_: answerability

**Family-Relative Answerability**:
Before per-seed K and M are sampled, every world in a declared candidate family with the same query-specific full experiment laws has at least one common valid terminal answer.
_Avoid_: a truth exists, finite-sample accuracy, universal identifiability over undeclared worlds

**Structural Discovery Task**:
A task whose terminal answer is a causal structure object rather than a numerical effect or deployment action. The current structural discovery tasks are minimal backdoor adjustment sets and mediator paths.
_Avoid_: numerical target task, graph dump
