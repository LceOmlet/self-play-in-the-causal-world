# CPT-World Active Causal Tasks

This context defines the task-level language for querying one hidden finite CPT world through selected batched experiments.

## Language

**Manipulability Width (K)**:
The number of visible non-anchor variables that are legal hard-intervention targets in one task seed.
_Avoid_: intervention budget, action budget

**Maximum Parent Count ($P_{max}$)**:
The fixed upper bound of three direct parents for any generated node; it controls local interaction order and CPT table width without limiting causal path length.
_Avoid_: edge probability, graph size, intervention width

**Parent-Count-Uniform Structure Prior**:
After uniformly randomizing the node order, each node draws its number of parents uniformly from zero through the smaller of three and its predecessor count, then draws that many parents uniformly from its predecessors. The induced DAG distribution is intentionally not uniform over labeled DAGs; it gives equal prior mass to the available local interaction orders while bounding CPT width.
_Avoid_: uniform DAG prior, independent-edge prior

**Task Structural Eligibility**:
The graph-level condition required for a sampled world to instantiate a query type, such as the existence of an indirect directed path for a mediator task. Eligibility may condition the structure distribution of that task family; it must not inspect effect magnitude, answerability diagnostics, model performance, or intended difficulty.
_Avoid_: answerability filter, difficulty filter, nonzero-effect filter

**Size-Preserving Eligibility Sampling**:
An episode first draws its node count uniformly from 3 through 15 and keeps that count fixed while resampling structures until the selected task is structurally eligible. This preserves the declared node-count distribution instead of favoring larger graphs that satisfy eligibility more often.
_Avoid_: resampling node count after eligibility failure, accepted-size distribution

**Node-Cardinality Prior**:
Each node independently draws its number of states uniformly from two through five. Together with the three-parent bound, this limits one conditional table to at most 125 rows and 625 probability entries while giving every fixed-size cardinality vector equal probability.
_Avoid_: binary-only world, task-conditioned cardinality

**Base-and-Joint-Effect CPT Parameterization**:
For a node with \(R\) joint parent configurations and \(d\) child states, represent its CPT as \(P(Y\mid c)=b_Y+\Delta_Y(c)\). Every effect row sums to zero, and every child-state column averages to zero across parent configurations. Roots use \(b_Y\) directly. Every legal CPT has a unique representation of this form, so this is a parameterization rather than a restricted mechanism family.
_Avoid_: independent conditional-row priors, a privileged reference parent state, separate main/pair/triple interaction families

**State-Exchangeable Base Distribution**:
For a \(d\)-state node, draw \(b_Y\sim\mathrm{Dirichlet}(1,\ldots,1)\). This is uniform on the \(d\)-state probability simplex, invariant to state relabelling, and reduces to \(p\sim\mathrm{Uniform}(0,1)\) for a binary node.
_Avoid_: entropy-targeted hierarchy, preferred state, radial base-distribution construction

**Radially Uniform Joint-Effect Strength**:
Draw an isotropic Gaussian table, project it onto the subspace whose rows sum to zero and whose columns average to zero, and normalize it to obtain \(D_Y\). This samples directions isotropically in the complete legal joint-effect space and is invariant to parent-configuration and child-state relabelling. Find the largest positive scale \(a_{\max}\) for which \(b_Y+aD_Y(c)\) remains a probability distribution for every parent configuration, draw \(s_Y\sim\mathrm{Uniform}(0,1)\), and set \(\Delta_Y=s_Ya_{\max}D_Y\). The sampler does not separately balance interaction orders: aggregate higher-order variation grows naturally with the number and cardinalities of the parents because those interaction subspaces have more dimensions.
_Avoid_: treating a multistate effect as one signed scalar, separate strength parameters by interaction order, conditioning effects on task answers, claiming that every individual edge effect or interaction order is uniformly distributed

**Truth-Unfiltered Mechanism Sampling**:
Once a world has a legal CPT and the selected task is structurally eligible, its mechanisms are retained without inspecting the numerical task answer. Exact zero effects and tied optimal actions remain part of the task distribution and are measured as distribution statistics.
_Avoid_: nonzero-effect gate, decision-gap gate, answer-conditioned sampling

**Continuous CPT Runtime Semantics**:
Generated CPT parameters, probability inference, and intervention laws use `float64`. CPT finalization rejects nonfinite values, rejects probabilities below \(-10^{-12}\), maps values in \([-10^{-12},0)\) to zero, rejects rows whose sums differ from one by more than \(10^{-12}\), and renormalizes every accepted row. The finalized CPT is the single input to both interaction sampling and truth computation. This tolerance never rounds model answers or task rewards. Reproducibility comes from storing the realized WorldSpec, not from quantizing probabilities to bounded-denominator rationals. Exact `Fraction` arithmetic is reserved for fixed fixtures and reference tests. This numeric architecture is implemented throughout the generated-world path.
_Avoid_: rational-denominator parameter in the main sampler, exact equality on generated floating probabilities, describing fixed `Fraction` fixtures as the generated-world representation

**Causal Sufficiency**:
Every common cause belongs to the WorldSpec DAG; there is no graph-external hidden confounding, and the exogenous mechanisms of distinct nodes are independent. The graph and CPT therefore define every single-world hard-intervention law by mechanism replacement. Within one node, the cross-parent-configuration response coupling remains unrestricted and is bounded rather than selected for counterfactual queries. Hiding the world from the model does not mean omitting causal variables from the verifier.
_Avoid_: unrepresented common cause, observational CPT treated as interventional under latent confounding, fixed within-node response function

**State-Symmetric Estimand Anchors**:
After a task's variable anchors are chosen, ATE and counterfactual-transition instances draw an ordered pair of distinct intervention-variable states \((x_{\mathrm{ref}},x_{\mathrm{cmp}})\) uniformly from all such pairs and draw the target outcome state \(y^\star\) uniformly. Experimental-decision instances draw \(y^\star\) uniformly. These values define the terminal estimand only; they do not change K, M, legal experimental actions, batch sampling, or budget. Backdoor-adjustment and mediator-structure tasks have no state anchors.
_Avoid_: fixed state_0/state_1 target, calling the reference value a passive baseline, treating estimand anchors as episode actions

**Experimental-Decision Role Eligibility**:
An ordered pair \((D,Y)\) is structurally eligible when \(D\ne Y\) and \(D\) is an ancestor of \(Y\). The outcome need not be a leaf. After choosing \((D,Y)\), sample `minimize` or `maximize` independently with equal probability; the objective is not a variable-role assignment.
_Avoid_: leaf-outcome restriction, redundant parent-count condition, treating objective as a variable anchor

**Total-Effect Role Eligibility**:
An ordered pair \((X,Y)\) is structurally eligible for a total-effect task exactly when \(X\ne Y\) and a directed path from \(X\) to \(Y\) exists. Under the CPT-World DAG and mechanism-replacement semantics, no such path implies that every intervention value of \(X\) induces the same distribution of \(Y\). Observational association caused by a common ancestor does not create an interventional effect.
_Avoid_: observational association as causal eligibility, numerical nonzero-effect filter, arbitrary distinct-node pair

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
The sharp interval for the probability that the outcome is in a named target state under the comparison intervention and not in that state under the reference intervention. The interval optimizes over every finite nonparametric mechanism completion jointly compatible with the complete DAG, every CPT row, consistency, mechanism replacement, and the other public CPT-World assumptions. No completion is selected or assigned a prior. Coupling only the two endpoint interventional marginals gives an outer bound or a simple special case, not the general truth owner.
_Avoid_: hidden-SCM point answer, endpoint-marginal coupling as a universal oracle, preferred functional family

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
