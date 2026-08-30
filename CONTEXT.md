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

**Base-and-Score CPT Parameterization**:
For a non-root node, draw one state-exchangeable base distribution \(b_Y\) and one row- and column-centred categorical score table \(D_Y(c,y)\). ET-V2 maps the base and score to every conditional row by RMS-normalized exponential tilting. Roots use \(b_Y\) directly. Parent configurations and child states have no privileged reference value.
_Avoid_: independent conditional-row priors, a privileged reference parent state, describing the score as an additive probability displacement

**State-Exchangeable Base Distribution**:
For a \(d\)-state node, draw \(b_Y\sim\mathrm{Dirichlet}(1,\ldots,1)\). This is uniform on the \(d\)-state probability simplex, invariant to state relabelling, and reduces to \(p\sim\mathrm{Uniform}(0,1)\) for a binary node.
_Avoid_: entropy-targeted hierarchy, preferred state, radial base-distribution construction

**Parent-Subset-Exchangeable Score Energy**:
For every nonempty subset of a node's parents, draw an independent isotropic unit direction in that subset's pure categorical functional-ANOVA subspace. Draw their squared-energy shares jointly from \(\mathrm{Dirichlet}(1,\ldots,1)\), then combine directions using the square roots of those shares. Every parent subset has the same expected energy; shares remain random and are almost never equal within one world. This balances actual parent interactions rather than interaction orders or subspace dimensions, preserves parent/state relabelling symmetry, and keeps all interaction subsets active without answer filtering.
_Avoid_: fixed equal shares, equal expected energy per interaction order, dimension-proportional energy, interaction sparsity, claiming that realized shares are equal

**Unit-Expected ET-V2 Score Strength**:
After ET-V2 normalizes the combined score table to unit elementwise RMS, draw its nonnegative amplitude uniformly on \([0,\sqrt{3}]\). The resulting squared log-score perturbation has expectation one because \(\mathbb E[s_Y^2]=1\). The bound is derived from the conserved score-energy unit; it is not tuned from task answers. The distribution remains bounded, includes weak through strong mechanisms, and preserves strictly positive generated CPT rows.
_Avoid_: uniform amplitude on \([0,1]\), unbounded strength tails, calibrating strength from final ATE or decision gaps, claiming that probability-scale effects are uniform

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

**Compact Joint Histogram**:
A lossless summary of the requested-measure empirical joint counts that declares the measured columns once and represents every observed state-index tuple with its count. Omitted joint assignments still have count zero, so compactness does not remove dependence information.
_Avoid_: raw sample rows, marginal summary, pairwise summary, lossy sketch

**Feedback Cell Ceiling**:
The maximum of 128 observed joint assignments that one batch feedback may contain. It bounds one model-visible payload without truncating an admitted batch or limiting how many atomic samples may be aggregated into each joint count.
_Avoid_: sample cap, observation budget, lossy truncation, top-k cells

**Terminal-Quality Reward**:
The scalar used by the current RL milestone is a continuous measure of the quality of the structured terminal answer, preserving the ordering carried by numerical error, identified-region endpoint error, normalized decision regret, and partial structural correctness. Experimental observations consumed, query count, turn count, and token count remain separate diagnostics and do not alter this reward; binary success indicators remain evaluation diagnostics rather than training rewards.
_Avoid_: binary-only reward, cost-adjusted reward, experimental-efficiency reward, trajectory-length reward

**ATE Terminal Quality**:
The terminal quality of the complete categorical total-effect vector is one minus one quarter of its L1 error. It uses every outcome-state component, reduces exactly to the previous scalar definition for binary outcomes, and maps the complete legal vector-error range to `[0, 1]`.
_Avoid_: single-outcome projection, squared-error reward, thresholded effect accuracy

**Individual-Counterfactual Terminal Quality**:
The terminal quality of a reported individual-counterfactual identified region is one minus the mean absolute error of its lower and upper endpoints. Exact truth compares directly with the sharp endpoints; epsilon-sharp truth measures distance to each endpoint's certified range.
_Avoid_: compatible-point terminal, preferred interval midpoint, hidden-SCM point target, confidence field

**Experimental-Decision Terminal Quality**:
For the fixed deployment variable, let the candidate-state outcome probabilities span `[p_min, p_max]`. Terminal quality is one minus exact probability regret divided by `p_max - p_min`; if the span is zero, every state is tied and receives one. Raw regret and optimal-action accuracy remain separate diagnostics.
_Avoid_: canonical-action-only reward, unnormalized-regret reward, best-second-gap normalization, experiment-cost-adjusted regret

**Backdoor Task Success**:
A backdoor-adjustment answer succeeds when the model's one complete submitted covariate set standardizes the observational law to the true intervention distribution for every treatment state. The score uses every submitted variable, so conditioning on a mediator or collider can worsen the answer while harmless redundant covariates remain harmless.
_Avoid_: minimal-set-family enumeration, graph-overlap success, filtering the submission through a true set

**Backdoor Terminal Quality**:
Let `E` be the mean total-variation distance between the submitted set's standardized outcome distributions and the hard-do outcome distributions over all treatment states. Let `B` be the same error for the empty set. Exact adjustment receives one; when `B>0`, quality is `B/(B+E)`; when `B=0`, zero adjustment error receives one and positive error receives zero. This prioritizes variables according to their effect on the target adjustment error without changing the world distribution.
_Avoid_: structural F1, set-family matching, path-strength thresholds, multiplying structural and effect scores

**Mediator Task Success**:
A mediator answer succeeds only when both its mediator-variable set and its consecutive directed-path-edge set exactly match their truths.
_Avoid_: mediator-only success, order-only success, partial-structure success

**Mediator Terminal Quality**:
The training quality of a mediator answer is the equal-weight arithmetic mean of mediator-set F1 and consecutive-path-edge F1. The two components remain separate diagnostics, and their joint exact match remains the task-success metric.
_Avoid_: product-combined mediator reward, harmonic-combined mediator reward, pooled node-edge micro-F1

**Unfinished Terminal Quality**:
A model-caused trajectory that ends without a legal terminal answer has terminal quality zero. Recoverable protocol errors do not alter a later legal answer's quality, while environment or verifier failures produce no model-training sample.
_Avoid_: negative protocol penalty, dropping model-caused unfinished trajectories, scoring infrastructure failures

**Task Family**:
The query types admitted by the main seed-generation pipeline. Registration or fixture-only execution does not by itself make a query type part of the task family.
_Avoid_: query registry, fixture collection

**Uniform Task-Family Training Mixture**:
The current RL training distribution assigns equal mass to the five admitted task families, so each family contributes one fifth of episodes. This balancing does not alter a family's node-count, role, K, M, state-anchor, or mechanism distribution.
_Avoid_: uniform query-registry mixture, difficulty-balanced mixture, adaptive curriculum

**RL Training Closure Smoke**:
A bounded run that succeeds only when a real policy completes an interactive task, receives the owner-produced terminal reward, updates trainable parameters, and saves and resumes a checkpoint. It validates the training integration rather than task performance or learning quality.
_Avoid_: inference smoke, environment smoke, performance experiment

**Common-Randomness GRPO Group**:
A comparison group whose policy rollouts share one hidden world, task, interaction surface, budget, and action-keyed outcome tape while sampling their policy decisions independently. Different groups use different task and tape seeds, so shared randomness removes within-group environment noise without making all trajectories identical.
_Avoid_: shared transcript, independently resampled group tasks, identical rollout copies

**Unscaled Group-Relative Advantage**:
The policy signal obtained by subtracting the common-randomness group's mean terminal quality from each rollout's terminal quality without dividing by the group's standard deviation. Every task's owner-produced reward enters GRPO unchanged.
_Avoid_: trainer-only reward shaping, group-standardized advantage, cross-task reward normalization

**Individual Counterfactual Probability**:
The probability of a named outcome for the same individual under a counterfactual assigned treatment, conditional on that individual's assigned factual treatment and observed factual outcome. The model returns the lower and upper endpoints of its sharp identified region over every finite nonparametric mechanism completion compatible with the complete DAG, all CPT rows, consistency, mechanism replacement, and the other public CPT-World assumptions. No completion is selected or assigned a prior. Endpoint interventional marginals provide only a Fréchet outer interval in the generic case.
_Avoid_: population transition query, compatible-point terminal, hidden-SCM point label, preferred functional family

**Objective Computability**:
The evaluator can derive one exact task answer from the sealed CPT-World semantics. This does not imply that the visible experiment surface determines that answer.
_Avoid_: answerability

**Family-Relative Answerability**:
Before per-seed K and M are sampled, every world in a declared candidate family with the same query-specific full experiment laws has at least one common valid terminal answer.
_Avoid_: a truth exists, finite-sample accuracy, universal identifiability over undeclared worlds

**Structural Discovery Task**:
A task whose terminal answer is a causal structure object rather than a numerical effect or deployment action. Mediator-path recovery is the current structural discovery task; backdoor adjustment is scored by the causal distribution recovered by the submitted set.
_Avoid_: numerical target task, graph dump
