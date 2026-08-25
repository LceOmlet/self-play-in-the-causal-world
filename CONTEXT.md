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

**Compact Joint Histogram**:
A lossless summary of the requested-measure empirical joint counts that declares the measured columns once and represents every observed state-index tuple with its count. Omitted joint assignments still have count zero, so compactness does not remove dependence information.
_Avoid_: raw sample rows, marginal summary, pairwise summary, lossy sketch

**Feedback Cell Ceiling**:
The maximum of 128 observed joint assignments that one batch feedback may contain. It bounds one model-visible payload without truncating an admitted batch or limiting how many atomic samples may be aggregated into each joint count.
_Avoid_: sample cap, observation budget, lossy truncation, top-k cells

**Terminal-Quality Reward**:
The scalar used by the current RL milestone is a continuous measure of the quality of the structured terminal answer, preserving the ordering carried by numerical error, interval distance, regret, and partial structural correctness. Experimental observations consumed, query count, turn count, and token count remain separate diagnostics and do not alter this reward; binary success indicators remain evaluation diagnostics rather than training rewards.
_Avoid_: binary-only reward, cost-adjusted reward, experimental-efficiency reward, trajectory-length reward

**ATE Terminal Quality**:
The terminal quality of a reported total effect is one minus half of its absolute error, mapping the complete legal error range to `[0, 1]`. Every equal reduction in absolute error receives the same reward improvement.
_Avoid_: squared-error reward, thresholded effect accuracy

**Individual-Counterfactual Terminal Quality**:
The terminal quality of a reported individual counterfactual probability is one minus its absolute distance to the hidden sharp compatible interval. Every value inside the interval receives full quality, and distances are measured on the original probability scale without interval-width normalization.
_Avoid_: preferred interval midpoint, hidden-SCM point target, interval-width-normalized reward

**Experimental-Decision Terminal Quality**:
The terminal quality of a deployment intervention is one minus its exact probability regret under the seed's minimize or maximize objective. Every intervention state attaining the optimal outcome probability receives full quality, including noncanonical tied optima.
_Avoid_: canonical-action accuracy, decision-gap-normalized reward, experiment-cost-adjusted regret

**Backdoor Task Success**:
A backdoor-adjustment answer succeeds only when it reports all and only the inclusion-minimal valid adjustment sets. Missing a set, adding a set, or changing the membership of any set is not a successful answer.
_Avoid_: any-valid-set success, partial-family success

**Backdoor Terminal Quality**:
The training quality of a backdoor-adjustment answer is a soft family F1 obtained by maximum-weight one-to-one matching between predicted and true adjustment sets, using set-level Dice overlap as the matching weight. It reaches one exactly when Backdoor Task Success holds; exact family match remains the task-success metric.
_Avoid_: atomic-family F1 as training reward, exact-match-only training reward

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
The policy signal obtained by subtracting the common-randomness group's mean terminal quality from each rollout's terminal quality without dividing by the group's standard deviation. It retains the magnitude of quality differences on the shared `[0, 1]` scale while remaining relative within each task instance.
_Avoid_: raw terminal quality as advantage, group-standardized advantage, cross-task reward normalization

**Individual Counterfactual Probability**:
The probability of a named outcome for the same individual under a counterfactual assigned treatment, conditional on that individual's assigned factual treatment and observed factual outcome. The model returns one scalar. The verifier accepts it only when it lies in the hidden sharp interval over every finite nonparametric mechanism completion compatible with the complete DAG, all CPT rows, consistency, mechanism replacement, and the other public CPT-World assumptions. No completion is selected or assigned a prior. Endpoint interventional marginals provide only a Fréchet outer interval in the generic case.
_Avoid_: population transition query, model-returned interval, hidden-SCM point label, preferred functional family

**Objective Computability**:
The evaluator can derive one exact task answer from the sealed CPT-World semantics. This does not imply that the visible experiment surface determines that answer.
_Avoid_: answerability

**Family-Relative Answerability**:
Before per-seed K and M are sampled, every world in a declared candidate family with the same query-specific full experiment laws has at least one common valid terminal answer.
_Avoid_: a truth exists, finite-sample accuracy, universal identifiability over undeclared worlds

**Structural Discovery Task**:
A task whose terminal answer is a causal structure object rather than a numerical effect or deployment action. The current structural discovery tasks are minimal backdoor adjustment sets and mediator paths.
_Avoid_: numerical target task, graph dump
