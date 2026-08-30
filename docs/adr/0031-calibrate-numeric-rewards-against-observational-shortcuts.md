---
status: superseded
implementation: implemented
superseded_by: 0033-use-continuous-fixed-budget-shortcut-calibration
---

# Calibrate numerical rewards against observational plug-in shortcuts

ATE, individual counterfactual ROI, and best intervention use the task-local
observational plug-in error as their accuracy unit.  For model error `E` and
observational shortcut error `B`, quality is `B / (B + E)` when `B>0`.

Best intervention now returns the complete candidate causal-value profile.
Scoring its pairwise gaps preserves all deployment comparisons and distinguishes
causal from observational profiles even when both profiles select the same
terminal action.  The selected action and normalized regret are derived
diagnostics.

The numerical rewards enter GRPO without the previous ceiling-sensitive
transform because the observational baseline is already fixed at one half.
World generation, role sampling, K/M, and experiment budgets are unchanged.
