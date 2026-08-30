---
status: accepted
implementation: implemented
supersedes: 0033-use-continuous-fixed-budget-shortcut-calibration
restores: 0030-normalize-decision-regret-by-candidate-span
---

# Use normalized regret for best intervention

Best-intervention terminal quality uses the returned state's exact probability
regret divided by the complete candidate probability span. The observational
shortcut regret is divided by the same span before shortcut calibration. Raw
probability regret remains a diagnostic only.

When the span is zero, every candidate state is tied and both normalized
regrets are zero. The terminal answer remains one selected deployment state.
