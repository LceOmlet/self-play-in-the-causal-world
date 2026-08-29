---
status: accepted
implementation: implemented
supersedes: 0031-calibrate-numeric-rewards-against-observational-shortcuts
---

# Use continuous fixed-budget shortcut calibration

Numerical terminal quality uses `(B+s)/(B+s+E)` with
`s=1/sqrt(2048)`. This removes the zero-separation threshold while retaining a
fixed, public accuracy scale derived from the experiment budget. Best
intervention returns one deployment state and uses raw probability regret for
both model error and observational-shortcut error. ATE retains full-vector TV
error, and individual counterfactual ROI retains certified endpoint MAE. No
task family applies a second training-time reward transformation.
