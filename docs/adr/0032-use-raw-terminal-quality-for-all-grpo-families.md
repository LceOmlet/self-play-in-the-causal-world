---
status: accepted
implementation: implemented
supersedes: 0023-use-ceiling-sensitive-advantage-utility
---

# Use raw terminal quality for every GRPO task family

All five task families pass their environment-owned terminal quality unchanged
to GRPO. No task-specific logarithmic or ceiling-sensitive transformation is
applied. This keeps the optimized quantity identical to the reported terminal
reward and removes the unused reward-utility epsilon from the training entry
point.

The environment-to-reward-adapter identity still holds in the official verl
DAPO integration. The statement above that the optimized quantity is identical
to reported terminal reward does not describe the full DAPO objective: the
official recipe adds its overlong penalty, group normalization and dynamic
filtering. Raw quality and shaped training reward must be reported separately.
