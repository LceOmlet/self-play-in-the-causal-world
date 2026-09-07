# Do not standardize GRPO rewards within groups

Historical decision: the user's subsequent explicit request for standard DAPO
supersedes this setting for the official verl entry point. That entry point uses
the official group-standardized advantage. See
[its objective and limitations](../official-dapo-integration-20260906.md).

GRPO advantages use each rollout's terminal quality minus its common-randomness group mean, with `scale_rewards="none"`; they are not divided by the group's reward standard deviation. All five terminal-quality functions already share the `[0, 1]` scale, and retaining reward-gap magnitude preserves the distinction between small and large quality improvements. Standard group scaling was rejected because it makes proportionally identical reward patterns produce nearly identical advantages regardless of their absolute quality gaps and can introduce task-difficulty bias.
