Fixed-task interpretation and matched official evaluation preparation.

`audit_fixed_tasks.py` uses all 25 original validation tasks, verifies the dataset,
admission record and frozen truth hashes, and submits constant and population
observation answers to the production environment/scorer. Population observation
uses hidden probabilities; it is not model performance or a finite-budget solver.
`fixed-task-baselines.json` contains every answer and task metric.

`check_backdoor_advantages.py` invokes the actual pinned upstream GRPO function
on four real retained backdoor groups. For completed, unpenalized answers,
`q=1-d/14` gives `A=-(d-mean(d))/(std(d)+14*epsilon)`. Its apparent score compression
does not compress normalized advantages by14. FP32 comparison with negative
distance changes advantages by at most4.02e-5 in these groups. The incomplete
step17 group is excluded, not assigned an invented task error.

For mean raw quality Q and valid-completion probability p, the existing scoring
definition only guarantees `max(0,14Q-13) <= p <= Q`. Thus Q=0.9 supplies no
positive lower bound on valid-completion probability. This explains the need
for strict metrics without treating affine scaling as a suppressed gradient.

`prepare_base_validation.py` resolves and verifies the official base `val_only`
command against the actual running configuration. It changes only initialization,
validation-only control and run/output paths. It does not launch GPU work,
implement a validator/trainer, or claim a completed baseline. The intended GPU
run must still verify source/data identities and exclusive GPU availability.

`evidence.tar.gz` contains70 members (1,317,170 bytes), SHA-256
`390e30c0f23894ae25658a49b28d7001be8e345fb467cb5b7cd6118c21e73bd2`.
It freezes raw source/configuration checks, both versions of the fixed-task audit,
all original validation truth files and admission metadata, the actual advantage
inputs/outputs and the through21 training snapshot. No trained model weights are
in the archive. Per-member hashes are in `evidence.sha256.json`.

At archive time, main PID400557 was still live with checkpoint24 published.
The full step25 validation and matched original-base GPU results were pending.
See `docs/heldout-learning-and-reward-scale-20260908.md` for the derivation,
metric interpretation, remaining execution work and scope limitations.
