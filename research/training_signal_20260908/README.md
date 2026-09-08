Actual training signals and stronger observation-rank baselines.

`audit_training_signal.py` verifies the four frozen source prefixes, all 29
completed official updates/116 retained trajectories, and the official sampler's
33 consumed tasks. Four tasks were filtered. The pinned official scalar GRPO
function is called on CPU; this is not GPU loss replay or another trainer.

All four successful strong-reversal training outputs through29 were read against
their events. Their final calculations use unadjusted observations, including the
trajectory with 33 successful interventions. Correct final actions do not certify
the causal method described in their text.

`audit_observation_rank.py` replays the original 50 diagnostic worlds/two tapes and
all 300 historical passive checkpoints. Rank2 uses only public objective/domain
and natural counts, then submits through production scoring. Pure observation
rank2 gets 57/100 overall and57/80 strong; the original active program gets70/100
and51/80. The prior 23%→70% comparison omitted this stronger noncausal baseline.
All seven previous environment/truth/scoring fingerprints and NumPy match.

These cohorts were already analyzed; no new-heldout or significance claim is
made. The exact submitted-world population-rank table is a separate diagnostic,
not a finite-observation policy result. No generator, reward, or training setting
was changed. See `docs/observation-rank-bias-and-training-signal-20260908.md`.

`plot_rank_comparison.py` and the existing `plot_trusted_progress.py` generate the
standalone figures using the existing local Matplotlib runtime. All three PNGs
were visually inspected; SVG/PDF, source hashes and input records are retained.

`training-signal-and-rank-evidence.tar.gz`:119 members,2,050,187 bytes, SHA-256
`7d60b9e9ae90737baa053cb787b9d69856ea4cc38c4dd6cf5b4199932e657e34`.
Original world archives are referenced by their existing repository hashes.
The first rank run failed serializing Fraction diagnostics; v2 completed all100.
Earlier extraction failed expecting terminal feedback in the decoded output;
the corrected reader excludes the terminal message. Both failures are preserved.
