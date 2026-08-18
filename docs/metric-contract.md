# Terminal diagnostic contract v1

Each valid terminal episode `i` has the reported vector

`(hat_tau_forward, hat_tau_reverse)`

and the corresponding generated truth

`(tau_forward, tau_reverse)`.

Exactly one truth coordinate is active and the other is exactly zero. The
generator records which coordinate is active; the evaluator does not infer it
from an effect threshold, a sign, an argmax, or the model's intervention choices.

For `N > 0` episodes, define component errors `e_i,f` and `e_i,r`. The frozen
diagnostics are

```text
vector_rmse = sqrt(sum_i (e_i,f^2 + e_i,r^2) / (2N))
active_mae  = sum_i abs(e_i,active(i)) / N
inactive_mae = sum_i abs(e_i,inactive(i)) / N
```

The denominator of `vector_rmse` is `2N`. It is therefore a component RMSE,
not the average vector norm and not the square root of a per-episode vector MSE
without the factor of two.

## Information-preservation rules

- Both effect values are mandatory and equally weighted.
- Metrics use unrounded validated values; presentation rounding happens later.
- No confidence is requested or consumed.
- No effect is thresholded, clipped, discretized, or replaced by a direction label.
- Negative and very small nonzero effects remain ordinary continuous values.
- Episode ID sets must match exactly between truth and prediction mappings.
- Empty input, non-finite values, out-of-range values, and broken truth
  certificates fail closed.

Protocol failures must not be silently removed to improve these diagnostics.
A benchmark runner must report valid-terminal coverage and settle failures under
its separately preregistered reward contract. This pure metrics API scores only
the canonical numeric terminal records it is given.

## Interpretation

- Low active and inactive MAE: finds real effects and suppresses spurious reverse effects.
- High active, low inactive MAE: conservative or unable to estimate the real effect.
- Low active, high inactive MAE: detects the real effect but hallucinates a reverse effect.
- High values on both: unreliable bidirectional numerical inference.

`vector_rmse` summarizes both failure modes and gives extra weight to large
numeric mistakes. None of these three metrics identifies sample efficiency.
