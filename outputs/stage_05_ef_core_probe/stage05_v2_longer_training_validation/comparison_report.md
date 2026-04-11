# Stage 05 V2 Longer-Training Validation

## Protocol
- dataset: `digits`
- seeds: `[0, 1, 2]`
- shared batch size: `128`
- shared shuffle_batches: `True`
- current budget epochs: `12`
- longer budget epochs: `24`

## Decision
- `stage05_v2_longer_training_materially_improves_configured_step_mechanism`: `True`
- `stage05_v2_longer_training_materially_improves_report_only_accuracy`: `True`
- longer budget still hits final training boundary on all seeds: `True`
- recommended next move: `continue_with_budget`
- rationale: `The stronger Stage 05 v2 budget still selects the final training epoch on every seed, so the budget question is not yet closed.`

## Supports
- The longer-training Stage 05 v2 candidate materially improves configured-step mechanism magnitude over the current budget.
- The longer-training Stage 05 v2 candidate materially improves report-only accuracy over the current budget.
- The stronger budget still hits the final training boundary on every seed.
- Current-budget configured-step validation energy delta vs identity mean: -0.000176362138.
- Longer-budget configured-step validation energy delta vs identity mean: -0.000205556461.
- Current-budget validation/test accuracy means: 0.270370 / 0.272840.
- Longer-budget validation/test accuracy means: 0.417284 / 0.400000.

## Does Not Support
- This validation does not reopen Stage 04 package-internal work.
- This validation does not change the Stage 05 v2 transport family, objective family, or selection rule.
- This validation does not claim that Stage 05 replaces the frozen Stage 04 bridge result on main.
- This validation does not yet justify opening a true Stage 05 v3 mechanism charter.
