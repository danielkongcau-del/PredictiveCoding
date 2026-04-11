# Stage 05 V2 Budget-Push Validation

## Protocol
- dataset: `digits`
- seeds: `[0, 1, 2]`
- shared batch size: `128`
- shared shuffle_batches: `True`
- reference budget epochs: `192`
- stronger budget epochs: `384`

## Decision
- `stage05_v2_budget_push_materially_improves_configured_step_mechanism`: `True`
- `stage05_v2_budget_push_materially_improves_report_only_accuracy`: `True`
- stronger budget still hits final training boundary on all seeds: `True`
- recommended next move: `continue_with_budget`
- rationale: `The stronger Stage 05 v2 budget still selects the final training epoch on every seed, so the same-family budget line still looks boundary-limited.`

## Supports
- The stronger-budget Stage 05 v2 candidate materially improves configured-step mechanism magnitude over the 192-epoch reference.
- The stronger-budget Stage 05 v2 candidate materially improves report-only accuracy over the 192-epoch reference.
- The stronger budget still hits the final training boundary on every seed.
- 192-epoch reference configured-step validation energy delta vs identity mean: -0.001024608932.
- 384-epoch candidate configured-step validation energy delta vs identity mean: -0.001944251311.
- 192-epoch reference validation/test accuracy means: 0.724691 / 0.718519.
- 384-epoch candidate validation/test accuracy means: 0.780247 / 0.785185.

## Does Not Support
- This validation does not reopen Stage 04 package-internal work.
- This validation does not change the Stage 05 v2 transport family, residual branch structure, corrected residual identity contract, or selection rule.
- This validation does not claim that Stage 05 replaces the frozen Stage 04 bridge result on main.
- This validation does not yet justify opening a true Stage 05 v3 mechanism charter.
