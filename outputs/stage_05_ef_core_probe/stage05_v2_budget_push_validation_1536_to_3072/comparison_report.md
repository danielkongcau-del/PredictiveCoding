# Stage 05 V2 Budget-Push Validation

## Protocol
- dataset: `digits`
- seeds: `[0, 1, 2]`
- shared batch size: `128`
- shared shuffle_batches: `True`
- reference budget epochs: `1536`
- stronger budget epochs: `3072`

## Decision
- `stage05_v2_budget_push_materially_improves_configured_step_mechanism`: `True`
- `stage05_v2_budget_push_materially_improves_report_only_accuracy`: `True`
- configured-step gain fraction vs reference: `0.24465571127137078`
- report accuracy gain vs reference: `{'val_accuracy_delta': 0.020987654320987703, 'test_accuracy_delta': 0.028395061728395232}`
- budget line still looks boundary-limited: `True`
- budget line should continue: `True`
- budget line should stop and open v3: `False`
- budget line interpretation: `boundary_limited_mechanism_prototype`
- stronger budget still hits final training boundary on all seeds: `True`
- recommended next move: `continue_with_budget`
- rationale: `The stronger Stage 05 v2 budget still materially improves configured-step mechanism and report-only accuracy while also selecting the final training epoch on every seed, so the same-family budget line still looks boundary-limited enough to continue.`

## Contextual Accuracy Note
- Stage 05 v2 stronger budget validation/test accuracy means: `0.908641975308642` / `0.9160493827160495`
- frozen Stage 04 validation/test accuracy: `0.811111111111111` / `0.8197530864197531`
- standalone digits_pc validation/test accuracy: `0.8444444444444444` / `0.9185185185185185`
- standalone digits_mlp validation/test accuracy: `0.9111111111111111` / `0.9481481481481482`
- note: `Diagnostic-only accuracy context: the stronger Stage 05 v2 budget is above the frozen Stage 04 bridge accuracy level, mixed relative to the standalone digits_pc baseline accuracy level, and below the standalone digits_mlp baseline accuracy level; this comparison informs budgeting only and does not change the Stage 05 mechanism-first gate.`

## Supports
- The stronger-budget Stage 05 v2 candidate materially improves configured-step mechanism magnitude over the 1536-epoch reference.
- The stronger-budget Stage 05 v2 candidate materially improves report-only accuracy over the 1536-epoch reference.
- The stronger budget still hits the final training boundary on every seed.
- 1536-epoch reference configured-step validation energy delta vs identity mean: -0.004980554368.
- 3072-epoch candidate configured-step validation energy delta vs identity mean: -0.006199075440.
- 1536-epoch reference validation/test accuracy means: 0.887654 / 0.887654.
- 3072-epoch candidate validation/test accuracy means: 0.908642 / 0.916049.
- Diagnostic-only context: Stage 05 v2 stronger-budget validation/test accuracy means remain at 0.908642 / 0.916049, versus frozen Stage 04 at 0.811111 / 0.819753, standalone digits_pc at 0.844444 / 0.918519, and standalone digits_mlp at 0.911111 / 0.948148.
- Diagnostic-only accuracy context: the stronger Stage 05 v2 budget is above the frozen Stage 04 bridge accuracy level, mixed relative to the standalone digits_pc baseline accuracy level, and below the standalone digits_mlp baseline accuracy level; this comparison informs budgeting only and does not change the Stage 05 mechanism-first gate.
- Interpretation: the Stage 05 v2 line is still behaving like a boundary-limited mechanism prototype.

## Does Not Support
- This validation does not reopen Stage 04 package-internal work.
- This validation does not change the Stage 05 v2 transport family, residual branch structure, corrected residual identity contract, or selection rule.
- This validation does not claim that Stage 05 replaces the frozen Stage 04 bridge result on main.
- This validation does not yet justify opening a true Stage 05 v3 mechanism charter.
