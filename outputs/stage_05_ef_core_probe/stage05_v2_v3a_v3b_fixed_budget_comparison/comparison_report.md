# Stage 05 v2 vs v3-A vs v3-B Trajectory Curriculum Comparison

## Protocol
- comparison scope: `fixed_budget_comparison`
- dataset: `digits`
- seeds: `[0, 1, 2]`
- shared batch size: `128`
- Stage 05 epochs: `1536`

## Decision
- `stage05_v3b_keeps_one_step_mechanism_positive`: `True`
- `stage05_v3b_materially_improves_configured_step_mechanism_vs_v3a`: `False`
- `stage05_v3b_shows_positive_gap_closure_signal_vs_v3a`: `False`
- gap_closure_decision: `no_material_gap_closure_signal_vs_v3a`
- recommended next move: `retain_v3a_as_active_reference`
- rationale: `The fixed-budget v3-B candidate does not yet show a strong enough configured-step advantage over v3-A to replace it as the active fixed-budget improvement reference.`

## Pairwise Deltas Vs V2
- configured-step validation energy delta vs identity delta: `-0.0006038434862037367`
- configured-step validation fixed-point residual delta vs identity delta: `-2.547410616476825e-06`

## Pairwise Deltas Vs V3-A
- configured-step validation energy delta vs identity delta: `-0.0001534021754053816`
- configured-step validation fixed-point residual delta vs identity delta: `-7.038069182450143e-07`

## Contextual 3072 Gap Closure
- v3-A configured-step energy gap closure: `0.36966230730808747`
- v3-B configured-step energy gap closure: `0.49555440633853987`
- v3-A configured-step residual gap closure: `0.17573315155025793`
- v3-B configured-step residual gap closure: `0.24282035035805707`