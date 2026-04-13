# Stage 05 v2 vs Promoted v3-B vs v3-C Comparison

## Protocol
- comparison scope: `fixed_budget_comparison`
- dataset: `digits`
- seeds: `[0, 1, 2]`
- shared batch size: `128`
- Stage 05 epochs: `1536`

## Decision
- `stage05_v3c_keeps_one_step_mechanism_positive`: `True`
- `stage05_v3c_materially_improves_configured_step_mechanism_vs_promoted_v3b`: `False`
- `stage05_v3c_shows_positive_gap_closure_signal_vs_promoted_v3b`: `True`
- gap_closure_decision: `directional_gain_without_clear_gap_closure`
- recommended next move: `keep_v3c_diagnostic_only_and_refine_implementation`
- rationale: `The v3-C diagnostic probe preserves the mechanism-first gate but does not yet materially and cleanly displace the promoted refined v3-B reference.`

## Pairwise Deltas Vs Promoted v3-B
- configured-step validation energy delta vs identity delta: `-0.00012638542340264366`
- configured-step validation fixed-point residual delta vs identity delta: `-5.546981464183299e-07`

## Pairwise Deltas Vs V2
- configured-step validation energy delta vs identity delta: `-0.0008661914386697093`
- configured-step validation fixed-point residual delta vs identity delta: `-3.7117821598544186e-06`