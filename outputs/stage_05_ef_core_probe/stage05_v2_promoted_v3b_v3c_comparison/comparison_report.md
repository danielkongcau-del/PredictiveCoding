# Stage 05 v2 vs Promoted v3-B vs v3-C Comparison

## Protocol
- comparison scope: `smoke_only`
- dataset: `digits`
- seeds: `[0]`
- shared batch size: `128`
- Stage 05 epochs: `4`

## Decision
- `stage05_v3c_keeps_one_step_mechanism_positive`: `True`
- `stage05_v3c_materially_improves_configured_step_mechanism_vs_promoted_v3b`: `False`
- `stage05_v3c_shows_positive_gap_closure_signal_vs_promoted_v3b`: `False`
- gap_closure_decision: `pending_real_fixed_budget_v2_vs_promoted_v3b_vs_v3c_comparison`
- recommended next move: `run_fixed_budget_v2_vs_promoted_v3b_vs_v3c_comparison`
- rationale: `The smoke run only verifies that the diagnostic-only v3-C candidate is wired, deterministic, and comparable against the promoted refined v3-B reference.`

## Pairwise Deltas Vs Promoted v3-B
- configured-step validation energy delta vs identity delta: `-2.115771946531453e-09`
- configured-step validation fixed-point residual delta vs identity delta: `-1.090886934821192e-11`

## Pairwise Deltas Vs V2
- configured-step validation energy delta vs identity delta: `2.3826698175022365e-06`
- configured-step validation fixed-point residual delta vs identity delta: `1.2313346660250109e-08`