# Stage 05 v2 vs Active v3-C vs Fused Contract Comparison

## Protocol
- comparison scope: `fixed_budget_comparison`
- dataset: `digits`
- seeds: `[0, 1, 2]`
- shared batch size: `128`
- Stage 05 epochs: `1536`

## Decision
- active refined v3-C reference: `stage05_v3c_stronger_semigroup_weight`
- fused candidate: `stage05_v3c_fused_trajectory_semigroup_contract`
- `fused_contract_materially_beats_active_v3c_reference`: `False`
- `fused_contract_avoids_obvious_report_accuracy_regression`: `True`
- `fused_contract_shows_positive_gap_closure_signal_vs_active_v3c`: `True`
- gap_closure_decision: `directional_but_not_material_gap_closure_vs_active_v3c`
- recommended next move: `keep_fusion_direction_and_refine_implementation`
- rationale: `The fused v3-C candidate moves configured-step mechanism in the right direction without yet materially and cleanly displacing the current active refined v3-C reference.`

## Pairwise Deltas Vs Active v3-C
- configured-step validation energy delta vs identity delta: `-3.6070771314289374e-08`
- configured-step validation fixed-point residual delta vs identity delta: `-1.5350606679884193e-10`

## Pairwise Deltas Vs V2
- configured-step validation energy delta vs identity delta: `-0.000990103394104186`
- configured-step validation fixed-point residual delta vs identity delta: `-4.250142459709199e-06`