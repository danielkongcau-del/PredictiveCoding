# Stage 05 v2 vs Active v3-C vs Coupled Defect-Projection Contract Comparison

## Protocol
- comparison scope: `fixed_budget_comparison`
- dataset: `digits`
- seeds: `[0, 1, 2]`
- shared batch size: `128`
- Stage 05 epochs: `1536`

## Decision
- active refined v3-C reference: `stage05_v3c_stronger_semigroup_weight`
- coupled defect-projection candidate: `stage05_v3c_coupled_defect_projection_trajectory_contract`
- `coupled_defect_projection_contract_materially_beats_active_v3c_reference`: `False`
- `coupled_defect_projection_contract_avoids_obvious_report_accuracy_regression`: `True`
- `coupled_defect_projection_contract_shows_positive_gap_closure_signal_vs_active_v3c`: `True`
- gap_closure_decision: `directional_but_not_material_gap_closure_vs_active_v3c`
- recommended next move: `keep_coupled_defect_projection_direction_and_refine_implementation`
- rationale: `The coupled defect-projection v3-C candidate moves configured-step mechanism in the right direction, but it does not yet materially and cleanly displace the current active refined v3-C reference.`

## Pairwise Deltas Vs Active v3-C
- configured-step validation energy delta vs identity delta: `-9.401600908822323e-05`
- configured-step validation fixed-point residual delta vs identity delta: `-3.9714172550900373e-07`

## Pairwise Deltas Vs V2
- configured-step validation energy delta vs identity delta: `-0.0010840833324210948`
- configured-step validation fixed-point residual delta vs identity delta: `-4.647130679151404e-06`