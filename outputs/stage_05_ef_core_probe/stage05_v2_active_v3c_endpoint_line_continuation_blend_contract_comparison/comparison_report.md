# Stage 05 v2 vs Active v3-C vs Endpoint-Line Continuation-Blend Contract Comparison

## Protocol
- comparison scope: `fixed_budget_comparison`
- dataset: `digits`
- seeds: `[0, 1, 2]`
- shared batch size: `128`
- Stage 05 epochs: `1536`

## Decision
- active refined v3-C reference: `stage05_v3c_stronger_semigroup_weight`
- endpoint-line continuation-blend candidate: `stage05_v3c_endpoint_line_continuation_blend_trajectory_contract`
- `endpoint_line_continuation_blend_contract_materially_beats_active_v3c_reference`: `False`
- `endpoint_line_continuation_blend_contract_avoids_obvious_report_accuracy_regression`: `True`
- `endpoint_line_continuation_blend_contract_shows_positive_gap_closure_signal_vs_active_v3c`: `True`
- gap_closure_decision: `directional_but_not_material_gap_closure_vs_active_v3c`
- recommended next move: `keep_endpoint_line_continuation_blend_direction_and_refine_implementation`
- rationale: `The endpoint-line continuation-blend v3-C candidate moves configured-step mechanism in the right direction, but it does not yet materially and cleanly displace the current active refined v3-C reference.`

## Pairwise Deltas Vs Active v3-C
- configured-step validation energy delta vs identity delta: `-0.00010648557828197662`
- configured-step validation fixed-point residual delta vs identity delta: `-4.5277710370737165e-07`

## Pairwise Deltas Vs V2
- configured-step validation energy delta vs identity delta: `-0.0010965529016148483`
- configured-step validation fixed-point residual delta vs identity delta: `-4.702766057349771e-06`