# Stage 05 v2 vs Active v3-C vs Midpoint-Reconstructed Contract Comparison

## Protocol
- comparison scope: `fixed_budget_comparison`
- dataset: `digits`
- seeds: `[0, 1, 2]`
- shared batch size: `128`
- Stage 05 epochs: `1536`

## Decision
- active refined v3-C reference: `stage05_v3c_stronger_semigroup_weight`
- midpoint-reconstructed candidate: `stage05_v3c_midpoint_reconstructed_trajectory_contract`
- `midpoint_reconstructed_contract_materially_beats_active_v3c_reference`: `False`
- `midpoint_reconstructed_contract_avoids_obvious_report_accuracy_regression`: `True`
- `midpoint_reconstructed_contract_shows_positive_gap_closure_signal_vs_active_v3c`: `False`
- gap_closure_decision: `no_positive_gap_closure_signal_vs_active_v3c`
- recommended next move: `retain_stage05_v3c_stronger_semigroup_weight_as_active_reference`
- rationale: `The midpoint-reconstructed v3-C candidate does not yet show a strong enough configured-step gain over the current active refined v3-C reference.`

## Pairwise Deltas Vs Active v3-C
- configured-step validation energy delta vs identity delta: `3.0566621458949506e-07`
- configured-step validation fixed-point residual delta vs identity delta: `1.3463440786132725e-09`

## Pairwise Deltas Vs V2
- configured-step validation energy delta vs identity delta: `-0.0009897616571182821`
- configured-step validation fixed-point residual delta vs identity delta: `-4.248642609563787e-06`