# Stage05 v2 vs Active v3-C vs Precision-Weighted Continuation-Corrector Comparison

## Protocol
- scope: `fixed_budget_comparison`
- seeds: `[0, 1, 2]`
- dataset: `digits`
- Stage 05 epochs: `1536`

## Decision
- active refined v3-C reference: `stage05_v3c_stronger_semigroup_weight`
- precision-weighted continuation-corrector candidate: `stage05_v3c_precision_weighted_continuation_corrector_trajectory_contract`
- `precision_weighted_continuation_corrector_contract_materially_beats_active_v3c_reference`: `False`
- `precision_weighted_continuation_corrector_contract_avoids_obvious_report_accuracy_regression`: `True`
- `precision_weighted_continuation_corrector_contract_shows_positive_gap_closure_signal_vs_active_v3c`: `True`
- `recommended_next_move`: `keep_precision_weighted_continuation_corrector_direction_and_refine_implementation`
- rationale: `The precision-weighted continuation-corrector v3-C candidate moves configured-step mechanism in the right direction, but it does not yet materially and cleanly displace the current active refined v3-C reference.`

## Pairwise Deltas Vs Active v3-C
- configured-step validation energy delta vs identity delta: `-6.290335839477061e-05`
- configured-step validation fixed-point residual delta vs identity delta: `-2.5679286279775153e-07`

## Pairwise Deltas Vs V2
- configured-step validation energy delta vs identity delta: `-0.0010529706817276423`
- configured-step validation fixed-point residual delta vs identity delta: `-4.506781816440151e-06`