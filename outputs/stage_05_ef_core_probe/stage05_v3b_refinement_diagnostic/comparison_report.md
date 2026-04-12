# Stage 05 v3-B Refinement Diagnostic

## Protocol
- comparison scope: `fixed_budget_comparison`
- dataset: `digits`
- seeds: `[0, 1, 2]`
- shared batch size: `128`
- Stage 05 epochs: `1536`

## Decision
- best variant: `stage05_v3b_stronger_traj_curr_weight`
- `narrow_v3b_refinement_materially_beats_v3b_control`: `False`
- `narrow_v3b_refinement_materially_beats_v3a_reference`: `True`
- recommended next move: `promote_refined_v3b_and_recompare`
- rationale: `A narrow v3-B refinement materially improves configured-step mechanism over the active v3-A reference without an obvious report-only accuracy regression, so it is worth promoting into a fresh fixed-budget re-comparison.`

## Pairwise Deltas Vs V3-B Control
- `stage05_v3b_alpha_earlier_transition` configured-step energy delta vs identity delta: `1.0208240191110033e-05`
- `stage05_v3b_alpha_earlier_transition` configured-step fixed-point residual delta vs identity delta: `5.730636671165239e-08`
- `stage05_v3b_stronger_traj_curr_weight` configured-step energy delta vs identity delta: `-0.00013596252906332892`
- `stage05_v3b_stronger_traj_curr_weight` configured-step fixed-point residual delta vs identity delta: `-6.096733969592636e-07`

## Pairwise Deltas Vs V3-A
- `stage05_v3b_alpha_earlier_transition` configured-step energy delta vs identity delta: `-0.00014319393521427157`
- `stage05_v3b_alpha_earlier_transition` configured-step fixed-point residual delta vs identity delta: `-6.465005515333619e-07`
- `stage05_v3b_stronger_traj_curr_weight` configured-step energy delta vs identity delta: `-0.0002893647044687105`
- `stage05_v3b_stronger_traj_curr_weight` configured-step fixed-point residual delta vs identity delta: `-1.313480315204278e-06`

## Contextual 3072 Gap Closure
- `stage05_v3a_explicit_transport_drift_contract` configured-step energy / residual gap closure: `0.36966230730808747` / `0.17573315155025793`
- `stage05_v3b_trajectory_curriculum_contract` configured-step energy / residual gap closure: `0.49555440633853987` / `0.24282035035805707`
- `stage05_v3b_alpha_earlier_transition` configured-step energy / residual gap closure: `0.487176840755329` / `0.23735788115396936`
- `stage05_v3b_stronger_traj_curr_weight` configured-step energy / residual gap closure: `0.6071343635851628` / `0.3009346986677075`