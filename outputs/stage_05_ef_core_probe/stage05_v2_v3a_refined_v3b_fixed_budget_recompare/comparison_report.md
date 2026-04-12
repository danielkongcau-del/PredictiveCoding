# Stage 05 v2 vs v3-A vs Promoted v3-B Comparison

## Protocol
- comparison scope: `fixed_budget_comparison`
- dataset: `digits`
- seeds: `[0, 1, 2]`
- shared batch size: `128`
- Stage 05 epochs: `1536`
- promoted candidate: `stage05_v3b_stronger_traj_curr_weight`

## Decision
- `stage05_v3b_keeps_one_step_mechanism_positive`: `True`
- `stage05_v3b_materially_improves_configured_step_mechanism_vs_v3a`: `True`
- `stage05_v3b_shows_positive_gap_closure_signal_vs_v3a`: `True`
- gap_closure_decision: `promoted_refined_v3b_materially_beats_v3a`
- recommended next move: `promote_refined_v3b_as_active_reference`
- rationale: `The promoted refined v3-B candidate `stage05_v3b_stronger_traj_curr_weight` materially improves configured-step mechanism over the active fixed-budget v3-A reference, preserves the minimum mechanism checks, and does not show an obvious report-only accuracy regression.`

## Pairwise Deltas Vs V2
- candidate: `stage05_v3b_stronger_traj_curr_weight`
- configured-step validation energy delta vs identity delta: `-0.0007398060152670657`
- configured-step validation fixed-point residual delta vs identity delta: `-3.1570840134360887e-06`

## Pairwise Deltas Vs V3-A
- candidate: `stage05_v3b_stronger_traj_curr_weight`
- configured-step validation energy delta vs identity delta: `-0.0002893647044687105`
- configured-step validation fixed-point residual delta vs identity delta: `-1.313480315204278e-06`

## Promoted Candidate Decision
- `promoted_refined_v3b_materially_beats_v3a`: `True`
- `promoted_refined_v3b_avoids_obvious_report_accuracy_regression`: `True`
- `promoted_refined_v3b_replaces_v3a_as_active_reference`: `True`

## Contextual 3072 Gap Closure
- v3-A configured-step energy gap closure: `0.36966230730808747`
- candidate configured-step energy gap closure: `0.6071343635851628`
- v3-A configured-step residual gap closure: `0.17573315155025793`
- candidate configured-step residual gap closure: `0.3009346986677075`