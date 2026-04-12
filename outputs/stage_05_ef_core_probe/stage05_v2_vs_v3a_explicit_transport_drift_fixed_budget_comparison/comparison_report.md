# Stage 05 v2 vs v3-A Explicit Transport-Drift Contract

## Protocol
- comparison scope: `fixed_budget_comparison`
- dataset: `digits`
- seeds: `[0, 1, 2]`
- shared batch size: `128`
- shared shuffle_batches: `True`
- Stage 05 epochs: `1536`

## Decision
- `stage05_v3a_shows_positive_gap_closure_signal_vs_v2`: `True`
- `stage05_v3a_materially_improves_configured_step_mechanism`: `True`
- `stage05_v3a_avoids_obvious_report_accuracy_regression`: `True`
- gap_closure_decision: `material_positive_gap_closure_signal`
- recommended next move: `proceed_to_stage05_v3b_curriculum_charter`
- rationale: `The fixed-budget v3-A candidate materially improves configured-step mechanism over v2, narrows the gap to the contextual 3072-epoch reference, and does not show an obvious report-only accuracy regression.`

## Pairwise Deltas Vs V2
- one-step validation energy delta vs identity delta: `-0.0004565189253461592`
- configured-step validation energy delta vs identity delta: `-0.00045044131079835514`
- configured-step validation fixed-point residual delta vs identity delta: `-1.8436036982318108e-06`
- validation accuracy delta: `-0.006172839506172867`
- test accuracy delta: `-0.006172839506172793`

## Supports
- The Stage 05 v3-A candidate path writes the standard Stage 05 artifacts.
- The v3-A candidate keeps artifact-independent target construction and the existing aggregate residual identity target.
- The comparison exposes explicit pairwise deltas versus the current Stage 05 v2 reference.
- The fixed-budget v3-A candidate keeps one-step validation energy delta vs identity negative on every seed.
- The fixed-budget v3-A candidate materially improves configured-step mechanism over the v2 reference.
- The fixed-budget v3-A candidate avoids an obvious report-only accuracy regression.
- Pairwise configured-step validation energy delta vs identity mean difference vs v2: -0.000450441311.
- Pairwise configured-step validation fixed-point residual delta vs identity mean difference vs v2: -0.000001843604.

## Does Not Support
- This comparison does not justify replacing the frozen Stage 04 bridge on main.
- This comparison does not promote task accuracy to the Stage 05 gate.
- This comparison does not reopen Stage 04 package-internal work.

## Contextual 3072 Reference
- source: `outputs/stage_05_ef_core_probe/stage05_v2_budget_push_validation_1536_to_3072/aggregate_summary.json`
- configured-step validation energy delta vs identity mean: `-0.006199075439848138`
- configured-step validation fixed-point residual delta vs identity mean: `-2.8942715605407296e-05`
- validation/test accuracy means: `0.908641975308642` / `0.9160493827160495`
