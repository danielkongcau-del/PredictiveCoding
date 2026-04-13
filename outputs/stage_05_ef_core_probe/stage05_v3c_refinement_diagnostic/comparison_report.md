# Stage 05 v3-C Refinement Diagnostic

## Protocol
- comparison scope: `fixed_budget_comparison`
- dataset: `digits`
- seeds: `[0, 1, 2]`
- shared batch size: `128`

## Decision
- best variant: `stage05_v3c_stronger_semigroup_weight`
- `narrow_v3c_refinement_materially_beats_v3c_control`: `False`
- `narrow_v3c_refinement_materially_beats_promoted_v3b_reference`: `True`
- `narrow_v3c_refinement_avoids_obvious_report_accuracy_regression`: `True`
- recommended next move: `promote_refined_v3c_and_recompare`
- rationale: `The stronger semigroup-weight refinement materially improves configured-step mechanism over the promoted refined v3-B reference while preserving the current mechanism-first gate.`

## Pairwise Deltas Vs v3-C Control
- configured-step validation energy delta vs identity delta: `-0.00012387588466316232`
- configured-step validation fixed-point residual delta vs identity delta: `-5.382067937879815e-07`

## Pairwise Deltas Vs Promoted v3-B
- configured-step validation energy delta vs identity delta: `-0.000250261308065806`
- configured-step validation fixed-point residual delta vs identity delta: `-1.0929049402063115e-06`