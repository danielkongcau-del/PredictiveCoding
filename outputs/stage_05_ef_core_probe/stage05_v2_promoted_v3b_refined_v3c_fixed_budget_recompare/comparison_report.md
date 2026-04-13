# Stage 05 v2 vs Promoted v3-B vs v3-C Comparison

## Protocol
- comparison scope: `fixed_budget_comparison`
- dataset: `digits`
- seeds: `[0, 1, 2]`
- shared batch size: `128`
- Stage 05 epochs: `1536`

## Decision
- refined v3-C formal comparison candidate: `stage05_v3c_stronger_semigroup_weight`
- `stage05_v3c_keeps_one_step_mechanism_positive`: `True`
- `stage05_v3c_materially_improves_configured_step_mechanism_vs_promoted_v3b`: `True`
- `stage05_v3c_shows_positive_gap_closure_signal_vs_promoted_v3b`: `True`
- gap_closure_decision: `positive_gap_closure_signal_vs_promoted_v3b`
- recommended next move: `promote_refined_v3c_as_active_reference`
- rationale: `The refined v3-C formal comparison candidate `stage05_v3c_stronger_semigroup_weight` improves configured-step mechanism over the promoted refined v3-B reference, preserves the mechanism-first gate, and improves contextual gap closure versus the 3072-epoch same-family reference.`

## Pairwise Deltas Vs Promoted v3-B
- configured-step validation energy delta vs identity delta: `-0.000250261308065806`
- configured-step validation fixed-point residual delta vs identity delta: `-1.0929049402063115e-06`

## Pairwise Deltas Vs V2
- configured-step validation energy delta vs identity delta: `-0.0009900673233328716`
- configured-step validation fixed-point residual delta vs identity delta: `-4.2499889536424e-06`