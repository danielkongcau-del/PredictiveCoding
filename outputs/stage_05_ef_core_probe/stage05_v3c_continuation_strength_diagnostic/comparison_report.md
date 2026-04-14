# Stage 05 Continuation-Strength Diagnostic

## Protocol
- comparison scope: `fixed_budget_comparison`
- dataset: `digits`
- seeds: `[0, 1, 2]`
- shared batch size: `128`
- Stage 05 epochs: `1536`
- continuation blend scale: `1.5`

## Decision
- local best existing predecessor: `stage05_v3c_endpoint_line_continuation_blend_trajectory_contract`
- scaled candidate: `stage05_v3c_scaled_continuation_blend_trajectory_contract`
- `scaled_continuation_blend_contract_materially_beats_active_v3c_reference`: `False`
- `scaled_continuation_blend_contract_avoids_obvious_report_accuracy_regression`: `True`
- `scaled_continuation_blend_contract_is_strongest_narrow_microfamily_candidate`: `True`
- `scaled_continuation_blend_contract_supports_continuation_strength_as_remaining_lever`: `True`
- `narrow_v3c_contract_consolidation_line_is_locally_saturated`: `True`
- `final_decision`: `freeze_narrow_v3c_contract_consolidation_line_as_locally_saturated`
- rationale: `The scaled continuation-blend diagnostic does not materially and cleanly displace the current active refined v3-C reference. The narrow v3-C contract-consolidation micro-family should be treated as locally saturated and the next new Stage 05 mechanism pass should move above this micro-family rather than remain inside it.`

## Pairwise Deltas Vs Active v3-C
- energy delta: `-0.00013847578449683687`
- residual delta: `-5.967061577316692e-07`

## Pairwise Deltas Vs Continuation-Blend Predecessor
- energy delta: `-3.199020621486026e-05`
- residual delta: `-1.439290540242976e-07`

## Micro-family Ranking
- `#1` `stage05_v3c_scaled_continuation_blend_trajectory_contract` (energy `-0.006109097476166642`, residual `-2.3298483524227394e-05`)
- `#2` `stage05_v3c_endpoint_line_continuation_blend_trajectory_contract` (energy `-0.006077107269951781`, residual `-2.3154554470203098e-05`)
- `#3` `stage05_v3c_coupled_defect_projection_trajectory_contract` (energy `-0.006064637700758028`, residual `-2.309891909200473e-05`)
- `#4` `stage05_v3c_precision_weighted_continuation_corrector_trajectory_contract` (energy `-0.006033525050064575`, residual `-2.2958570229293478e-05`)
- `#5` `stage05_v3c_endpoint_line_midpoint_trajectory_contract` (energy `-0.006015732548759099`, residual `-2.287737578720965e-05`)