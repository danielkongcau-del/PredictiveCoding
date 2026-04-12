# Stage 05 V2 Efficiency Diagnostic At 1536

## Protocol
- dataset: `digits`
- seeds: `[0, 1, 2]`
- shared batch size: `128`
- shared shuffle_batches: `True`
- fixed budget ceiling epochs: `1536`
- tested axes: `{'lambda_id_warmup_epochs': 1, 'lambda_id_ramp_epochs': 1}`
- contextual reference epochs: `3072`

## Decision
- `same_family_efficiency_change_materially_improves_configured_step_mechanism`: `False`
- `same_family_efficiency_change_materially_improves_report_only_accuracy`: `False`
- `same_family_efficiency_change_materially_narrows_gap_to_3072_reference`: `False`
- configured-step gain fraction vs reference: `0.00048434519746932834`
- report accuracy gain vs reference: `{'val_accuracy_delta': 0.0, 'test_accuracy_delta': 0.0}`
- configured-step gap closed fraction vs 3072 reference: `{'energy': 0.0034571540023809045, 'residual': 0.0008518822920464886}`
- report accuracy gap closed fraction vs 3072 reference: `{'val_accuracy': 0.0, 'test_accuracy': 0.0}`
- optimized candidate still hits final training boundary on all seeds: `True`
- recommended next move: `open_stage05_v3_charter`
- rationale: `The narrow same-family schedule change does not materially improve the 1536-epoch reference, so same-family efficiency tuning is not a strong enough next move and a true Stage 05 v3 mechanism charter is now justified.`

## Contextual 3072 Reference
- source: `outputs/stage_05_ef_core_probe/stage05_v2_budget_push_validation_1536_to_3072/aggregate_summary.json`
- validation/test accuracy means: `0.908641975308642` / `0.9160493827160495`
- configured-step validation energy delta vs identity mean: `-0.006199075439848138`
- configured-step validation fixed-point residual delta vs identity mean: `-2.8942715605407296e-05`

## Supports
- The optimized 1536-epoch Stage 05 v2 candidate does not materially improve configured-step mechanism over the current 1536-epoch default.
- The optimized 1536-epoch Stage 05 v2 candidate does not materially improve report-only accuracy over the current 1536-epoch default.
- The optimized 1536-epoch Stage 05 v2 candidate does not materially narrow the gap to the contextual 3072-epoch reference.
- Current 1536-epoch default configured-step validation energy delta vs identity mean: -0.004980554368.
- Optimized 1536-epoch candidate configured-step validation energy delta vs identity mean: -0.004984766983.
- Contextual 3072-epoch reference configured-step validation energy delta vs identity mean: -0.006199075440.
- Current 1536-epoch default validation/test accuracy means: 0.887654 / 0.887654.
- Optimized 1536-epoch candidate validation/test accuracy means: 0.887654 / 0.887654.
- Contextual 3072-epoch reference validation/test accuracy means: 0.908642 / 0.916049.

## Does Not Support
- This diagnostic does not reopen Stage 04 package-internal work.
- This diagnostic does not change the Stage 05 v2 transport family, residual branch structure, corrected residual identity contract, or selection rule.
- This diagnostic does not claim that Stage 05 replaces the frozen Stage 04 bridge result on main.
