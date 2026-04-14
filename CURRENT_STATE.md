# CURRENT_STATE.md

This file is the short operational summary for the repository.

- Use it for current state only.
- Use [PLANS.md](/e:/CodeSpace/PredictiveCoding/PLANS.md) for the active forward plan.
- Use [archive/PLANS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/PLANS_HISTORY.md) and [archive/validation_history.md](/e:/CodeSpace/PredictiveCoding/archive/validation_history.md) for long historical detail.

## Active Branch And Line

- Active branch:
  - `main`
- Active algorithmic line on `main`:
  - `FMPC Stage 04 Incremental Bridge`

For the full numbered stage map and directory layout, use [README.md](/e:/CodeSpace/PredictiveCoding/README.md).

## Current Defaults

- Current adopted Stage 04 bridge default on `main`:
  - `tf2_corrective_transport_terminal_angleclip_default`
- Current canonical Stage 04 identity default:
  - `feature_aware_tangents = false`
- Historical corrective working reference:
  - `tf2_corrective_transport_default`

## Relevant Math Layer

- Baseline root spec:
  - [spec_math.md](/e:/CodeSpace/PredictiveCoding/spec_math.md)
- Stage 04 bridge math:
  - [specs/stage_03_transport_core_v1.md](/e:/CodeSpace/PredictiveCoding/specs/stage_03_transport_core_v1.md)
  - [specs/stage_04_incremental_bridge.md](/e:/CodeSpace/PredictiveCoding/specs/stage_04_incremental_bridge.md)
- Stage 05 exploratory math:
  - [specs/stage_05_ef_core_probe.md](/e:/CodeSpace/PredictiveCoding/specs/stage_05_ef_core_probe.md)

## Stage 04 Status

The corrective Incremental Bridge package is now treated as closed from the current state.

Condensed reasons:

- the current hard full-vector `30` degree terminal angle clip remains the local winner under the fixed selector/gate contract
- selector / checkpoint is not the main limiter
- simple head-fit / separability is not the main limiter
- readout alignment is sealed as a no-op on the adopted package
- detached bootstrap-source, one-step target lag, and bootstrap-to-identity curriculum follow-ups do not materially improve the adopted package
- the late-rollout successor-value and successor-increment line ends in a strengthened formulation-level blocker rather than an adoption-viable local fix

Operational conclusion:

- the adopted corrective Stage 04 bridge package is locally saturated under the current selector/gate contract
- do not reopen package-internal Stage 04 digging unless:
  - a genuinely different issue family appears from new evidence, or
  - the project explicitly decides to leave the current package or selector-gate contract

## Stage 05 Status

The current open work is the post-bridge exploratory line:

- `FMPC Stage 05 EF Core Probe`

Current probe status:

- implementation exists under:
  - [src/pc/stage_05_ef_core_probe/fmpc_ef_exploratory_probe.py](/e:/CodeSpace/PredictiveCoding/src/pc/stage_05_ef_core_probe/fmpc_ef_exploratory_probe.py)
- the current Stage 05 baseline contract remains a corrected residual MeanFlow core v1:
  - residual transport family on top of the exact local flow
  - bootstrap residual supervision
  - corrected residual identity curriculum
- the current narrow Stage 05 v2 candidate now also exists in the same implementation:
  - two-branch residual decomposition
  - `m_traj_input = concat([z_t, target_onehot, t, r])`
  - `m_state_input = concat([g_t, e_out_t, F_t])`
  - explicit branchwise corrected residual identity target
- target construction remains artifact-independent
- the first canonical probe shows positive mechanism signal:
  - one-step validation energy improves vs identity
  - configured two-step validation energy improves vs identity
  - configured two-step fixed-point residual improves vs identity
- the formal frozen-bridge vs corrected-core comparison now exists under:
  - [outputs/stage_05_ef_core_probe/frozen_bridge_vs_corrected_core_comparison](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/frozen_bridge_vs_corrected_core_comparison)
- current comparison result:
  - the older frozen-bridge comparison used the Stage 05 v1 baseline
  - it remains a valid bridge-vs-baseline reference artifact
- the formal Stage 05 v1 vs v2 comparison now exists under:
  - [outputs/stage_05_ef_core_probe/corrected_residual_core_v1_vs_v2_comparison](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/corrected_residual_core_v1_vs_v2_comparison)
- current v1 vs v2 result:
  - Stage 05 v2 keeps one-step validation energy delta vs identity negative on all comparison seeds
  - Stage 05 v2 keeps configured-step validation energy delta vs identity negative on all comparison seeds
  - Stage 05 v2 keeps configured-step validation fixed-point residual delta vs identity negative on all comparison seeds
  - Stage 05 v2 improves mean configured-step validation energy delta vs identity over v1
  - Stage 05 v2 improves mean configured-step validation fixed-point residual delta vs identity over v1
- the refreshed frozen-bridge vs Stage 05 v2 comparison now exists under:
  - [outputs/stage_05_ef_core_probe/frozen_bridge_vs_two_branch_corrected_core_comparison](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/frozen_bridge_vs_two_branch_corrected_core_comparison)
- refreshed v2 vs frozen-bridge result:
  - Stage 05 v2 keeps one-step validation energy delta vs identity negative on all comparison seeds
  - Stage 05 v2 keeps configured-step validation energy delta vs identity negative on all comparison seeds
  - Stage 05 v2 keeps configured-step validation fixed-point residual delta vs identity negative on all comparison seeds
  - Stage 05 v2 is stronger than the frozen bridge on one-step mechanism
  - Stage 05 v2 is weaker than the frozen bridge on configured-step mechanism
  - Stage 05 v2 is weaker than the frozen bridge on report-only accuracy
  - the refreshed comparison supports continued Stage 05 exploration
  - the refreshed comparison supports using Stage 05 v2 as the new exploratory reference
  - the refreshed comparison does not support replacing the frozen Stage 04 bridge result on `main`
- the dedicated Stage 05 v2 diagnostics now exist under:
  - [outputs/stage_05_ef_core_probe/stage05_v2_diagnostics](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_diagnostics)
- current v2 diagnostic result:
  - all comparison seeds select the final training epoch
  - configured-step mechanism metrics are still improving at the training boundary
  - validation accuracy is also still improving at the training boundary
  - the selected epoch is already the highest validation-accuracy epoch on every seed
  - the forward state branch contribution is materially active rather than negligible
  - the current low report-only accuracy is not primarily a selection-rule artifact
  - the current narrow diagnosis is:
    - `likely_undertrained`
- the narrow Stage 05 v2 longer-training validation now exists under:
  - [outputs/stage_05_ef_core_probe/stage05_v2_longer_training_validation](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_longer_training_validation)
- the next narrow Stage 05 v2 budget-push validation now also exists under:
  - [outputs/stage_05_ef_core_probe/stage05_v2_budget_push_validation_1536_to_3072](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_budget_push_validation_1536_to_3072)
- current budget-push result:
  - the same v2 family with a stronger `3072`-epoch budget materially improves configured-step mechanism magnitude over the current `1536`-epoch reference
  - the same stronger budget also materially improves report-only validation and test accuracy
  - the stronger budget still selects the final training epoch on every seed
  - the explicit stop-rule layer still says:
    - `budget_line_still_looks_boundary_limited = true`
    - `budget_line_should_continue = true`
    - `budget_line_should_stop_and_open_v3 = false`
    - `budget_line_interpretation = boundary_limited_mechanism_prototype`
  - the stronger budget now sits above the frozen Stage 04 bridge accuracy level in the diagnostic context, while remaining mixed relative to the standalone `digits_pc` baseline and below the standalone `digits_mlp` baseline
  - that budget result remains useful as contextual same-family evidence, not as the active next move
- the narrow Stage 05 v2 efficiency diagnostic at the fixed `1536`-epoch ceiling now also exists under:
  - [outputs/stage_05_ef_core_probe/stage05_v2_efficiency_diagnostic_at_1536](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_efficiency_diagnostic_at_1536)
- current efficiency result:
  - the tested same-family schedule change only moves configured-step validation energy delta vs identity from `-0.004980554368336933` to `-0.004984766983336293`
  - configured-step validation fixed-point residual delta vs identity only moves from `-1.8451788412853327e-05` to `-1.8460725447955813e-05`
  - report-only validation and test accuracy remain unchanged at `0.8876543209876543 / 0.8876543209876543`
  - the gap-closure fractions versus the contextual `3072`-epoch reference remain negligible
  - the optimized `1536`-epoch candidate still selects the final training epoch on every seed
  - the diagnostic conclusion is therefore:
    - `same_family_efficiency_change_materially_improves_configured_step_mechanism = false`
    - `same_family_efficiency_change_materially_improves_report_only_accuracy = false`
    - `same_family_efficiency_change_materially_narrows_gap_to_3072_reference = false`
    - `recommended_next_move = implement_stage05_v3a_candidate`
- the next Stage 05 charter was then defined as:
  - `Stage 05 v3-A`
  - `explicit transport-drift contract`
- current v3-A charter motivation:
  - it is driven by the working hypothesis that the current residual target may entangle transport residual and anchor-drift residual too tightly
  - that hypothesis is being treated as charter motivation, not as a proved repository conclusion
- the first minimal v3-A candidate path now exists in the Stage 05 probe:
  - candidate name: `stage05_v3a_explicit_transport_drift_contract`
  - it reuses the current two-branch scaffold but replaces the bootstrap supervision with an explicit `q_boot / d_boot` split
  - a smoke-level `v2 vs v3-A` comparison entry now also exists and writes formal aggregate artifacts
- the first real fixed-budget `v2 vs v3-A` comparison now also exists under:
  - [outputs/stage_05_ef_core_probe/stage05_v2_vs_v3a_explicit_transport_drift_fixed_budget_comparison](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_vs_v3a_explicit_transport_drift_fixed_budget_comparison)
- current fixed-budget v3-A comparison result:
  - the fixed-budget v3-A candidate materially improves configured-step mechanism over the `1536`-epoch v2 reference
  - it keeps one-step mechanism positive on every comparison seed
  - it shows a positive gap-closure signal relative to the contextual `3072`-epoch v2 reference
  - it avoids an obvious report-only accuracy regression under the current Stage 05 rule, although validation and test accuracy remain slightly below the fixed-budget v2 reference
  - the current comparison decision is:
    - `stage05_v3a_shows_positive_gap_closure_signal_vs_v2 = true`
    - `stage05_v3a_materially_improves_configured_step_mechanism = true`
    - `recommended_next_move = proceed_to_stage05_v3b_curriculum_charter`
- the next Stage 05 charter is now defined as:
  - `Stage 05 v3-B`
  - `trajectory curriculum contract`
- current v3-B charter motivation:
  - it is driven by the working hypothesis that, after explicit transport-drift decomposition, the main remaining fixed-budget inefficiency is trajectory-level
  - more specifically, the current remaining weakness may come from optimization conflict between direct horizon matching and recursive continuation consistency
  - that hypothesis is being treated as charter motivation, not as a proved repository conclusion
- the first minimal v3-B candidate path now also exists in the Stage 05 probe:
  - candidate name: `stage05_v3b_trajectory_curriculum_contract`
  - it keeps the v3-A explicit transport-drift decomposition and adds one aggregate trajectory curriculum loss
  - a smoke-ready `v2 vs v3-A vs v3-B` comparison entry now also exists
- the first real fixed-budget `v2 vs v3-A vs v3-B` comparison now also exists under:
  - [outputs/stage_05_ef_core_probe/stage05_v2_v3a_v3b_fixed_budget_comparison](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_v3a_v3b_fixed_budget_comparison)
- original fixed-budget v3-B comparison result:
  - the fixed-budget v3-B candidate keeps one-step validation energy delta vs identity negative on every comparison seed
  - it materially improves configured-step mechanism over the fixed-budget v2 control
  - it directionally improves configured-step mechanism over the fixed-budget v3-A reference on every comparison seed
  - its configured-step gain over v3-A remains below the current materiality threshold
  - it slightly lowers report-only validation and test accuracy relative to v3-A, but not enough to count as an obvious regression under the current Stage 05 rule
  - it improves contextual `3072`-epoch gap-closure fractions relative to v3-A, but not enough to justify promotion to the current fixed-budget Stage 05 improvement reference
  - the current comparison decision is:
    - `stage05_v3b_materially_improves_configured_step_mechanism_vs_v3a = false`
    - `stage05_v3b_shows_positive_gap_closure_signal_vs_v3a = false`
    - `recommended_next_move = retain_v3a_as_active_reference`
- the narrow Stage 05 v3-B refinement diagnostic now also exists under:
  - [outputs/stage_05_ef_core_probe/stage05_v3b_refinement_diagnostic](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v3b_refinement_diagnostic)
- current v3-B refinement result:
  - the best tested refinement variant is `stage05_v3b_stronger_traj_curr_weight`
  - it keeps one-step mechanism positive on every comparison seed
  - it materially improves configured-step mechanism over the fixed-budget v3-A reference under the current threshold
  - it does not materially improve configured-step mechanism over the current v3-B control under the same threshold
  - it avoids an obvious report-only accuracy regression relative to v3-A
  - the current refinement decision is:
    - `narrow_v3b_refinement_materially_beats_v3a_reference = true`
    - `recommended_next_move = promote_refined_v3b_and_recompare`
- the fresh fixed-budget refined v3-B recompare now also exists under:
  - [outputs/stage_05_ef_core_probe/stage05_v2_v3a_refined_v3b_fixed_budget_recompare](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_v3a_refined_v3b_fixed_budget_recompare)
- current refined v3-B recompare result:
  - the promoted refined candidate is `stage05_v3b_stronger_traj_curr_weight`
  - it materially improves configured-step mechanism over the fixed-budget v3-A reference under the current threshold
  - it materially improves configured-step mechanism over the fixed-budget v2 control
  - it keeps one-step mechanism positive on every comparison seed
  - it avoids an obvious report-only accuracy regression relative to v3-A
  - the current recompare decision is:
    - `promoted_refined_v3b_materially_beats_v3a = true`
    - `recommended_next_move = promote_refined_v3b_as_active_reference`
- the post-promotion Stage 05 planning conclusion is now:
  - the fixed-budget `v3-A -> refined v3-B` promotion question is closed
  - the most credible next mechanism question is no longer another v3-B promotion or refinement pass
  - the current ranked post-promotion mechanism suspects are:
    - endpoint / semigroup consistency contract
    - still-insufficient trajectory curriculum contract
    - corrected residual identity contract
  - the repository has now moved from a planning-only `Stage 05 v3-C` charter to a first minimal diagnostic-only `Stage 05 v3-C` probe
  - `Stage 05 v3-C` is defined by the working hypothesis that the current Stage 05 family still lacks an explicit endpoint / semigroup consistency contract across split horizons
  - this is a working hypothesis, not a proved repository conclusion
- Stage 05 evaluation remains mechanism-first:
  - task accuracy is report-only and is not the current acceptance gate
- current interpretation:
  - the two-branch corrected residual MeanFlow core remains the fixed-budget Stage 05 control
  - the refined v3-C candidate `stage05_v3c_stronger_semigroup_weight` is now the current fixed-budget Stage 05 improvement reference
  - the promoted refined v3-B result `stage05_v3b_stronger_traj_curr_weight` is now the previous fixed-budget comparison reference
  - the original fixed-budget v3-B result remains a tested predecessor branch, not the current fixed-budget improvement reference
  - it improves mechanism magnitude over the Stage 05 v1 baseline under the current multiseed rule
  - the refreshed bridge comparison now also supports treating Stage 05 v2 as the current exploratory reference
  - the longer-training and budget-push validations showed real same-family budget upside
  - the fixed-budget efficiency diagnostic then showed that a narrow same-family schedule change does not recover a material fraction of that upside
  - on this simple task, further pure epoch escalation is now treated as economically unjustified from the current state
  - the fixed-budget Stage 05 `v2 vs v3-A` comparison now shows that v3-A is a stronger configured-step branch than the fixed-budget v2 reference under the current mechanism-first rule
  - the original fixed-budget Stage 05 `v2 vs v3-A vs v3-B` comparison showed that v3-B was directionally stronger than v3-A on configured-step mechanism, but not yet strong enough for promotion
  - the refined v3-B recompare now shows that `stage05_v3b_stronger_traj_curr_weight` materially beats the fixed-budget v3-A reference and should replace it as the active fixed-budget improvement reference
  - v3-A already improved configured-step mechanism over the fixed-budget v2 control, so target-entanglement is no longer the lead unresolved mechanism question
  - refined v3-B then improved configured-step mechanism again over v3-A, so trajectory-level structure matters, but the remaining gap still points above pure weight or schedule tweaking
  - the current v3-B implementation still does not impose an explicit endpoint / semigroup consistency contract
  - the first minimal diagnostic-only v3-C probe `stage05_v3c_endpoint_semigroup_consistency_contract` now exists on top of the promoted refined v3-B scaffold
  - the smoke comparison artifact now also exists under `outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_v3c_comparison/`
  - that smoke artifact does not yet establish a fixed-budget mechanism win over the promoted refined v3-B reference; it only verifies wiring, deterministic artifacts, and comparison readiness
  - the first real fixed-budget `v2 vs promoted-v3B vs v3-C` comparison now also exists under:
    - [outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_v3c_fixed_budget_comparison](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_v3c_fixed_budget_comparison)
  - that fixed-budget v3-C comparison says:
    - v3-C keeps one-step mechanism positive on all comparison seeds
    - v3-C keeps configured-step mechanism positive on all comparison seeds
    - v3-C directionally improves configured-step mechanism over the promoted refined v3-B reference
    - v3-C improves contextual gap-closure fractions relative to the `3072`-epoch v2 reference beyond the promoted refined v3-B reference
    - v3-C avoids an obvious report-only accuracy regression relative to the promoted refined v3-B reference
    - v3-C does not materially improve configured-step mechanism over the promoted refined v3-B reference under the current threshold
    - the current comparison decision is therefore:
      - `recommended_next_move = keep_v3c_diagnostic_only_and_refine_implementation`
  - the narrow fixed-budget v3-C refinement diagnostic now also exists under:
    - [outputs/stage_05_ef_core_probe/stage05_v3c_refinement_diagnostic](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v3c_refinement_diagnostic)
  - that refinement diagnostic says:
    - the best tested refinement variant is `stage05_v3c_stronger_semigroup_weight`
    - it keeps one-step mechanism positive on all comparison seeds
    - it keeps configured-step mechanism positive on all comparison seeds
    - it materially improves configured-step mechanism over the promoted refined v3-B reference under the current threshold
    - it does not materially improve configured-step mechanism over the current v3-C control under the same threshold
    - it avoids an obvious report-only accuracy regression relative to the promoted refined v3-B reference
    - the current refinement decision is therefore:
      - `recommended_next_move = promote_refined_v3c_and_recompare`
  - the fresh fixed-budget refined v3-C recompare now also exists under:
    - [outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_refined_v3c_fixed_budget_recompare](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_refined_v3c_fixed_budget_recompare)
  - that refined v3-C recompare says:
    - the formal comparison candidate is `stage05_v3c_stronger_semigroup_weight`
    - it materially improves configured-step mechanism over the promoted refined v3-B reference under the current threshold
    - it materially improves configured-step mechanism over the fixed-budget v2 control
    - it keeps one-step and configured-step mechanism positive on all comparison seeds
    - it avoids an obvious report-only accuracy regression relative to the promoted refined v3-B reference
    - it improves contextual gap closure relative to the `3072`-epoch same-family v2 reference beyond the promoted refined v3-B reference
    - the current recompare decision is therefore:
      - `recommended_next_move = promote_refined_v3c_as_active_reference`
  - the first fixed-budget fused-contract comparison now also exists under:
    - [outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_fused_contract_comparison](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_fused_contract_comparison)
  - that fused-contract comparison says:
    - the fused candidate is `stage05_v3c_fused_trajectory_semigroup_contract`
    - it keeps one-step and configured-step mechanism positive on all comparison seeds
    - it avoids an obvious report-only accuracy regression relative to `stage05_v3c_stronger_semigroup_weight`
    - it improves contextual gap closure relative to the `3072`-epoch same-family v2 reference beyond the active refined v3-C reference
    - it does not materially improve configured-step mechanism over `stage05_v3c_stronger_semigroup_weight` under the current threshold
    - the current fused-contract decision is therefore:
      - `recommended_next_move = keep_fusion_direction_and_refine_implementation`
  - the fixed-budget endpoint-line midpoint comparison now also exists under:
    - [outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_endpoint_line_midpoint_contract_comparison](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_endpoint_line_midpoint_contract_comparison)
  - that endpoint-line midpoint comparison says:
    - the candidate is `stage05_v3c_endpoint_line_midpoint_trajectory_contract`
    - it keeps one-step and configured-step mechanism positive on all comparison seeds
    - it directionally improves configured-step mechanism over `stage05_v3c_stronger_semigroup_weight`
    - it improves contextual gap closure relative to the `3072`-epoch same-family v2 reference beyond the active refined v3-C reference
    - it avoids an obvious report-only accuracy regression relative to `stage05_v3c_stronger_semigroup_weight`
    - it does not materially improve configured-step mechanism over `stage05_v3c_stronger_semigroup_weight` under the current threshold
    - the current endpoint-line midpoint decision is therefore:
      - `recommended_next_move = keep_endpoint_line_midpoint_direction_and_refine_implementation`
  - the fixed-budget endpoint-line continuation-blend comparison now also exists under:
    - [outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_endpoint_line_continuation_blend_contract_comparison](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_endpoint_line_continuation_blend_contract_comparison)
  - that endpoint-line continuation-blend comparison says:
    - the candidate is `stage05_v3c_endpoint_line_continuation_blend_trajectory_contract`
    - it keeps one-step and configured-step mechanism positive on all comparison seeds
    - it directionally improves configured-step mechanism over `stage05_v3c_stronger_semigroup_weight`
    - it improves contextual gap closure relative to the `3072`-epoch same-family v2 reference beyond the active refined v3-C reference
    - it avoids an obvious report-only accuracy regression relative to `stage05_v3c_stronger_semigroup_weight`
    - it does not materially improve configured-step mechanism over `stage05_v3c_stronger_semigroup_weight` under the current threshold
    - the current endpoint-line continuation-blend decision is therefore:
      - `recommended_next_move = keep_endpoint_line_continuation_blend_direction_and_refine_implementation`
  - the fixed-budget coupled defect-projection comparison now also exists under:
    - [outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_coupled_defect_projection_contract_comparison](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_coupled_defect_projection_contract_comparison)
  - that coupled defect-projection comparison says:
    - the candidate is `stage05_v3c_coupled_defect_projection_trajectory_contract`
    - it keeps one-step and configured-step mechanism positive on all comparison seeds
    - it directionally improves configured-step mechanism over `stage05_v3c_stronger_semigroup_weight`
    - it improves contextual gap closure relative to the `3072`-epoch same-family v2 reference beyond the active refined v3-C reference
    - it avoids an obvious report-only accuracy regression relative to `stage05_v3c_stronger_semigroup_weight`
    - it does not materially improve configured-step mechanism over `stage05_v3c_stronger_semigroup_weight` under the current threshold
    - the current coupled defect-projection decision is therefore:
      - `recommended_next_move = keep_coupled_defect_projection_direction_and_refine_implementation`
  - the fixed-budget precision-weighted continuation-corrector comparison now also exists under:
    - [outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_precision_weighted_continuation_corrector_contract_comparison](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_precision_weighted_continuation_corrector_contract_comparison)
  - that precision-weighted continuation-corrector comparison says:
    - the candidate is `stage05_v3c_precision_weighted_continuation_corrector_trajectory_contract`
    - it keeps one-step and configured-step mechanism positive on all comparison seeds
    - it directionally improves configured-step mechanism over `stage05_v3c_stronger_semigroup_weight`
    - it improves contextual gap closure relative to the `3072`-epoch same-family v2 reference beyond the active refined v3-C reference
    - it avoids an obvious report-only accuracy regression relative to `stage05_v3c_stronger_semigroup_weight`
    - it does not materially improve configured-step mechanism over `stage05_v3c_stronger_semigroup_weight` under the current threshold
    - its configured-step gain is weaker than the earlier `stage05_v3c_endpoint_line_continuation_blend_trajectory_contract` gain
    - the current precision-weighted continuation-corrector decision is therefore:
      - `recommended_next_move = keep_precision_weighted_continuation_corrector_direction_and_refine_implementation`
  - the current repository-level structural interpretation is now:
    - `absorb_semigroup_into_main_trajectory_contract`
    - endpoint / semigroup consistency now looks complementary to the main trajectory contract rather than like a permanently attached auxiliary-only term
    - current evidence still does not justify `refactor_main_contract_around_endpoint_semigroup_consistency`
    - no new planning-only charter is required from the current state
  - it still does not justify replacing the frozen Stage 04 bridge result on `main`

## Current Recommendation

- Keep the Stage 04 bridge result frozen on `main`.
- Do not open another package-internal Stage 04 diagnostic suite from this state.
- Do not use any current Stage 05 comparison as a claim that Stage 05 should replace the frozen bridge result on `main`.
- Keep the fixed-budget Stage 05 v2 result as the immediate Stage 05 control.
- Keep Stage 05 mechanism-first and task accuracy report-only.
- Do not continue pure same-family budget escalation from the current state.
- Do not continue pure same-family efficiency tweaking from the current state.
- Keep the fixed-budget Stage 05 v2 result as the immediate control.
- Use `stage05_v3c_stronger_semigroup_weight` as the current fixed-budget Stage 05 improvement reference.
- Treat `stage05_v3b_stronger_traj_curr_weight` as the previous comparison reference, not as the current active improvement reference.
- Treat the refined v3-C versus promoted refined v3-B promotion question as closed.
- Keep `stage05_v3c_endpoint_semigroup_consistency_contract` as the original diagnostic-only v3-C control, not as the active fixed-budget reference.
- Treat `stage05_v3c_stronger_semigroup_weight` as the promoted refined v3-C result, not as a remaining recompare candidate.
- Treat `stage05_v3c_fused_trajectory_semigroup_contract` as the first contract-consolidation candidate, not as the active fixed-budget reference.
- Treat `stage05_v3c_endpoint_line_continuation_blend_trajectory_contract` as the strongest tested predecessor refinement candidate, not as the active fixed-budget reference.
- Treat `stage05_v3c_coupled_defect_projection_trajectory_contract` as a tested coupled-refinement candidate, not as the active fixed-budget reference.
- Treat `stage05_v3c_precision_weighted_continuation_corrector_trajectory_contract` as the latest tested asymmetric continuation-corrector candidate, not as the active fixed-budget reference.
- Adopt `absorb_semigroup_into_main_trajectory_contract` as the current Stage 05 structural interpretation.
- Do not keep semigroup consistency framed as a permanently attached auxiliary-only term in future mainline work.
- Do not yet refactor the main Stage 05 contract around endpoint / semigroup consistency alone.
- Do not infer from this result alone that Stage 05 should replace the frozen Stage 04 bridge on `main`.
- No new planning-only Stage 05 charter is required by the current evidence chain.
- If Stage 05 continues, the next pass should keep `stage05_v3c_stronger_semigroup_weight` as the active reference and refine the continuation-target line above the endpoint-line midpoint scaffold, using `stage05_v3c_endpoint_line_continuation_blend_trajectory_contract` as the strongest tested same-family directional predecessor rather than defaulting to another midpoint-feedback pass.

## Reopen Conditions

Stage 04 package-internal work should be reopened only if one of these becomes true:

1. a genuinely different issue family appears from new evidence
2. the project explicitly chooses to leave the current corrective package
3. the project explicitly chooses to leave the current selector-gate contract

## Relevant Files

- Stage 04 implementation:
  - [src/pc/stage_04_incremental_bridge/fmpc_tf2.py](/e:/CodeSpace/PredictiveCoding/src/pc/stage_04_incremental_bridge/fmpc_tf2.py)
- Stage 05 probe:
  - [src/pc/stage_05_ef_core_probe/fmpc_ef_exploratory_probe.py](/e:/CodeSpace/PredictiveCoding/src/pc/stage_05_ef_core_probe/fmpc_ef_exploratory_probe.py)
- current active plan:
  - [PLANS.md](/e:/CodeSpace/PredictiveCoding/PLANS.md)
- current active validation contract:
  - [validation.md](/e:/CodeSpace/PredictiveCoding/validation.md)
- prompt-drafting context:
  - [PROMPT_CONTEXT.md](/e:/CodeSpace/PredictiveCoding/PROMPT_CONTEXT.md)
- low-context repository entry:
  - [LLM_BRIEF.md](/e:/CodeSpace/PredictiveCoding/LLM_BRIEF.md)

## Relevant Artifacts

- Stage 04 authority artifact tree:
  - [outputs/stage_04_incremental_bridge](/e:/CodeSpace/PredictiveCoding/outputs/stage_04_incremental_bridge)
- Stage 05 exploratory probe artifact:
  - [outputs/stage_05_ef_core_probe/fmpc_ef_exploratory_probe](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/fmpc_ef_exploratory_probe)
- Stage 05 comparison artifact:
  - [outputs/stage_05_ef_core_probe/frozen_bridge_vs_corrected_core_comparison](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/frozen_bridge_vs_corrected_core_comparison)
- Stage 05 v1 vs v2 comparison artifact:
  - [outputs/stage_05_ef_core_probe/corrected_residual_core_v1_vs_v2_comparison](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/corrected_residual_core_v1_vs_v2_comparison)
- Stage 05 refreshed bridge vs v2 comparison artifact:
  - [outputs/stage_05_ef_core_probe/frozen_bridge_vs_two_branch_corrected_core_comparison](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/frozen_bridge_vs_two_branch_corrected_core_comparison)
- Stage 05 v2 diagnostics artifact:
  - [outputs/stage_05_ef_core_probe/stage05_v2_diagnostics](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_diagnostics)
- Stage 05 v2 longer-training validation artifact:
  - [outputs/stage_05_ef_core_probe/stage05_v2_longer_training_validation](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_longer_training_validation)
- Stage 05 v2 budget-push validation artifact:
  - [outputs/stage_05_ef_core_probe/stage05_v2_budget_push_validation_1536_to_3072](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_budget_push_validation_1536_to_3072)
- Stage 05 v2 efficiency diagnostic artifact:
  - [outputs/stage_05_ef_core_probe/stage05_v2_efficiency_diagnostic_at_1536](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_efficiency_diagnostic_at_1536)
- Stage 05 v3-B refinement diagnostic artifact:
  - [outputs/stage_05_ef_core_probe/stage05_v3b_refinement_diagnostic](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v3b_refinement_diagnostic)
- Stage 05 refined v3-B recompare artifact:
  - [outputs/stage_05_ef_core_probe/stage05_v2_v3a_refined_v3b_fixed_budget_recompare](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_v3a_refined_v3b_fixed_budget_recompare)
- Stage 05 v3-C diagnostic-only smoke comparison artifact:
  - [outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_v3c_comparison](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_v3c_comparison)
- Stage 05 v3-C fixed-budget comparison artifact:
  - [outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_v3c_fixed_budget_comparison](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_v3c_fixed_budget_comparison)
- Stage 05 v3-C refinement diagnostic artifact:
  - [outputs/stage_05_ef_core_probe/stage05_v3c_refinement_diagnostic](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v3c_refinement_diagnostic)
- Stage 05 refined v3-C recompare artifact:
  - [outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_refined_v3c_fixed_budget_recompare](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_refined_v3c_fixed_budget_recompare)
- Stage 05 fused-contract comparison artifact:
  - [outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_fused_contract_comparison](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_fused_contract_comparison)
- Stage 05 coupled defect-projection comparison artifact:
  - [outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_coupled_defect_projection_contract_comparison](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_coupled_defect_projection_contract_comparison)
- Stage 05 precision-weighted continuation-corrector comparison artifact:
  - [outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_precision_weighted_continuation_corrector_contract_comparison](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_precision_weighted_continuation_corrector_contract_comparison)
