# Validation and Acceptance Criteria

This file now keeps the active validation contract only.

- Historical validation detail has moved to [archive/validation_history.md](/e:/CodeSpace/PredictiveCoding/archive/validation_history.md).
- The goal of this file is to preserve current acceptance rules without forcing every new session to read the full historical evidence chain.

## General Validation Philosophy

A valid change should still demonstrate:

1. shape correctness
2. numerical sanity
3. behavioral sanity
4. reproducibility
5. traceability to the math layer rooted at `spec_math.md`

## Stage 04 Validation Status

The current Stage 04 corrective bridge package is frozen as the bridge result on `main`.

Current control:

- `tf2_corrective_transport_terminal_angleclip_default`

Current validation interpretation:

- the adopted corrective bridge package is locally saturated under the current selector-gate contract
- no different package-internal issue family is currently recommended for pursuit
- package-internal Stage 04 digging should stop from this state unless:
  - a genuinely different issue family appears from new evidence, or
  - the project explicitly decides to leave the current package or selector-gate contract

### Stage 04 Sealed Families

The following families are sealed from the current state:

- terminal / unified-cone follow-ups
- successor-value and successor-increment package-internal follow-ups
- readout alignment follow-ups
- bootstrap-source bias follow-ups
- target-lag follow-ups
- bootstrap-to-identity curriculum follow-ups

### Why Stage 04 Is Closed

Condensed evidence summary:

- the current hard full-vector `30` degree terminal angle clip remains the local winner
- selector / checkpoint is not the main limiter
- simple head-fit / separability is not the main limiter
- readout alignment is an exact no-op on the adopted package
- older bootstrap-target-side follow-ups do not materially improve the adopted package
- the late-rollout successor-value and successor-increment line ends in a strengthened formulation-level blocker rather than an adoption-viable local fix

## Stage 05 Exploratory Validation Contract

Stage 05 is mechanism-first.

Current Stage 05 core contract:

- corrected residual MeanFlow transport family
- bootstrap residual supervision
- corrected residual identity curriculum
- no teacher dependency in target construction

Current Stage 05 implementation status:

- the v1 single-branch corrected residual core remains the backward-compatible baseline
- the v2 two-branch corrected residual core now also exists
- the v2 branch structure is:
  - `m_psi = m_traj + m_state`
  - `m_traj_input = concat([z_t, target_onehot, t, r])`
  - `m_state_input = concat([g_t, e_out_t, F_t])`

Primary acceptance metrics:

- one-step energy decrease relative to identity or no-transport
- few-step fixed-point residual decrease relative to identity or no-transport
- deterministic artifact generation
- artifact-independent target construction

Secondary report-only metrics:

- validation accuracy
- test accuracy

What this means operationally:

- task accuracy is not the gate for the current exploratory stage
- a weak task metric does not invalidate a probe if mechanism metrics are positive
- a strong task metric alone does not justify promotion without a clear mechanism advantage

## Stage 05 Required Artifacts

Current canonical exploratory runs should produce:

- `config.json`
- `epoch_metrics.csv`
- `summary.json`

Summary fields should continue to make these items explicit:

- stage identity
- transport family
- residual identity mode
- whether target construction is artifact-independent
- local flow definition
- direct anchor source
- transport scope
- transport steps
- bootstrap target contract
- selection-metric source
- report-metric source
- mechanism metrics such as energy and residual deltas

## Current Stage 05 Known Result

The first canonical Stage 05 exploratory probe now shows:

- positive one-step validation energy delta vs identity
- positive configured two-step validation energy delta vs identity
- positive configured two-step fixed-point residual delta vs identity
- deterministic artifact generation
- artifact-independent target construction

At the same time:

- validation and test accuracy remain low
- the probe does not challenge the frozen Stage 04 bridge result yet

Current interpretation:

- the exploratory line has now cleared the narrow v1 to v2 mechanism-improvement check
- the two-branch Stage 05 v2 core is now the current exploratory candidate
- the refreshed frozen-bridge comparison also supports treating Stage 05 v2 as the new exploratory reference
- the dedicated Stage 05 v2 diagnostics now also say:
  - all comparison seeds select the final training epoch
  - configured-step mechanism metrics and validation accuracy still improve at the training boundary
  - the current low report-only accuracy is not primarily a selection-rule artifact
  - the current narrow diagnosis is `likely_undertrained`
- the dedicated Stage 05 v2 longer-training validation now also says:
  - a stronger same-family `24`-epoch budget materially improves configured-step mechanism magnitude
  - the same stronger budget materially improves report-only validation and test accuracy
  - the stronger budget still selects the final training epoch on every seed
- the next Stage 05 v2 budget-push validation now also says:
  - a stronger same-family `3072`-epoch budget materially improves configured-step mechanism magnitude over the `1536`-epoch reference
  - the same stronger budget materially improves report-only validation and test accuracy
  - the stronger budget still selects the final training epoch on every seed
  - the explicit stop-rule layer still says:
    - `budget_line_still_looks_boundary_limited = true`
    - `budget_line_should_continue = true`
    - `budget_line_should_stop_and_open_v3 = false`
    - `budget_line_interpretation = boundary_limited_mechanism_prototype`
  - the contextual accuracy note now places the stronger Stage 05 v2 budget:
    - above the frozen Stage 04 bridge accuracy level
    - mixed relative to the standalone `digits_pc` baseline
    - below the standalone `digits_mlp` baseline
- the Stage 05 v2 efficiency diagnostic at the fixed `1536`-epoch ceiling now also says:
  - the tested same-family schedule change does not materially improve configured-step mechanism over the current `1536`-epoch default
  - it does not materially improve report-only validation and test accuracy
  - it does not materially narrow the gap to the contextual `3072`-epoch reference
  - the optimized `1536`-epoch candidate still selects the final training epoch on every seed
  - the current decision is therefore:
    - `recommended_next_move = implement_stage05_v3a_candidate`
- the current Stage 05 v3-A charter is now motivated by the working hypothesis that the current residual target may entangle transport residual and anchor-drift residual too tightly, limiting configured-step efficiency
- the first minimal v3-A candidate codepath and smoke comparison entry now exist in-repo
- the first real fixed-budget `v2 vs v3-A` comparison now also says:
  - the fixed-budget v3-A candidate materially improves configured-step mechanism over the `1536`-epoch v2 reference
  - it keeps one-step mechanism positive on every comparison seed
  - it shows a positive gap-closure signal relative to the contextual `3072`-epoch v2 reference
  - it avoids an obvious report-only accuracy regression under the current Stage 05 rule, although validation and test accuracy remain slightly below the fixed-budget v2 reference
  - the current decision is therefore:
    - `recommended_next_move = proceed_to_stage05_v3b_curriculum_charter`
- the first minimal Stage 05 v3-B candidate path and smoke-ready three-way comparison entry now also exist in-repo:
  - candidate name: `stage05_v3b_trajectory_curriculum_contract`
  - the candidate keeps the v3-A explicit transport-drift decomposition and adds one aggregate trajectory curriculum loss
  - the first real fixed-budget `v2 vs v3-A vs v3-B` comparison now also exists under:
    - `outputs/stage_05_ef_core_probe/stage05_v2_v3a_v3b_fixed_budget_comparison/`
  - that fixed-budget comparison says:
    - the v3-B candidate materially improves configured-step mechanism over the fixed-budget v2 control
    - the v3-B candidate directionally improves configured-step mechanism over the fixed-budget v3-A reference
    - the v3-B candidate does not materially improve configured-step mechanism over the fixed-budget v3-A reference under the current threshold
    - the current decision is therefore:
      - `recommended_next_move = retain_v3a_as_active_reference`
- the narrow fixed-budget v3-B refinement diagnostic now also exists under:
  - `outputs/stage_05_ef_core_probe/stage05_v3b_refinement_diagnostic/`
  - the strongest tested refinement variant is `stage05_v3b_stronger_traj_curr_weight`
  - it materially improves configured-step mechanism over the fixed-budget v3-A reference under the current threshold
  - it does not materially improve configured-step mechanism over the original v3-B control under the same threshold
  - it avoids an obvious report-only accuracy regression relative to v3-A
  - the current refinement decision is therefore:
    - `recommended_next_move = promote_refined_v3b_and_recompare`
- the fresh fixed-budget refined v3-B recompare now also exists under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_v3a_refined_v3b_fixed_budget_recompare/`
  - the promoted refined candidate is `stage05_v3b_stronger_traj_curr_weight`
  - it materially improves configured-step mechanism over the fixed-budget v3-A reference under the current threshold
  - it materially improves configured-step mechanism over the fixed-budget v2 control
  - it avoids an obvious report-only accuracy regression relative to v3-A
  - the current recompare decision is therefore:
    - `recommended_next_move = promote_refined_v3b_as_active_reference`
- the post-promotion Stage 05 planning conclusion is now:
  - the fixed-budget `v3-A -> refined v3-B` promotion question is closed
  - current repo evidence ranks the remaining mechanism suspects as:
    - endpoint / semigroup consistency contract
    - still-insufficient trajectory curriculum contract
    - corrected residual identity contract
  - the most credible next charter is therefore a planning-only `Stage 05 v3-C`
  - `Stage 05 v3-C` is defined by the working hypothesis that the current refined v3-B scaffold still lacks an explicit endpoint / semigroup consistency contract across split horizons
- it still does not justify replacing the frozen bridge result on `main`

## Current Recommended Validation Move

The next validation move is:

- keep the fixed-budget Stage 05 v2 result as the immediate control
- treat the latest same-family budget reference as the completed `1536 -> 3072 epochs` validation under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_budget_push_validation_1536_to_3072/`
- treat the fixed-budget efficiency check as the closing same-family efficiency result under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_efficiency_diagnostic_at_1536/`
- treat the smoke-level v2 vs v3-A artifact as the implementation sanity check under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_vs_v3a_explicit_transport_drift_comparison/`
- treat the fixed-budget v2 vs v3-A comparison as the current v3-A evidence base under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_vs_v3a_explicit_transport_drift_fixed_budget_comparison/`
- treat the fixed-budget v2 vs v3-A vs v3-B comparison as the current v3-B evidence base under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_v3a_v3b_fixed_budget_comparison/`
- treat the fixed-budget v3-B refinement diagnostic as the current immediate v3-B follow-up evidence under:
  - `outputs/stage_05_ef_core_probe/stage05_v3b_refinement_diagnostic/`
- treat the fixed-budget refined v3-B recompare as the completed v3-B promotion decision under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_v3a_refined_v3b_fixed_budget_recompare/`
- treat the fixed-budget refined v3-C recompare as the current active Stage 05 improvement-reference decision under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_refined_v3c_fixed_budget_recompare/`
- do not continue pure same-family budget escalation from this state
- do not continue pure same-family efficiency tweaking from this state
- keep the fixed-budget Stage 05 v2 result as the immediate control
- keep `stage05_v3c_stronger_semigroup_weight` as the current fixed-budget Stage 05 improvement reference
- treat `stage05_v3b_stronger_traj_curr_weight` as the previous comparison reference
- treat the refined v3-C versus promoted refined v3-B promotion question as closed
- adopt `absorb_semigroup_into_main_trajectory_contract` as the current Stage 05 structural interpretation
- do not keep endpoint / semigroup consistency framed as a permanently attached auxiliary-only term
- do not yet adopt `refactor_main_contract_around_endpoint_semigroup_consistency`
- do not open a new planning-only charter by default from the current state

Any next narrower Stage 05 follow-up should preserve:

- artifact-independent target construction
- deterministic artifact generation
- the fixed-budget v2 result as the immediate control
- the mechanism-first validation contract for Stage 05
- task accuracy as a secondary signal only
- the refined v3-C scaffold as the current Stage 05 improvement reference
- the original v3-C probe as diagnostic-only control rather than the active fixed-budget reference
- the current structural interpretation that semigroup consistency should be absorbed into the main trajectory contract

Any next narrower Stage 05 follow-up should be accepted only if it:

- preserves explicit transport-drift decomposition from v3-A
- preserves the refined v3-C scaffold as the active comparison reference
- keeps artifact-independent target construction
- keeps deterministic artifact generation
- keeps one-step mechanism metrics positive under the current Stage 05 rule
- makes explicit whether semigroup consistency is absorbed into the main trajectory contract rather than appended as another auxiliary-only term
- reports explicit pairwise deltas versus:
  - the active refined Stage 05 v3-C reference
  - the fixed-budget v2 control
- makes the consolidated main trajectory-contract identity explicit
- keeps task accuracy as a secondary signal only

Any next narrower Stage 05 follow-up must minimally produce:

- a comparison run or suite against:
  - the fixed-budget Stage 05 v2 control
  - `stage05_v3c_stronger_semigroup_weight`
- a dedicated artifact directory
- aggregate summary fields that report:
  - whether endpoint / semigroup consistency is enabled
  - whether semigroup consistency is absorbed into the main trajectory contract
  - the split-horizon semigroup identity
  - pairwise deltas versus the active refined v3-C reference
  - pairwise deltas versus v2
  - a gap-closure style decision field
  - `recommended_next_move`

Current v3-C status:

- the first minimal diagnostic-only v3-C probe now exists:
  - `stage05_v3c_endpoint_semigroup_consistency_contract`
- the first smoke-ready `v2 vs promoted-v3B vs v3-C` comparison entry now also exists under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_v3c_comparison/`
- that smoke artifact should be treated only as a wiring and deterministic-artifact check
- the first real fixed-budget comparison now also exists under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_v3c_fixed_budget_comparison/`
- that fixed-budget comparison says:
  - v3-C keeps one-step and configured-step mechanism positive under the current Stage 05 gate
  - v3-C directionally improves configured-step mechanism over `stage05_v3b_stronger_traj_curr_weight`
  - v3-C shows a positive contextual gap-closure signal over the promoted refined v3-B reference
  - v3-C avoids an obvious report-only accuracy regression
  - v3-C still does not materially beat the promoted refined v3-B reference under the current threshold
- the narrow fixed-budget v3-C refinement diagnostic now also exists under:
  - `outputs/stage_05_ef_core_probe/stage05_v3c_refinement_diagnostic/`
- that refinement diagnostic says:
  - the best tested refinement variant is `stage05_v3c_stronger_semigroup_weight`
  - it materially improves configured-step mechanism over `stage05_v3b_stronger_traj_curr_weight` under the current threshold
  - it does not materially improve configured-step mechanism over the current v3-C control under the same threshold
  - it keeps one-step and configured-step mechanism positive
  - it avoids an obvious report-only accuracy regression
- the fresh fixed-budget refined v3-C recompare now also exists under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_refined_v3c_fixed_budget_recompare/`
- that refined recompare says:
  - `stage05_v3c_stronger_semigroup_weight` materially improves configured-step mechanism over `stage05_v3b_stronger_traj_curr_weight` under the current threshold
  - it materially improves configured-step mechanism over fixed-budget `v2`
  - it avoids an obvious report-only accuracy regression relative to the promoted refined v3-B reference
- the first fixed-budget fused-contract comparison now also exists under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_fused_contract_comparison/`
- that fused-contract comparison says:
  - `stage05_v3c_fused_trajectory_semigroup_contract` keeps one-step and configured-step mechanism positive
  - it avoids an obvious report-only accuracy regression relative to `stage05_v3c_stronger_semigroup_weight`
  - it improves contextual gap closure relative to the `3072`-epoch same-family reference beyond the active refined v3-C reference
  - it does not materially improve configured-step mechanism over `stage05_v3c_stronger_semigroup_weight` under the current threshold
- the fixed-budget endpoint-line midpoint comparison now also exists under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_endpoint_line_midpoint_contract_comparison/`
- that endpoint-line midpoint comparison says:
  - `stage05_v3c_endpoint_line_midpoint_trajectory_contract` keeps one-step and configured-step mechanism positive
  - it directionally improves configured-step mechanism over `stage05_v3c_stronger_semigroup_weight`
  - it improves contextual gap closure relative to the `3072`-epoch same-family reference beyond the active refined v3-C reference
  - it avoids an obvious report-only accuracy regression relative to `stage05_v3c_stronger_semigroup_weight`
  - it does not materially improve configured-step mechanism over `stage05_v3c_stronger_semigroup_weight` under the current threshold
- the current repository-level structural interpretation is now:
  - `absorb_semigroup_into_main_trajectory_contract`
  - current evidence supports semigroup consistency as part of the main trajectory contract
  - current evidence does not yet support replacing the trajectory framing with endpoint / semigroup consistency alone
- the next acceptance step is therefore:
  - keep `stage05_v3c_stronger_semigroup_weight` as the active Stage 05 improvement reference
  - keep fixed-budget `v2` as the immediate control
  - treat `stage05_v3c_endpoint_line_midpoint_trajectory_contract` as the current narrow refinement candidate rather than the active reference
  - require any next endpoint-line midpoint refinement to beat the active refined v3-C reference under the same mechanism-first rule
- the fixed-budget endpoint-line continuation-blend comparison now also exists under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_endpoint_line_continuation_blend_contract_comparison/`
- that endpoint-line continuation-blend comparison says:
  - `stage05_v3c_endpoint_line_continuation_blend_trajectory_contract` keeps one-step and configured-step mechanism positive
  - it directionally improves configured-step mechanism over `stage05_v3c_stronger_semigroup_weight`
  - it improves contextual gap closure relative to the `3072`-epoch same-family reference beyond the active refined v3-C reference
  - it avoids an obvious report-only accuracy regression relative to `stage05_v3c_stronger_semigroup_weight`
  - it does not materially improve configured-step mechanism over `stage05_v3c_stronger_semigroup_weight` under the current threshold
- the next acceptance step is therefore refined again:
  - keep `stage05_v3c_stronger_semigroup_weight` as the active Stage 05 improvement reference
  - keep fixed-budget `v2` as the immediate control
  - treat `stage05_v3c_endpoint_line_continuation_blend_trajectory_contract` as the current narrow refinement candidate rather than the active reference
  - require any next continuation-target refinement to beat the active refined v3-C reference under the same mechanism-first rule
- the fixed-budget coupled defect-projection comparison now also exists under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_coupled_defect_projection_contract_comparison/`
- that coupled defect-projection comparison says:
  - `stage05_v3c_coupled_defect_projection_trajectory_contract` keeps one-step and configured-step mechanism positive
  - it directionally improves configured-step mechanism over `stage05_v3c_stronger_semigroup_weight`
  - it improves contextual gap closure relative to the `3072`-epoch same-family reference beyond the active refined v3-C reference
  - it avoids an obvious report-only accuracy regression relative to `stage05_v3c_stronger_semigroup_weight`
  - it does not materially improve configured-step mechanism over `stage05_v3c_stronger_semigroup_weight` under the current threshold
- the next acceptance step is therefore refined again:
  - keep `stage05_v3c_stronger_semigroup_weight` as the active Stage 05 improvement reference
  - keep fixed-budget `v2` as the immediate control
  - treat `stage05_v3c_coupled_defect_projection_trajectory_contract` as the current narrow refinement candidate rather than the active reference
  - require any next coupled midpoint-continuation refinement to beat the active refined v3-C reference under the same mechanism-first rule

It should not be framed as:

- a Stage 04 package-internal reopening
- a new adoption sweep inside the saturated corrective bridge package
- another pure same-family budget escalation
- another pure same-family efficiency tweak

## Relevant Active Artifacts

- frozen Stage 04 authority artifact tree:
  - `outputs/stage_04_incremental_bridge/`
- Stage 05 exploratory probe:
  - `outputs/stage_05_ef_core_probe/fmpc_ef_exploratory_probe/`
- Stage 05 frozen-bridge comparison:
  - `outputs/stage_05_ef_core_probe/frozen_bridge_vs_corrected_core_comparison/`
- Stage 05 v1 vs v2 comparison:
  - `outputs/stage_05_ef_core_probe/corrected_residual_core_v1_vs_v2_comparison/`
- Stage 05 refreshed bridge vs v2 comparison:
  - `outputs/stage_05_ef_core_probe/frozen_bridge_vs_two_branch_corrected_core_comparison/`
- Stage 05 v2 diagnostics:
  - `outputs/stage_05_ef_core_probe/stage05_v2_diagnostics/`
- Stage 05 v2 longer-training validation:
  - `outputs/stage_05_ef_core_probe/stage05_v2_longer_training_validation/`
- Stage 05 v2 budget-push validation:
  - `outputs/stage_05_ef_core_probe/stage05_v2_budget_push_validation_1536_to_3072/`
- Stage 05 v2 efficiency diagnostic:
  - `outputs/stage_05_ef_core_probe/stage05_v2_efficiency_diagnostic_at_1536/`
- Stage 05 v2 vs v3-A fixed-budget comparison:
  - `outputs/stage_05_ef_core_probe/stage05_v2_vs_v3a_explicit_transport_drift_fixed_budget_comparison/`
- Stage 05 v3-B refinement diagnostic:
  - `outputs/stage_05_ef_core_probe/stage05_v3b_refinement_diagnostic/`
- Stage 05 refined v3-B recompare:
  - `outputs/stage_05_ef_core_probe/stage05_v2_v3a_refined_v3b_fixed_budget_recompare/`
- Stage 05 endpoint-line midpoint fixed-budget comparison:
  - `outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_endpoint_line_midpoint_contract_comparison/`
- Stage 05 endpoint-line continuation-blend fixed-budget comparison:
  - `outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_endpoint_line_continuation_blend_contract_comparison/`
- Stage 05 coupled defect-projection fixed-budget comparison:
  - `outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_coupled_defect_projection_contract_comparison/`
- Stage 05 v3-C refinement diagnostic:
  - `outputs/stage_05_ef_core_probe/stage05_v3c_refinement_diagnostic/`
- Stage 05 refined v3-C recompare:
  - `outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_refined_v3c_fixed_budget_recompare/`

## Document Layering

Use the current docs this way:

- low-context summary:
  - [LLM_BRIEF.md](/e:/CodeSpace/PredictiveCoding/LLM_BRIEF.md)
- math layer:
  - [spec_math.md](/e:/CodeSpace/PredictiveCoding/spec_math.md)
  - applicable stage addenda under [specs/](/e:/CodeSpace/PredictiveCoding/specs)
- operational state:
  - [CURRENT_STATE.md](/e:/CodeSpace/PredictiveCoding/CURRENT_STATE.md)
- active plan:
  - [PLANS.md](/e:/CodeSpace/PredictiveCoding/PLANS.md)
- historical validation detail:
  - [archive/validation_history.md](/e:/CodeSpace/PredictiveCoding/archive/validation_history.md)
