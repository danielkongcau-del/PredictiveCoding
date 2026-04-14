# RESULTS.md

This file keeps the current frozen results summary only.

- Historical long-form results now live in [archive/RESULTS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/RESULTS_HISTORY.md).
- Use [CURRENT_STATE.md](/e:/CodeSpace/PredictiveCoding/CURRENT_STATE.md), [PLANS.md](/e:/CodeSpace/PredictiveCoding/PLANS.md), and [validation.md](/e:/CodeSpace/PredictiveCoding/validation.md) for active FMPC decisions.

## Current Result Layers

- Phase 2 toy-benchmark freeze summary
- Phase 3 standalone real-data baseline snapshot
- FMPC Stage 04 frozen bridge snapshot
- FMPC Stage 05 exploratory probe snapshot

## Frozen Toy-Benchmark Summary

Strongest current Phase 2 evidence chain:

- Phase 2g matched PC/MLP search
- Phase 2g.1 local boundary check
- Phase 2g.1-refreshed downstream multiseed and budget studies

Current benchmark-level conclusions:

- `toy_regression`:
  - boundary-check-refined PC beats boundary-check-refined MLP on held-out test
  - PC remains ahead across the current multiseed refresh
  - extra PC inference budget is not needed to keep that lead
- `toy_sine_regression`:
  - boundary-check-refined MLP beats boundary-check-refined PC on held-out test
  - MLP remains ahead on most seeds in the current multiseed refresh
  - extra PC inference budget does not reverse that result under the refined base config

Interpretation:

- Phase 2 is closed as a toy-benchmark methodology phase
- Phase 2 conclusions are benchmark-dependent, not universal
- the current evidence does not support the claim that PC is optimizing a completely misaligned internal quantity
- the current searches are still finite and non-exhaustive

## Standalone Real-Data Baseline Snapshot

Current canonical baseline summaries on `sklearn.datasets.load_digits`:

- `digits_mlp`:
  - `best_epoch = 99`
  - `val_accuracy = 0.9111111111111111`
  - `test_accuracy = 0.9481481481481482`
- `digits_pc`:
  - `best_epoch = 55`
  - `val_accuracy = 0.8444444444444444`
  - `test_accuracy = 0.9185185185185185`

Current interpretation:

- this is a standalone baseline snapshot, not a completed real-data PC-vs-MLP comparison
- matched real-data tuning and multiseed aggregation are still not implemented

Relevant artifacts:

- `outputs/digits_mlp/`
- `outputs/digits_pc/`
- `outputs/digits_baselines/`
- optional retained reference: `outputs/digits_pc_stabilization/`

## FMPC Stage 04 Frozen Bridge Snapshot

Current adopted bridge default on `main`:

- `tf2_corrective_transport_terminal_angleclip_default`

Current status:

- FMPC Stage 04 Incremental Bridge is frozen as the bridge result on `main`
- the corrective bridge package is treated as locally saturated under the current selector-gate contract
- package-internal Stage 04 digging should stay closed unless genuinely new evidence appears or the project explicitly leaves the current package or contract

Relevant artifacts:

- `outputs/stage_04_incremental_bridge/`

## FMPC Stage 05 Exploratory Probe Snapshot

Current probe:

- `src/pc/stage_05_ef_core_probe/fmpc_ef_exploratory_probe.py`
- current implemented contracts:
  - corrected residual MeanFlow v1
  - corrected residual MeanFlow v2 two-branch residual core
  - bootstrap residual supervision
  - corrected residual identity curriculum

Current mechanism-level result:

- current Stage 05 v2 multiseed snapshot:
  - mean one-step validation energy delta vs identity:
    - `-0.00017276987415493217`
  - mean configured two-step validation energy delta vs identity:
    - `-0.0001763621381912032`
  - mean configured two-step validation fixed-point residual delta vs identity:
    - `-9.455861891589161e-07`
  - mean validation accuracy:
    - `0.27037037037037037`
  - mean test accuracy:
    - `0.27283950617283953`

Current interpretation:

- the current Stage 05 v2 two-branch corrected residual MeanFlow probe has positive mechanism signal
- it improves mechanism magnitude over the Stage 05 v1 baseline under the current multiseed rule
- task accuracy is still report-only and remains well below the frozen Stage 04 bridge result
- the dedicated Stage 05 v2 diagnostics also show:
  - all comparison seeds select the final training epoch
  - configured-step mechanism metrics are still improving at the boundary
  - validation accuracy is also still improving at the boundary
  - the selected epoch is already the highest validation-accuracy epoch on every seed
  - the current narrow diagnosis is `likely_undertrained`

Relevant artifacts:

- `outputs/stage_05_ef_core_probe/fmpc_ef_exploratory_probe/`
- `outputs/stage_05_ef_core_probe/corrected_residual_core_v1_vs_v2_comparison/`
- `outputs/stage_05_ef_core_probe/stage05_v2_diagnostics/`

## FMPC Stage 05 Budget-Push Validation Snapshot

Current formal validation:

- `outputs/stage_05_ef_core_probe/stage05_v2_budget_push_validation_1536_to_3072/`

Current comparison result:

- current Stage 05 v2 reference budget:
  - mean configured-step validation energy delta vs identity:
    - `-0.004980554368336933`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-1.8451788412853327e-05`
  - mean validation accuracy:
    - `0.8876543209876543`
  - mean test accuracy:
    - `0.8876543209876543`
  - selected epoch:
    - `1536 / 1536` on every seed
- stronger same-family Stage 05 v2 budget:
  - mean configured-step validation energy delta vs identity:
    - `-0.006199075439848138`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-2.8942715605407296e-05`
  - mean validation accuracy:
    - `0.908641975308642`
  - mean test accuracy:
    - `0.9160493827160495`
  - selected epoch:
    - `3072 / 3072` on every seed

Current interpretation:

- the stronger same-family Stage 05 v2 budget materially improves configured-step mechanism magnitude
- the stronger same-family budget also materially improves report-only accuracy
- the stronger budget still selects the final training epoch on every seed
- the explicit stop-rule layer still says:
  - `budget_line_still_looks_boundary_limited = true`
  - `budget_line_should_continue = true`
  - `budget_line_should_stop_and_open_v3 = false`
  - `budget_line_interpretation = boundary_limited_mechanism_prototype`
- the diagnostic contextual note now says the stronger Stage 05 v2 budget is:
  - above the frozen Stage 04 bridge accuracy level
  - mixed relative to the standalone `digits_pc` baseline
  - below the standalone `digits_mlp` baseline
- that budget result remains important as contextual same-family evidence, but it is no longer the active next move after the efficiency check below

## FMPC Stage 05 Efficiency Diagnostic Snapshot

Current formal validation:

- `outputs/stage_05_ef_core_probe/stage05_v2_efficiency_diagnostic_at_1536/`

Current comparison result:

- current `1536`-epoch Stage 05 v2 default:
  - mean configured-step validation energy delta vs identity:
    - `-0.004980554368336933`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-1.8451788412853327e-05`
  - mean validation accuracy:
    - `0.8876543209876543`
  - mean test accuracy:
    - `0.8876543209876543`
- tested optimized `1536`-epoch Stage 05 v2 candidate:
  - tested axes:
    - `lambda_id_warmup_epochs = 1`
    - `lambda_id_ramp_epochs = 1`
  - mean configured-step validation energy delta vs identity:
    - `-0.004984766983336293`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-1.8460725447955813e-05`
  - mean validation accuracy:
    - `0.8876543209876543`
  - mean test accuracy:
    - `0.8876543209876543`
- contextual `3072`-epoch Stage 05 v2 reference:
  - mean configured-step validation energy delta vs identity:
    - `-0.006199075439848138`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-2.8942715605407296e-05`
  - mean validation accuracy:
    - `0.908641975308642`
  - mean test accuracy:
    - `0.9160493827160495`

Current interpretation:

- the tested same-family schedule change does not materially improve configured-step mechanism
- it does not materially improve report-only accuracy
- it does not materially narrow the gap to the contextual `3072`-epoch reference
- the optimized `1536`-epoch candidate still selects the final training epoch on every seed
- the next move therefore becomes:
  - stop pure same-family budget escalation from the current state
  - open a true Stage 05 v3 mechanism charter
  - do not treat this as any claim that Stage 05 replaces the frozen Stage 04 bridge result on `main`

## FMPC Stage 05 Frozen-Bridge Comparison Snapshot

Current formal comparison:

- `outputs/stage_05_ef_core_probe/frozen_bridge_vs_two_branch_corrected_core_comparison/`

Current comparison result:

- Stage 04 frozen bridge:
  - mean one-step validation energy delta vs identity:
    - `0.0003669573859489221`
  - mean configured-step validation energy delta vs identity:
    - `-0.004070538730777469`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-1.8072299104931235e-05`
  - mean validation accuracy:
    - `0.811111111111111`
  - mean test accuracy:
    - `0.8197530864197531`
- Stage 05 two-branch corrected residual core v2:
  - mean one-step validation energy delta vs identity:
    - `-0.00017276987415493217`
  - mean configured-step validation energy delta vs identity:
    - `-0.0001763621381912032`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-9.455861891589161e-07`
  - mean validation accuracy:
    - `0.27037037037037037`
  - mean test accuracy:
    - `0.27283950617283953`

Current interpretation:

- Stage 05 v2 clears the refreshed multiseed mechanism-first exploration rule
- Stage 05 v2 is stronger than the frozen bridge on one-step mechanism
- Stage 05 v2 remains weaker than the frozen bridge on configured-step mechanism magnitude
- Stage 05 v2 remains far below the frozen bridge on report-only accuracy
- Stage 05 v2 does not replace the frozen bridge result on `main`
- the refreshed comparison supports using Stage 05 v2 as the new exploratory reference
- the newer v2 diagnostics point first to training / budget limitation on the v2 reference, not to a selection-rule artifact

Relevant artifacts:

- `outputs/stage_05_ef_core_probe/frozen_bridge_vs_two_branch_corrected_core_comparison/`

## FMPC Stage 05 V1 vs V2 Comparison Snapshot

Current formal comparison:

- `outputs/stage_05_ef_core_probe/corrected_residual_core_v1_vs_v2_comparison/`

Current comparison result:

- Stage 05 corrected residual core v1:
  - mean one-step validation energy delta vs identity:
    - `-0.00015049783324024477`
  - mean configured-step validation energy delta vs identity:
    - `-0.00015441938077428072`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-8.351992466021287e-07`
  - mean validation accuracy:
    - `0.2740740740740741`
  - mean test accuracy:
    - `0.2716049382716049`
- Stage 05 corrected residual core v2:
  - mean one-step validation energy delta vs identity:
    - `-0.00017276987415493217`
  - mean configured-step validation energy delta vs identity:
    - `-0.0001763621381912032`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-9.455861891589161e-07`
  - mean validation accuracy:
    - `0.27037037037037037`
  - mean test accuracy:
    - `0.27283950617283953`

Current interpretation:

- Stage 05 v2 improves mean configured-step mechanism magnitude over v1
- Stage 05 v2 keeps all required multiseed mechanism-first checks negative vs identity
- task accuracy remains report-only in this comparison
- this comparison does not claim that Stage 05 replaces the frozen Stage 04 bridge result on `main`

Relevant artifacts:

- `outputs/stage_05_ef_core_probe/corrected_residual_core_v1_vs_v2_comparison/`

## FMPC Stage 05 V2 vs V3-A vs V3-B Fixed-Budget Snapshot

Current formal comparison:

- `outputs/stage_05_ef_core_probe/stage05_v2_v3a_v3b_fixed_budget_comparison/`

Current comparison result:

- fixed-budget Stage 05 v2 control:
  - mean configured-step validation energy delta vs identity:
    - `-0.004980554368336933`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-1.8451788412853327e-05`
  - mean validation accuracy:
    - `0.8876543209876543`
  - mean test accuracy:
    - `0.8876543209876543`
- fixed-budget Stage 05 v3-A reference:
  - mean configured-step validation energy delta vs identity:
    - `-0.005430995679135288`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-2.0295392111085136e-05`
  - mean validation accuracy:
    - `0.8814814814814814`
  - mean test accuracy:
    - `0.8814814814814814`
- fixed-budget Stage 05 v3-B candidate:
  - mean configured-step validation energy delta vs identity:
    - `-0.00558439785454067`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-2.099919902933015e-05`
  - mean validation accuracy:
    - `0.8790123456790123`
  - mean test accuracy:
    - `0.880246913580247`

Current interpretation:

- the fixed-budget v3-B candidate materially improves configured-step mechanism over the fixed-budget v2 control
- the fixed-budget v3-B candidate directionally improves configured-step mechanism over the fixed-budget v3-A reference
- the fixed-budget v3-B candidate does not materially improve configured-step mechanism over the fixed-budget v3-A reference under the current threshold
- the fixed-budget v3-B candidate keeps one-step mechanism positive and avoids an obvious report-only accuracy regression relative to v3-A
- the current decision is therefore:
  - keep the fixed-budget v3-A result as the active Stage 05 improvement reference
  - do not open a v3-C charter from the current state on the basis of the present fixed-budget v3-B result

## FMPC Stage 05 V3-B Refinement Diagnostic Snapshot

Current formal comparison:

- `outputs/stage_05_ef_core_probe/stage05_v3b_refinement_diagnostic/`

Current comparison result:

- strongest tested refinement:
  - `stage05_v3b_stronger_traj_curr_weight`
- fixed-budget Stage 05 v3-B control:
  - mean configured-step validation energy delta vs identity:
    - `-0.00558439785454067`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-2.099919902933015e-05`
- strongest refined Stage 05 v3-B candidate:
  - mean configured-step validation energy delta vs identity:
    - `-0.005720360383603999`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-2.1608872426289415e-05`
  - mean validation accuracy:
    - `0.8827160493827161`
  - mean test accuracy:
    - `0.882716049382716`

Current interpretation:

- the strongest refined v3-B candidate materially improves configured-step mechanism over the fixed-budget v3-A reference under the current threshold
- it does not materially improve configured-step mechanism over the original v3-B control under the same threshold
- it keeps one-step mechanism positive and avoids an obvious report-only accuracy regression relative to v3-A
- the current decision is therefore:
  - keep the fixed-budget v3-A result as the active Stage 05 improvement reference until recompare completes
  - promote `stage05_v3b_stronger_traj_curr_weight` as the immediate refined v3-B recompare candidate
  - do not open a v3-C charter before that refined fixed-budget recompare

## FMPC Stage 05 Refined V3-B Recompare Snapshot

Current formal comparison:

- `outputs/stage_05_ef_core_probe/stage05_v2_v3a_refined_v3b_fixed_budget_recompare/`

Current comparison result:

- fixed-budget Stage 05 v2 control:
  - mean configured-step validation energy delta vs identity:
    - `-0.004980554368336933`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-1.8451788412853327e-05`
  - mean validation accuracy:
    - `0.8876543209876543`
  - mean test accuracy:
    - `0.8876543209876543`
- fixed-budget Stage 05 v3-A reference:
  - mean configured-step validation energy delta vs identity:
    - `-0.005430995679135288`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-2.0295392111085136e-05`
  - mean validation accuracy:
    - `0.8814814814814814`
  - mean test accuracy:
    - `0.8814814814814814`
- promoted refined v3-B candidate:
  - `stage05_v3b_stronger_traj_curr_weight`
  - mean configured-step validation energy delta vs identity:
    - `-0.005720360383603999`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-2.1608872426289415e-05`
  - mean validation accuracy:
    - `0.8827160493827161`
  - mean test accuracy:
    - `0.882716049382716`

Current interpretation:

- the promoted refined v3-B candidate materially improves configured-step mechanism over the fixed-budget v3-A reference under the current threshold
- it materially improves configured-step mechanism over the fixed-budget v2 control
- it keeps one-step mechanism positive and avoids an obvious report-only accuracy regression relative to v3-A
- the current decision is therefore:
  - promote `stage05_v3b_stronger_traj_curr_weight` as the active fixed-budget Stage 05 improvement reference
  - keep the fixed-budget v2 result as the immediate control
  - do not treat this as a replacement claim against the frozen Stage 04 bridge on `main`

## FMPC Stage 05 V2 vs Promoted V3-B vs V3-C Fixed-Budget Snapshot

Current formal comparison:

- `outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_v3c_fixed_budget_comparison/`

Current comparison result:

- fixed-budget Stage 05 v2 control:
  - mean configured-step validation energy delta vs identity:
    - `-0.004980554368336933`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-1.8451788412853327e-05`
  - mean validation accuracy:
    - `0.8876543209876543`
  - mean test accuracy:
    - `0.8876543209876543`
- promoted refined v3-B reference:
  - `stage05_v3b_stronger_traj_curr_weight`
  - mean configured-step validation energy delta vs identity:
    - `-0.005720360383603999`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-2.1608872426289415e-05`
  - mean validation accuracy:
    - `0.8827160493827161`
  - mean test accuracy:
    - `0.882716049382716`
- diagnostic-only v3-C candidate:
  - `stage05_v3c_endpoint_semigroup_consistency_contract`
  - mean configured-step validation energy delta vs identity:
    - `-0.0058467458070066425`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-2.2163570572707745e-05`
  - mean validation accuracy:
    - `0.8814814814814814`
  - mean test accuracy:
    - `0.882716049382716`

Current interpretation:

- the diagnostic-only v3-C candidate directionally improves configured-step mechanism over the promoted refined v3-B reference
- it also improves contextual gap closure relative to the `3072`-epoch same-family v2 reference
- it keeps one-step mechanism positive and avoids an obvious report-only accuracy regression relative to the promoted refined v3-B reference
- it does not materially improve configured-step mechanism over the promoted refined v3-B reference under the current threshold
- the current decision is therefore:
  - keep `stage05_v3b_stronger_traj_curr_weight` as the active fixed-budget Stage 05 improvement reference
  - keep `stage05_v3c_endpoint_semigroup_consistency_contract` diagnostic-only
  - use a narrow v3-C refinement pass as the next move rather than immediate promotion

## FMPC Stage 05 V3-C Refinement Diagnostic Snapshot

Current formal comparison:

- `outputs/stage_05_ef_core_probe/stage05_v3c_refinement_diagnostic/`

Current compared candidates:

- fixed-budget immediate control:
  - `stage_05_two_branch_corrected_residual_core_v2`
- promoted refined v3-B reference:
  - `stage05_v3b_stronger_traj_curr_weight`
- current v3-C control:
  - `stage05_v3c_endpoint_semigroup_consistency_contract`
- strongest refined v3-C candidate:
  - `stage05_v3c_stronger_semigroup_weight`

Current configured-step mechanism means:

- promoted refined v3-B energy delta vs identity:
  - `-0.005720360383603999`
- current v3-C control energy delta vs identity:
  - `-0.0058467458070066425`
- refined v3-C energy delta vs identity:
  - `-0.0059706216916698045`
- promoted refined v3-B fixed-point residual delta vs identity:
  - `-2.1608872426289415e-05`
- current v3-C control fixed-point residual delta vs identity:
  - `-2.2163570572707745e-05`
- refined v3-C fixed-point residual delta vs identity:
  - `-2.2701777366495727e-05`

Current interpretation:

- the stronger semigroup-weight refinement continues to move configured-step mechanism in the favorable direction
- it materially improves configured-step mechanism over the promoted refined v3-B reference under the current threshold
- it does not materially improve configured-step mechanism over the current v3-C control under the same threshold
- it keeps one-step and configured-step mechanism positive
- it avoids an obvious report-only accuracy regression

- the current decision is therefore:
  - keep `stage05_v3b_stronger_traj_curr_weight` as the active fixed-budget Stage 05 improvement reference only until the fresh recompare completes
  - promote `stage05_v3c_stronger_semigroup_weight` as the refined v3-C formal comparison candidate

## FMPC Stage 05 Refined V3-C Recompare Snapshot

Current formal comparison:

- `outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_refined_v3c_fixed_budget_recompare/`

Current comparison result:

- fixed-budget Stage 05 v2 control:
  - mean configured-step validation energy delta vs identity:
    - `-0.004980554368336933`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-1.8451788412853327e-05`
  - mean validation accuracy:
    - `0.8876543209876543`
  - mean test accuracy:
    - `0.8876543209876543`
- promoted refined v3-B reference:
  - `stage05_v3b_stronger_traj_curr_weight`
  - mean configured-step validation energy delta vs identity:
    - `-0.005720360383603999`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-2.1608872426289415e-05`
  - mean validation accuracy:
    - `0.8827160493827161`
  - mean test accuracy:
    - `0.882716049382716`
- refined v3-C formal comparison candidate:
  - `stage05_v3c_stronger_semigroup_weight`
  - mean configured-step validation energy delta vs identity:
    - `-0.0059706216916698045`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-2.2701777366495727e-05`
  - mean validation accuracy:
    - `0.880246913580247`
  - mean test accuracy:
    - `0.8839506172839506`

Current interpretation:

- the refined v3-C candidate materially improves configured-step mechanism over the promoted refined v3-B reference under the current threshold
- it also materially improves configured-step mechanism over the fixed-budget v2 control
- it keeps one-step mechanism positive and avoids an obvious report-only accuracy regression relative to the promoted refined v3-B reference
- it improves contextual gap closure relative to the `3072`-epoch same-family v2 reference beyond the promoted refined v3-B reference
- the current decision is therefore:
  - promote `stage05_v3c_stronger_semigroup_weight` as the active fixed-budget Stage 05 improvement reference
  - treat `stage05_v3b_stronger_traj_curr_weight` as the previous fixed-budget comparison reference
  - do not treat this as any claim that Stage 05 replaces the frozen Stage 04 bridge result on `main`

## First Exact-Fusion Consolidation Comparison

Artifact:

- `outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_fused_contract_comparison/`

Compared candidates:

- active refined v3-C reference:
  - `stage05_v3c_stronger_semigroup_weight`
  - mean configured-step validation energy delta vs identity:
    - `-0.0059706216916698045`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-2.2701777366495727e-05`
  - mean validation accuracy:
    - `0.880246913580247`
  - mean test accuracy:
    - `0.8839506172839506`
- exact-fusion consolidation candidate:
  - `stage05_v3c_fused_trajectory_semigroup_contract`
  - mean configured-step validation energy delta vs identity:
    - `-0.005970657762441119`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-2.2701930872562526e-05`
  - mean validation accuracy:
    - `0.880246913580247`
  - mean test accuracy:
    - `0.8839506172839506`

Current interpretation:

- the fused candidate preserves the current mechanism-first gate and avoids an obvious report-only accuracy regression
- it shows a positive but very small contextual gap-closure movement beyond the active refined v3-C reference
- it does not materially improve configured-step mechanism over `stage05_v3c_stronger_semigroup_weight` under the current threshold
- the current decision is therefore:
  - keep `stage05_v3c_stronger_semigroup_weight` as the active fixed-budget Stage 05 improvement reference
  - keep `stage05_v3c_fused_trajectory_semigroup_contract` as the first consolidation candidate rather than promoting it
  - refine the fusion direction rather than reopening promotion, budget escalation, or a new charter

## Endpoint-Line Midpoint Comparison

Artifact:

- `outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_endpoint_line_midpoint_contract_comparison/`

Compared candidates:

- active refined v3-C reference:
  - `stage05_v3c_stronger_semigroup_weight`
  - mean configured-step validation energy delta vs identity:
    - `-0.0059706216916698045`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-2.2701777366495727e-05`
  - mean validation accuracy:
    - `0.880246913580247`
  - mean test accuracy:
    - `0.8839506172839506`
- endpoint-line midpoint candidate:
  - `stage05_v3c_endpoint_line_midpoint_trajectory_contract`
  - mean configured-step validation energy delta vs identity:
    - `-0.006015732548759099`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-2.287737578720965e-05`
  - mean validation accuracy:
    - `0.8790123456790123`
  - mean test accuracy:
    - `0.8814814814814814`

Current interpretation:

- the endpoint-line midpoint candidate preserves the current mechanism-first gate and avoids an obvious report-only accuracy regression
- it directionally improves configured-step mechanism over `stage05_v3c_stronger_semigroup_weight`
- it improves contextual gap closure beyond the active refined v3-C reference
- it does not materially improve configured-step mechanism over `stage05_v3c_stronger_semigroup_weight` under the current threshold
- the current decision is therefore:
  - keep `stage05_v3c_stronger_semigroup_weight` as the active fixed-budget Stage 05 improvement reference
  - keep `stage05_v3c_endpoint_line_midpoint_trajectory_contract` as the current narrow refinement candidate rather than promoting it
  - continue the endpoint-line midpoint direction rather than reopening promotion or budget questions

## Stage 05 Endpoint-Line Continuation-Blend Fixed-Budget Comparison

Artifact:

- `outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_endpoint_line_continuation_blend_contract_comparison/`

Compared candidates:

- active refined v3-C reference:
  - `stage05_v3c_stronger_semigroup_weight`
  - mean configured-step validation energy delta vs identity:
    - `-0.0059706216916698045`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-2.2701777366495727e-05`
  - mean validation accuracy:
    - `0.880246913580247`
  - mean test accuracy:
    - `0.8839506172839506`
- endpoint-line continuation-blend candidate:
  - `stage05_v3c_endpoint_line_continuation_blend_trajectory_contract`
  - mean configured-step validation energy delta vs identity:
    - `-0.006077107269951781`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-2.3154554470203098e-05`
  - mean validation accuracy:
    - `0.880246913580247`
  - mean test accuracy:
    - `0.8814814814814814`

Current interpretation:

- the endpoint-line continuation-blend candidate preserves the current mechanism-first gate and avoids an obvious report-only accuracy regression
- it directionally improves configured-step mechanism over `stage05_v3c_stronger_semigroup_weight`
- it improves contextual gap closure beyond the active refined v3-C reference
- it does not materially improve configured-step mechanism over `stage05_v3c_stronger_semigroup_weight` under the current threshold
- the current decision is therefore:
  - keep `stage05_v3c_stronger_semigroup_weight` as the active fixed-budget Stage 05 improvement reference
  - keep `stage05_v3c_endpoint_line_continuation_blend_trajectory_contract` as the current narrow refinement candidate rather than promoting it
  - continue the continuation-target refinement direction rather than reopening promotion or budget questions

## Stage 05 Coupled Defect-Projection Fixed-Budget Comparison

Artifact:

- `outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_coupled_defect_projection_contract_comparison/`

Compared candidates:

- active refined v3-C reference:
  - `stage05_v3c_stronger_semigroup_weight`
  - mean configured-step validation energy delta vs identity:
    - `-0.0059706216916698045`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-2.2701777366495727e-05`
  - mean validation accuracy:
    - `0.880246913580247`
  - mean test accuracy:
    - `0.8839506172839506`
- coupled defect-projection candidate:
  - `stage05_v3c_coupled_defect_projection_trajectory_contract`
  - mean configured-step validation energy delta vs identity:
    - `-0.006064637700758028`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-2.309891909200473e-05`
  - mean validation accuracy:
    - `0.880246913580247`
  - mean test accuracy:
    - `0.882716049382716`

Current interpretation:

- the coupled defect-projection candidate preserves the current mechanism-first gate and avoids an obvious report-only accuracy regression
- it directionally improves configured-step mechanism over `stage05_v3c_stronger_semigroup_weight`
- it improves contextual gap closure beyond the active refined v3-C reference
- it does not materially improve configured-step mechanism over `stage05_v3c_stronger_semigroup_weight` under the current threshold
- the current decision is therefore:
  - keep `stage05_v3c_stronger_semigroup_weight` as the active fixed-budget Stage 05 improvement reference
  - keep `stage05_v3c_coupled_defect_projection_trajectory_contract` as the current narrow refinement candidate rather than promoting it
  - continue the coupled local defect-projection direction rather than reopening promotion or budget questions

## Stage 05 Precision-Weighted Continuation-Corrector Fixed-Budget Comparison

Artifact:

- `outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_precision_weighted_continuation_corrector_contract_comparison/`

Compared candidates:

- active refined v3-C reference:
  - `stage05_v3c_stronger_semigroup_weight`
  - mean configured-step validation energy delta vs identity:
    - `-0.0059706216916698045`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-2.2701777366495727e-05`
  - mean validation accuracy:
    - `0.880246913580247`
  - mean test accuracy:
    - `0.8839506172839506`
- precision-weighted continuation-corrector candidate:
  - `stage05_v3c_precision_weighted_continuation_corrector_trajectory_contract`
  - mean configured-step validation energy delta vs identity:
    - `-0.006033525050064575`
  - mean configured-step validation fixed-point residual delta vs identity:
    - `-2.2958570229293478e-05`
  - mean validation accuracy:
    - `0.8790123456790123`
  - mean test accuracy:
    - `0.8814814814814814`

Pairwise deltas versus active refined v3-C:

- configured-step validation energy delta vs identity:
  - `-6.290335839477061e-05`
- configured-step validation fixed-point residual delta vs identity:
  - `-2.5679286279775153e-07`
- validation accuracy delta:
  - `-0.0012345679012345883`
- test accuracy delta:
  - `-0.0024691358024691392`

Decision:

- it keeps one-step and configured-step mechanism positive on all comparison seeds
- it directionally improves configured-step mechanism over `stage05_v3c_stronger_semigroup_weight`
- it improves contextual gap closure relative to the `3072`-epoch same-family v2 reference beyond the active refined v3-C reference
- it avoids an obvious report-only accuracy regression relative to `stage05_v3c_stronger_semigroup_weight`
- it does not materially improve configured-step mechanism over `stage05_v3c_stronger_semigroup_weight` under the current threshold
- its configured-step gain is weaker than the earlier `stage05_v3c_endpoint_line_continuation_blend_trajectory_contract` result
- the current decision is therefore:
  - keep `stage05_v3c_stronger_semigroup_weight` as the active fixed-budget Stage 05 improvement reference
  - treat `stage05_v3c_precision_weighted_continuation_corrector_trajectory_contract` as a tested asymmetric continuation-corrector variant rather than promoting it
  - continue the continuation-target refinement direction, but do not treat this exact `eta_cont` weighting as the new same-family leader

## Where Detailed History Went

Use these files for longer historical context:

- historical Phase 2 and early real-data narrative:
  - [archive/RESULTS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/RESULTS_HISTORY.md)
- historical plan chain:
  - [archive/PLANS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/PLANS_HISTORY.md)
- historical validation chain:
  - [archive/validation_history.md](/e:/CodeSpace/PredictiveCoding/archive/validation_history.md)
