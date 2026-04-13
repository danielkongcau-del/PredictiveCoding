# PLANS.md

This file tracks the current forward plan only.

- Historical plan detail has moved to [archive/PLANS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/PLANS_HISTORY.md).
- The goal of this file is to stay short enough to be read at the start of a new session.

## Planning Anchors

- Active branch:
  - `main`
- Active algorithmic line:
  - `FMPC Stage 04 Incremental Bridge`
- Current defaults, current stage map, and frozen/open-line status live in:
  - [CURRENT_STATE.md](/e:/CodeSpace/PredictiveCoding/CURRENT_STATE.md)
- Planning should assume:
  - Stage 04 remains frozen as the bridge result
  - Stage 05 remains the current exploratory line

## Planning Rule

Current planning is split into two layers:

- short active plan:
  - this file
- historical plan log:
  - [archive/PLANS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/PLANS_HISTORY.md)

Do not add completed historical execution chains back into this file unless they change the active next move.

## Stage 04 Policy

Stage 04 is frozen as the current bridge result on `main`.

That means:

- keep the adopted corrective bridge package unchanged unless a genuinely different issue family appears
- do not open another package-internal Stage 04 suite from the current state
- do not reopen:
  - cone-family follow-ups
  - successor-value or successor-increment internal sweeps
  - readout-alignment follow-ups
  - bootstrap-source, target-lag, or curriculum follow-ups

## Stage 05 Exploratory Charter

### Objective

Evaluate whether the post-bridge corrected residual MeanFlow core can be strengthened in a narrow, mechanism-first way outside the saturated Stage 04 package.

### Why This Line Leaves Stage 04

This exploratory line is allowed only because:

- Stage 04 package-internal digging is locally saturated under the current selector-gate contract
- the next credible question is no longer a local Stage 04 repair
- the project now needs evidence about whether a different transport framing is worth pursuing

### Non-Goals

- replacing the active Stage 04 line on `main`
- claiming the exploratory probe already beats the frozen bridge result
- reopening the Stage 04 corrective package as if it were still the active search space
- introducing AlphaFlow, `muPC`-style scaling, or TF3 by default

### Current Known State

- the first Stage 05 exploratory probe already exists
- the Stage 05 addendum now exists for the corrected residual MeanFlow core v1 contract
- it is artifact-independent in target construction
- it shows positive mechanism signal on energy and fixed-point residual
- its task accuracy is still report-only and well below the frozen Stage 04 bridge result
- the formal frozen-bridge vs corrected residual core comparison now exists
- that comparison says the Stage 05 core has enough mechanism-first evidence to justify a narrow v2 charter
- that same comparison does not support replacing the frozen Stage 04 bridge result on `main`
- the Stage 05 v2 two-branch residual core now exists in the main Stage 05 probe implementation
- the formal Stage 05 v1 vs v2 comparison now exists
- that comparison says the Stage 05 v2 two-branch residual core improves mechanism magnitude over v1 under the current multiseed rule
- that same comparison still does not support any replacement claim against the frozen Stage 04 bridge result on `main`

## Immediate Execution Queue

### 1. Keep The Frozen Bridge Result Stable

Files to preserve as current Stage 04 control:

- `src/pc/stage_04_incremental_bridge/fmpc_tf2.py`
- `experiments/stage_04_incremental_bridge/fmpc_tf2.py`
- `tests/stage_04_incremental_bridge/`

Success condition:

- no semantic changes to the adopted Stage 04 bridge package while the current Stage 05 exploratory follow-up is running

### 2. Comparison Is Now Complete

Completed outcome:

- the formal comparison entry now exists under `src/pc/stage_05_ef_core_probe/`
- the comparison ran on shared `digits` data splits, shared seeds, and shared batch protocol
- the resulting evidence says Stage 05 clears the multiseed mechanism-first rule for a v2 charter
- the same evidence does not support replacing the frozen bridge result on `main`

Key artifact:

- `outputs/stage_05_ef_core_probe/frozen_bridge_vs_corrected_core_comparison/`

### 3. Stage 05 V2 Implementation Is Complete

Completed outcome:

- the Stage 05 addendum now documents the two-branch residual decomposition
- the Stage 05 probe now supports both:
  - v1 corrected residual MeanFlow baseline
  - v2 two-branch corrected residual MeanFlow candidate
- the Stage 05 smoke/unit test subset passes
- the formal Stage 05 v1 vs v2 comparison now exists and favors v2 on mechanism magnitude

Key artifacts:

- `outputs/stage_05_ef_core_probe/corrected_residual_core_v1_vs_v2_comparison/`

### 4. Refresh Frozen-Bridge vs Stage 05 V2 Comparison

Completed outcome:

- the refreshed frozen-bridge comparison now exists for the Stage 05 v2 candidate
- the refreshed comparison says:
  - Stage 05 v2 justifies continued exploration
  - Stage 05 v2 should be treated as the new exploratory reference
  - Stage 05 v2 is stronger than the frozen bridge on one-step mechanism
  - Stage 05 v2 is weaker than the frozen bridge on configured-step mechanism
  - Stage 05 v2 is weaker than the frozen bridge on report-only accuracy
  - Stage 05 v2 does not replace the frozen bridge result on `main`

Key artifact:

- `outputs/stage_05_ef_core_probe/frozen_bridge_vs_two_branch_corrected_core_comparison/`

### 5. Next Narrow Stage 05 Step Should Start From V2

Objective:

- define the next narrow Stage 05 mechanism-first step using the Stage 05 v2 two-branch core as the exploratory reference

Required framing:

- keep Stage 04 frozen on `main`
- do not reopen Stage 04 package-internal work
- do not treat the refreshed bridge comparison as a replacement claim
- keep Stage 05 mechanism-first and keep task accuracy report-only
- use current diagnostics to decide whether the next issue is:
  - training / budget
  - branch utilization
  - rollout accumulation
  - or selection pressure

### 6. Diagnose The Current Stage 05 V2 Reference Before Any V3 Charter

Completed outcome:

- the dedicated diagnostics now exist under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_diagnostics/`
- all Stage 05 v2 comparison seeds select the final training epoch
- configured-step mechanism metrics are still improving at the training boundary
- validation accuracy is also still improving at the training boundary
- the selected epoch is already the highest validation-accuracy epoch on every seed
- the state branch is materially active in forward residual magnitude rather than negligible
- the current narrow diagnosis is:
  - `likely_undertrained`

### 7. Implement The First Stage 05 V3-A Candidate

Objective:

- implement the first minimal Stage 05 v3-A candidate on top of the current Stage 05 v2 exploratory reference

Required framing:

- keep Stage 04 frozen on `main`
- do not reopen Stage 04 package-internal work
- do not continue pure same-family budget escalation from this state
- do not continue pure same-family efficiency tweaking from this state
- do not treat any Stage 05 result as a default-replacement claim against the frozen bridge result

Current evidence supporting this move:

- the stronger same-family `3072`-epoch budget materially improves configured-step mechanism and report-only accuracy
- the fixed-budget same-family efficiency diagnostic still fails to recover a material fraction of that same-family budget upside
- Stage 05 v2 therefore still looks useful as an exploratory reference, but not as a family that should keep being pushed only by epochs or schedule tweaks
- the next credible move is now to change the higher-level mechanism contract rather than another same-family local knob

Completed outcome:

- keep the current Stage 05 v2 family fixed as the exploratory reference:
  - `transport_family = two_branch_residual_meanflow_core`
  - `feature_aware_state_branch_tangents = true`
  - residual branch structure
  - corrected residual identity contract
  - selection rule
- define Stage 05 v3-A as a working-hypothesis-driven charter:
  - the current residual target may entangle transport residual and anchor-drift residual too tightly
  - that entanglement may be limiting configured-step efficiency more than one-step mechanism quality
- formalize the v3-A contract as:
  - `explicit transport-drift contract`
  - `u_psi = g_t + q_psi + d_psi`
- keep the shared protocol fixed for the next implementation pass:
  - `dataset = digits`
  - `seeds = (0, 1, 2)`
  - `batch_size = 128`
  - `layer_dims = (64, 16, 10)`
  - `transport_steps = 2`
  - `eval_steps = 15`
- require the next implementation pass to add:
  - a new Stage 05 v3-A candidate codepath
  - a new comparison entry or suite versus the current v2 reference
  - a matching smoke test
  - a dedicated artifact directory
  - aggregate summary fields for:
    - whether explicit transport-drift decomposition is enabled
    - pairwise deltas versus the current v2 reference
    - a gap-closure style decision field
    - `recommended_next_move`

- the repository now contains:
  - the v3-A candidate codepath `stage05_v3a_explicit_transport_drift_contract`
  - the explicit `gbar_boot / q_boot / d_boot` supervision split
  - a smoke-ready `v2 vs v3-A` comparison entry
  - matching Stage 05 smoke coverage

### 8. Run A Fixed-Budget V2 vs V3-A Comparison

Completed outcome:

- the fixed-budget comparison now exists under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_vs_v3a_explicit_transport_drift_fixed_budget_comparison/`
- the fixed-budget v3-A candidate materially improves configured-step mechanism over the current `1536`-epoch v2 reference
- the fixed-budget v3-A candidate keeps one-step mechanism positive on every comparison seed
- the fixed-budget v3-A candidate shows a positive gap-closure signal relative to the contextual `3072`-epoch v2 reference
- report-only validation and test accuracy remain slightly below the fixed-budget v2 reference, but not enough to count as an obvious regression under the current Stage 05 rule
- the current decision is therefore:
  - `recommended_next_move = proceed_to_stage05_v3b_curriculum_charter`

### 9. Implement The First Stage 05 V3-B Candidate

Completed outcome:

- the first minimal Stage 05 v3-B candidate now exists:
  - `stage05_v3b_trajectory_curriculum_contract`
- it keeps the v3-A explicit transport-drift decomposition intact
- it adds one aggregate trajectory curriculum loss on top of the v3-A scaffold
- a smoke-ready `v2 vs v3-A vs v3-B` comparison entry now also exists

Working hypothesis:

- after explicit transport-drift decomposition, the main remaining fixed-budget inefficiency is now more likely trajectory-level than target-entanglement-level
- the next mechanism change should therefore target the trajectory contract rather than another branch-local or schedule-local tweak

Normative v3-B charter direction:

- `Stage 05 v3-B = trajectory curriculum contract`
- keep the Stage 05 `t, r` notation unchanged
- introduce a curriculum split:
  - `alpha in (0, 1]`
  - `s = t + alpha * r`
  - `r_s = (1 - alpha) * r`
- use the exact trajectory decomposition:
  - `u^*_{t,r} = alpha * u^*_{t, alpha r} + (1 - alpha) * u^*_{s, r_s}`

Required framing:

- do not reopen Stage 04 package-internal work
- do not reopen another pure same-family budget escalation
- do not reopen another pure same-family efficiency tweak
- do not frame v3-B as replacing the frozen Stage 04 bridge result on `main`
- do not treat task accuracy as the primary gate

Immediate next move:

Completed outcome:

- the first real fixed-budget `v2 vs v3-A vs v3-B` comparison now exists under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_v3a_v3b_fixed_budget_comparison/`
- the fixed-budget v3-B candidate keeps one-step mechanism positive on every comparison seed
- it materially improves configured-step mechanism over the fixed-budget v2 control
- it directionally improves configured-step mechanism over the fixed-budget v3-A reference on every comparison seed
- its configured-step gain over v3-A remains below the current materiality threshold
- it slightly lowers report-only validation and test accuracy relative to v3-A, but not enough to count as an obvious regression under the current Stage 05 rule
- the current comparison decision is therefore:
  - `recommended_next_move = retain_v3a_as_active_reference`
- the narrow fixed-budget v3-B refinement diagnostic now also exists under:
  - `outputs/stage_05_ef_core_probe/stage05_v3b_refinement_diagnostic/`
- the strongest tested refinement is:
  - `stage05_v3b_stronger_traj_curr_weight`
- that refinement keeps one-step mechanism positive on every comparison seed
- it materially improves configured-step mechanism over the fixed-budget v3-A reference under the current threshold
- it does not materially improve configured-step mechanism over the original v3-B control under the same threshold
- it avoids an obvious report-only accuracy regression relative to v3-A
- the current refinement decision is therefore:
  - `recommended_next_move = promote_refined_v3b_and_recompare`
- the fresh fixed-budget refined v3-B recompare now also exists under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_v3a_refined_v3b_fixed_budget_recompare/`
- the promoted refined candidate is:
  - `stage05_v3b_stronger_traj_curr_weight`
- that refined recompare materially improves configured-step mechanism over the fixed-budget v3-A reference under the current threshold
- it also materially improves configured-step mechanism over the fixed-budget v2 control
- it keeps one-step mechanism positive and avoids an obvious report-only accuracy regression relative to v3-A
- the current recompare decision is therefore:
  - `recommended_next_move = promote_refined_v3b_as_active_reference`

Current planning implication:

- keep Stage 04 frozen on `main`
- keep Stage 05 mechanism-first and keep task accuracy report-only
- keep the fixed-budget v2 result as the immediate control
- promote `stage05_v3b_stronger_traj_curr_weight` as the current fixed-budget Stage 05 improvement reference
- treat the fixed-budget v3-A result as the previous fixed-budget comparison reference
- do not reopen the same fixed-budget recompare question as the immediate next move
- the post-promotion planning conclusion is now:
  - `v3-A` already addressed the target-entanglement question strongly enough that it is no longer the lead unresolved mechanism issue
  - refined `v3-B` then improved configured-step mechanism again, so trajectory-level supervision matters, but the remaining gap is not best described as another immediate v3-B promotion question
  - the current v3-B line still lacks an explicit endpoint / semigroup consistency contract across split horizons
- the planning-only `Stage 05 v3-C` charter is now open and has a first minimal diagnostic-only probe implementation
- any next Stage 05 planning or implementation pass should start from the promoted refined v3-B reference and compare against:
  - the fixed-budget v2 control
  - the promoted refined v3-B reference

### 10. Open The Stage 05 V3-C Charter

Objective:

- define the next Stage 05 mechanism question above the current refined v3-B trajectory-curriculum scaffold

Working ranking of remaining mechanism suspects:

- most credible:
  - endpoint / semigroup consistency contract
- second:
  - still-insufficient trajectory curriculum contract
- third:
  - corrected residual identity contract

Current evidence supporting this ranking:

- the fixed-budget `v2 vs v3-A` result already shows that explicit transport-drift decomposition materially improves configured-step mechanism
- the refined fixed-budget `v3-B` recompare then shows that stronger trajectory curriculum weighting materially improves configured-step mechanism again over v3-A
- those two results together mean target entanglement and basic trajectory weighting are both partly addressed
- the remaining fixed-budget gap is therefore more plausibly tied to a higher-level consistency layer than to another immediate branch-local or weight-local tweak
- the current v3-B contract still uses an aggregate detached continuation target and does not impose an explicit endpoint / semigroup consistency constraint
- the corrected residual identity contract is already explicit and branchwise in the current scaffold, so current repo evidence does not rank it as the lead unresolved issue

Normative v3-C charter direction:

- `Stage 05 v3-C = endpoint / semigroup consistency contract`
- keep the Stage 05 `t, r` notation unchanged
- define:
  - `s = t + alpha * r`
  - `r_s = (1 - alpha) * r`
- define the direct full-horizon endpoint prediction:
  - `z_hat_direct = z_t + r * u_hat(z_t, r, t; c)`
- define the split-horizon endpoint prediction:
  - `z_hat_mid = z_t + alpha * r * u_hat(z_t, alpha * r, t; c)`
  - `z_hat_split = z_t + alpha * r * u_hat(z_t, alpha * r, t; c) + (1 - alpha) * r * u_hat(z_hat_mid, r_s, s; c)`
- define the semigroup consistency residual:
  - `Delta_sg = z_hat_direct - z_hat_split`

Required framing:

- keep Stage 04 frozen on `main`
- keep Stage 05 mechanism-first and task accuracy report-only
- do not treat v3-C as a replacement claim against the frozen Stage 04 bridge result
- do not reopen another pure same-family budget escalation
- do not reopen another pure same-family efficiency tweak
- do not broaden the current v3-B parameter search as the immediate next move
- treat v3-C as a working hypothesis, not a proved fix

Required next implementation deliverables:

- a new Stage 05 v3-C candidate codepath on top of `stage05_v3b_stronger_traj_curr_weight`
- a new comparison entry or suite against:
  - the fixed-budget Stage 05 v2 control
  - the promoted refined Stage 05 v3-B reference
- a matching smoke test
- a dedicated artifact directory
- aggregate summary fields that report:
  - whether endpoint / semigroup consistency is enabled
  - the split-horizon semigroup identity
  - pairwise deltas versus the promoted refined v3-B reference
  - pairwise deltas versus the fixed-budget v2 control
  - a gap-closure style decision field
  - `recommended_next_move`

Current status update:

- the first minimal diagnostic-only v3-C candidate now exists:
  - `stage05_v3c_endpoint_semigroup_consistency_contract`
- it is implemented on top of `stage05_v3b_stronger_traj_curr_weight`
- it keeps the semigroup target single-sided detached in the first pass
- a smoke-ready `v2 vs promoted-v3B vs v3-C` comparison entry now also exists
- the smoke artifact now exists under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_v3c_comparison/`
- that smoke artifact is diagnostic only:
  - it verifies wiring and deterministic comparison readiness
  - it does not yet justify promoting v3-C over the promoted refined v3-B reference
- the first real fixed-budget comparison now also exists under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_v3c_fixed_budget_comparison/`
- that fixed-budget comparison says:
  - v3-C directionally improves configured-step mechanism over the promoted refined v3-B reference
  - v3-C improves contextual gap closure relative to the `3072`-epoch same-family reference
  - v3-C keeps the mechanism-first gate positive and avoids an obvious report-only accuracy regression
  - v3-C still does not materially displace the promoted refined v3-B reference under the current threshold
- the narrow v3-C refinement diagnostic now also exists under:
  - `outputs/stage_05_ef_core_probe/stage05_v3c_refinement_diagnostic/`
- that refinement diagnostic says:
  - the best tested refinement variant is `stage05_v3c_stronger_semigroup_weight`
  - it materially improves configured-step mechanism over `stage05_v3b_stronger_traj_curr_weight` under the current threshold
  - it does not materially improve configured-step mechanism over the current v3-C control under the same threshold
  - it avoids an obvious report-only accuracy regression under the current Stage 05 rule
- the immediate next move is therefore:
  - keep `stage05_v3b_stronger_traj_curr_weight` as the active fixed-budget Stage 05 improvement reference until recompare completes
  - treat `stage05_v3c_stronger_semigroup_weight` as the promoted refined v3-C candidate
  - run a fresh fixed-budget recompare of `v2` vs promoted refined `v3-B` vs refined `v3-C`

## Exploratory Acceptance Criteria

For the current Stage 05 exploratory stage, acceptance is mechanism-first.

Primary acceptance signals:

- one-step energy decrease relative to identity
- few-step fixed-point residual decrease relative to identity
- deterministic artifact correctness
- artifact-independent target construction

Secondary report-only signals:

- validation accuracy
- test accuracy

Stage 05 should not be promoted merely because of a small task-metric fluctuation without a clear mechanism advantage.

## Document Shortcuts

- prompt-drafting context:
  - [PROMPT_CONTEXT.md](/e:/CodeSpace/PredictiveCoding/PROMPT_CONTEXT.md)
- low-context repository entry:
  - [LLM_BRIEF.md](/e:/CodeSpace/PredictiveCoding/LLM_BRIEF.md)
- math layer:
  - [spec_math.md](/e:/CodeSpace/PredictiveCoding/spec_math.md)
  - [specs/stage_03_transport_core_v1.md](/e:/CodeSpace/PredictiveCoding/specs/stage_03_transport_core_v1.md)
  - [specs/stage_04_incremental_bridge.md](/e:/CodeSpace/PredictiveCoding/specs/stage_04_incremental_bridge.md)
  - [specs/stage_05_ef_core_probe.md](/e:/CodeSpace/PredictiveCoding/specs/stage_05_ef_core_probe.md)
- current frozen result summary:
  - [RESULTS.md](/e:/CodeSpace/PredictiveCoding/RESULTS.md)
- current validation contract:
  - [validation.md](/e:/CodeSpace/PredictiveCoding/validation.md)
- historical long-form detail:
  - [archive/PLANS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/PLANS_HISTORY.md)
  - [archive/validation_history.md](/e:/CodeSpace/PredictiveCoding/archive/validation_history.md)
