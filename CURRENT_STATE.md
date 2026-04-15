# CURRENT_STATE.md

This file is the short operational summary for the repository.

- Use it for current state only.
- Use [PLANS.md](/e:/CodeSpace/PredictiveCoding/PLANS.md) for the active forward plan.
- Use [archive/PLANS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/PLANS_HISTORY.md) and [archive/validation_history.md](/e:/CodeSpace/PredictiveCoding/archive/validation_history.md) for long historical detail.

## Active Branch, Implemented Line, And Forward Charter

- Active branch:
  - `main`
- Current adopted implemented line on `main`:
  - `FMPC Stage 04 Incremental Bridge`
- Current frozen high-budget mechanism reference stage:
  - `FMPC Stage 05 EF Core Probe`
- Current active forward charter:
  - `FMPC Stage 06 Low-Budget Efficiency`

For the full numbered stage map and directory layout, use [README.md](/e:/CodeSpace/PredictiveCoding/README.md).

## Current Defaults

- Current adopted Stage 04 bridge default on `main`:
  - `tf2_corrective_transport_terminal_angleclip_default`
- Current canonical Stage 04 identity default:
  - `feature_aware_tangents = false`
- Historical corrective working reference:
  - `tf2_corrective_transport_default`
- Default repository sync rule:
  - after any completed code or documentation change, sync to GitHub and Google Drive by default unless the user explicitly disables one side or a concrete blocker prevents it

## Relevant Math Layer

- Baseline root spec:
  - [spec_math.md](/e:/CodeSpace/PredictiveCoding/spec_math.md)
- Stage 04 bridge math:
  - [specs/stage_03_transport_core_v1.md](/e:/CodeSpace/PredictiveCoding/specs/stage_03_transport_core_v1.md)
  - [specs/stage_04_incremental_bridge.md](/e:/CodeSpace/PredictiveCoding/specs/stage_04_incremental_bridge.md)
- Stage 05 mechanism-reference math:
  - [specs/stage_05_ef_core_probe.md](/e:/CodeSpace/PredictiveCoding/specs/stage_05_ef_core_probe.md)
- Stage 06 efficiency-first charter:
  - [specs/stage_06_low_budget_efficiency.md](/e:/CodeSpace/PredictiveCoding/specs/stage_06_low_budget_efficiency.md)

## Stage 04 Status

The corrective Incremental Bridge package remains closed from the current state.

Operational conclusion:

- the adopted corrective Stage 04 bridge package is locally saturated under the current selector/gate contract
- do not reopen package-internal Stage 04 digging unless:
  - a genuinely different issue family appears from new evidence, or
  - the project explicitly decides to leave the current package or selector-gate contract

## Stage 05 Status

Stage 05 is no longer the default forward efficiency line.

Its role is now:

- frozen mechanism-first exploration stage
- reusable scaffold and evidence base
- high-budget mechanism reference, not default efficiency reference

The key repository facts are now:

- the Stage 05 v2 budget push from `1536` to `3072` epochs was real and boundary-limited:
  - configured-step energy moved from `-0.004980554368336933` to `-0.006199075439848138`
  - configured-step residual moved from `-1.84517884128533e-05` to `-2.89427156054073e-05`
  - selected epoch stayed at the final epoch on every seed for both budgets
  - the artifact still reports:
    - `budget_line_still_looks_boundary_limited = true`
    - `budget_line_should_continue = true`
- the Stage 05 v2 same-family efficiency tweak at the fixed `1536`-epoch ceiling was nearly a no-op:
  - configured-step energy only moved from `-0.004980554368336933` to `-0.004984766983336293`
  - configured-step residual only moved from `-1.84517884128533e-05` to `-1.84607254479558e-05`
  - report-only validation and test accuracy stayed unchanged
  - the artifact reports:
    - `same_family_efficiency_change_materially_improves_configured_step_mechanism = false`
    - `same_family_efficiency_change_materially_improves_report_only_accuracy = false`
    - `same_family_efficiency_change_materially_narrows_gap_to_3072_reference = false`
- the active refined v3-C reference is justified as the strongest high-budget Stage 05 mechanism result:
  - `stage05_v3c_stronger_semigroup_weight` materially beat promoted refined v3-B in the fixed-budget recompare
  - configured-step energy moved from `-0.005720360383603999` to `-0.0059706216916698045`
  - configured-step residual moved from `-2.16088724262894e-05` to `-2.27017773664957e-05`
- the active refined v3-C reference is also expensive:
  - Stage 05 v2 `1536` runtime proxy mean is about `240.9623` seconds
  - promoted refined v3-B runtime proxy mean is about `312.6482` seconds
  - active refined v3-C runtime proxy mean is about `776.1473` seconds
  - it should therefore remain a high-budget mechanism reference, not be treated by default as a more efficient route
- the narrow v3-C contract-consolidation micro-family is now locally saturated:
  - the final continuation-strength diagnostic identifies `stage05_v3c_endpoint_line_continuation_blend_trajectory_contract` as the strongest predecessor inside the micro-family
  - `stage05_v3c_scaled_continuation_blend_trajectory_contract` becomes the strongest tested narrow same-family candidate by configured-step ranking
  - but it still does not materially beat `stage05_v3c_stronger_semigroup_weight`
  - the final decision is:
    - `freeze_narrow_v3c_contract_consolidation_line_as_locally_saturated`

Operational interpretation:

- Stage 05 established that the mechanism signal exists
- Stage 05 did not establish that the strongest mechanism path is also an efficient path
- Stage 05 should now be treated as a frozen mechanism evidence base rather than the default next mainline

## Stage 06 Status

Stage 06 is now both formally open and code-backed.

Definition:

- Stage 06 asks whether the current scaffold can remain mechanism-positive under low-budget and low-compute matched-budget conditions

Current implemented Stage 06 line:

- `stage06_v1_objective_curriculum_energydrop_default`
- implementation path:
  - [src/pc/stage_06_low_budget_efficiency/fmpc_stage06_objective_curriculum.py](/e:/CodeSpace/PredictiveCoding/src/pc/stage_06_low_budget_efficiency/fmpc_stage06_objective_curriculum.py)
- current authoritative post-semantic-alignment baseline artifact:
  - [outputs/stage_06_low_budget_efficiency/stage06_v1_low_budget_comparison/stage06_v1_post_semantic_alignment_rebaseline](/e:/CodeSpace/PredictiveCoding/outputs/stage_06_low_budget_efficiency/stage06_v1_low_budget_comparison/stage06_v1_post_semantic_alignment_rebaseline)
- current authoritative v2 follow-up comparison artifact:
  - [outputs/stage_06_low_budget_efficiency/stage06_v2_low_budget_comparison/stage06_v2_initial_authoritative_comparison](/e:/CodeSpace/PredictiveCoding/outputs/stage_06_low_budget_efficiency/stage06_v2_low_budget_comparison/stage06_v2_initial_authoritative_comparison)
- historical pre-alignment artifact:
  - [outputs/stage_06_low_budget_efficiency/stage06_v1_low_budget_comparison/stage06_v1_initial_probe](/e:/CodeSpace/PredictiveCoding/outputs/stage_06_low_budget_efficiency/stage06_v1_low_budget_comparison/stage06_v1_initial_probe)

Current Stage 06 constraints:

- do not accept a candidate whose main proof appears only at `1536+` epochs
- do not treat long-budget runs as the first existence proof
- do not continue the narrow Stage 05 v3-C continuation / midpoint / coupled / precision / scaled micro-family by default
- start from the validated Stage 05 scaffold, but move the next search above that micro-family
- interpret that preserved scaffold as:
  - the Stage 05 two-branch `u_psi = g_t + q_psi + d_psi` parameterization
  - plus Stage 05 trajectory / semigroup target-builder reuse
  - not the Stage 05 v3-A branchwise transport / drift supervision contract

Current Stage 06 default hypothesis space:

- objective curriculum
- optimization-contract refinement
- update scheduling
- shorter or more direct error / transport pathways
- matched-budget efficiency contracts

Current main question:

- low-budget / low-compute viability, not another narrow Stage 05 target-geometry variation

Current Stage 06 result:

- the post-semantic-alignment rebaseline passed Tier 1 viability at `128` epochs:
  - one-step and configured-step mechanism-positive rates were both `1.0`
- but it failed the Tier 2 main gate at `256` epochs against the matched-budget Stage 05 control `stage05_v3c_stronger_semigroup_weight`:
  - configured-step energy delta mean was `-0.001306768276636246` for Stage 06 v1 vs `-0.0013464605038717343` for `stage05_v3c_stronger_semigroup_weight`
  - configured-step residual delta mean was `-3.975569074520645e-06` for Stage 06 v1 vs `-4.0279469750049745e-06` for `stage05_v3c_stronger_semigroup_weight`
  - the pairwise gain fractions stayed negative
- the result did not justify a `512` rescue:
  - `tier2_positive_trend_for_rescue = false`
  - `rescue_512_warranted = false`
- the old pre-alignment Stage 06 v1 artifact is now historical only and is superseded for current state judgment by the post-semantic-alignment rebaseline above

Current Stage 06 v1 baseline positioning:

- directly confirmed from the current code and the authoritative post-semantic-alignment artifact:
  - Stage 06 v1 keeps the A2 scaffold boundary:
    - Stage 05 two-branch parameterization
    - Stage 05 target-builder reuse
    - no Stage 05 v3-A branchwise supervision restoration
  - the implemented objective is a hard handoff contract:
    - `L_6A(k) = (1 - beta_obj(k)) * L_traj + beta_obj(k) * L_semi + lambda_energy_drop * L_drop + lambda_fixed_point * L_fp`
    - `beta_obj(k)` reaches `1` and stays there in the final quarter of training
  - the baseline therefore proves:
    - the A2 aggregate-objective line can stay mechanism-positive at Tier 1
    - the line can run at materially lower runtime than the matched-budget Stage 05 control
- it does not prove:
  - Tier 2 matched-budget superiority over `stage05_v3c_stronger_semigroup_weight`
  - clearly better cost-effectiveness under the current Stage 06 gate
  - that the current hard-handoff objective contract is promotable
- operational consequence:
  - `stage06_v1_objective_curriculum_energydrop_default` remains the authoritative Stage 06 baseline artifact
  - it is not a promoted forward direction

Current Stage 06 v2 follow-up result:

- directly confirmed from the dedicated v2 comparison artifact:
  - candidate:
    - `stage06_v2_persistent_overlap_objective_curriculum_energydrop_default`
  - authoritative comparison artifact:
    - [outputs/stage_06_low_budget_efficiency/stage06_v2_low_budget_comparison/stage06_v2_initial_authoritative_comparison](/e:/CodeSpace/PredictiveCoding/outputs/stage_06_low_budget_efficiency/stage06_v2_low_budget_comparison/stage06_v2_initial_authoritative_comparison)
  - single changed axis:
    - replace the v1 hard late `L_traj -> L_semi` handoff with a persistent-overlap late objective where late trajectory weight stays `0.25` and late semigroup weight stays `0.75`
- the authoritative v2 result says:
  - Tier 1 remains viable at `128` epochs
  - Tier 2 still fails against the matched-budget Stage 05 control `stage05_v3c_stronger_semigroup_weight`
  - `tier2_positive_trend_for_rescue = false`
  - `rescue_512_warranted = false`
  - `materially_beats_matched_budget_stage05_control = false`
  - `shows_better_cost_effectiveness_than_stage05_control = false`
- relative to the current v1 baseline, v2 shows:
  - slightly stronger raw configured-step mechanism at both `128` and `256` epochs
  - but materially worse runtime proxy and worse mechanism gain per runtime
  - no stronger rescue signal
- operational consequence:
  - v2 is now an implemented and artifact-tested Stage 06 follow-up
  - it does not replace v1 as the authoritative Stage 06 baseline
  - it does not justify auto-opening a `v2.1` / `v2.2` schedule-retuning line from the current state

## Current Recommendation

- Keep the Stage 04 bridge result frozen on `main`.
- Keep Stage 05 frozen as the high-budget mechanism reference stage.
- Keep `stage05_v3c_stronger_semigroup_weight` as the current high-budget Stage 05 mechanism reference.
- Do not treat `stage05_v3c_stronger_semigroup_weight` as an efficiency reference by default.
- Treat the narrow Stage 05 v3-C contract-consolidation micro-family as closed from the current state.
- Use Stage 06 as the active implementation and validation charter.
- Keep `stage06_v1_objective_curriculum_energydrop_default` as the current post-semantic-alignment Stage 06 baseline artifact, but do not promote it past the current matched-budget Stage 05 control `stage05_v3c_stronger_semigroup_weight`.
- Keep `stage06_v2_persistent_overlap_objective_curriculum_energydrop_default` as an artifact-tested A2-family follow-up, not as the authoritative Stage 06 baseline.
- Do not auto-promote the v2 persistent-overlap schedule tweak, because it still fails the Tier 2 main gate and does not improve cost-effectiveness enough to displace v1.
- Do not auto-open another immediate `v2.x` schedule-retuning pass from the current state.
- Require any next new forward probe to be:
  - low-budget-first
  - matched-budget
  - mechanism-positive and cost-aware
  - above the saturated Stage 05 geometry micro-family
  - materially different from the current Stage 06 v1 objective-contract line if another Stage 06 pass is opened

## Relevant Files

- Stage 04 implementation:
  - [src/pc/stage_04_incremental_bridge/fmpc_tf2.py](/e:/CodeSpace/PredictiveCoding/src/pc/stage_04_incremental_bridge/fmpc_tf2.py)
- Stage 05 mechanism reference:
  - [src/pc/stage_05_ef_core_probe/fmpc_ef_exploratory_probe.py](/e:/CodeSpace/PredictiveCoding/src/pc/stage_05_ef_core_probe/fmpc_ef_exploratory_probe.py)
- Stage 06 implementation:
  - [src/pc/stage_06_low_budget_efficiency/fmpc_stage06_objective_curriculum.py](/e:/CodeSpace/PredictiveCoding/src/pc/stage_06_low_budget_efficiency/fmpc_stage06_objective_curriculum.py)
- Stage 06 charter:
  - [specs/stage_06_low_budget_efficiency.md](/e:/CodeSpace/PredictiveCoding/specs/stage_06_low_budget_efficiency.md)
- current active plan:
  - [PLANS.md](/e:/CodeSpace/PredictiveCoding/PLANS.md)
- current active validation contract:
  - [validation.md](/e:/CodeSpace/PredictiveCoding/validation.md)

## Relevant Artifacts

- Stage 04 authority artifact tree:
  - [outputs/stage_04_incremental_bridge](/e:/CodeSpace/PredictiveCoding/outputs/stage_04_incremental_bridge)
- Stage 05 budget-push validation artifact:
  - [outputs/stage_05_ef_core_probe/stage05_v2_budget_push_validation_1536_to_3072](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_budget_push_validation_1536_to_3072)
- Stage 05 efficiency diagnostic artifact:
  - [outputs/stage_05_ef_core_probe/stage05_v2_efficiency_diagnostic_at_1536](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_efficiency_diagnostic_at_1536)
- Stage 05 refined v3-C recompare artifact:
  - [outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_refined_v3c_fixed_budget_recompare](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_refined_v3c_fixed_budget_recompare)
- Stage 05 continuation-strength diagnostic artifact:
  - [outputs/stage_05_ef_core_probe/stage05_v3c_continuation_strength_diagnostic](/e:/CodeSpace/PredictiveCoding/outputs/stage_05_ef_core_probe/stage05_v3c_continuation_strength_diagnostic)
- Stage 06 authoritative post-semantic-alignment baseline artifact:
  - [outputs/stage_06_low_budget_efficiency/stage06_v1_low_budget_comparison/stage06_v1_post_semantic_alignment_rebaseline](/e:/CodeSpace/PredictiveCoding/outputs/stage_06_low_budget_efficiency/stage06_v1_low_budget_comparison/stage06_v1_post_semantic_alignment_rebaseline)
- Stage 06 authoritative v2 follow-up comparison artifact:
  - [outputs/stage_06_low_budget_efficiency/stage06_v2_low_budget_comparison/stage06_v2_initial_authoritative_comparison](/e:/CodeSpace/PredictiveCoding/outputs/stage_06_low_budget_efficiency/stage06_v2_low_budget_comparison/stage06_v2_initial_authoritative_comparison)
