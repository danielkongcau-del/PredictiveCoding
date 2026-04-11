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

### 7. If Stage 05 Continues, Target Longer Training / Budget On V2 First

Objective:

- keep the current Stage 05 v2 transport family fixed and test whether a longer training / budget pass on the same reference closes more of the configured-step and report-only gap before inventing a new family

Required framing:

- keep Stage 04 frozen on `main`
- do not reopen Stage 04 package-internal work
- do not change the Stage 05 v2 transport family before the budget question is answered
- do not treat any Stage 05 result as a default-replacement claim against the frozen bridge result

Completed outcome:

- the dedicated budget-push validation now exists under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_budget_push_validation_1536_to_3072/`
- the same Stage 05 v2 family with a stronger `3072`-epoch budget materially improves:
  - configured-step validation energy delta vs identity
  - configured-step validation fixed-point residual delta vs identity
  - report-only validation accuracy
  - report-only test accuracy
- the stronger budget still selects the final training epoch on every seed
- the explicit stop-rule layer now also says:
  - `budget_line_still_looks_boundary_limited = true`
  - `budget_line_should_continue = true`
  - `budget_line_should_stop_and_open_v3 = false`
  - `budget_line_interpretation = boundary_limited_mechanism_prototype`
- the current answer is therefore:
  - keep pushing budget on the same v2 family
  - do not open a true Stage 05 v3 mechanism charter yet

Current execution plan:

- keep the current Stage 05 v2 family fixed
- if Stage 05 continues, run the next narrow budget push beyond the current `3072`-epoch reference on the same:
  - `transport_family = two_branch_residual_meanflow_core`
  - `feature_aware_state_branch_tangents = true`
  - residual branch structure
  - corrected residual identity contract
  - selection rule
- keep the shared protocol fixed:
  - `dataset = digits`
  - `seeds = (0, 1, 2)`
  - `batch_size = 128`
  - `layer_dims = (64, 16, 10)`
  - `transport_steps = 2`
  - `eval_steps = 15`
- require the next budget pass to report again:
  - configured-step validation mechanism metrics
  - val/test accuracy
  - selected epoch
  - whether selection still hits the final training boundary
  - runtime
- require the next budget pass to add an explicit stop-rule layer that answers:
  - whether the same-family budget line still looks boundary-limited
  - whether the budget line should continue
  - or whether budget should stop and a true Stage 05 v3 charter should be opened
- use the next formal budget comparison:
  - current reference: `3072 epochs`
  - stronger candidate: still to be chosen by the same hard stop rule, only if this same-family line remains open
- enforce the hard stop rule:
  - continue budget only if the stronger budget still materially improves configured-step mechanism
  - and still materially improves report-only accuracy
  - and still selects the final training epoch on all seeds
- otherwise stop same-family budget escalation and open a true Stage 05 v3 charter

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
