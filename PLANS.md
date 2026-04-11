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

Evaluate whether the post-bridge corrected residual MeanFlow core has enough mechanism signal to justify a v2 charter outside the saturated Stage 04 package.

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

## Immediate Execution Queue

### 1. Keep The Frozen Bridge Result Stable

Files to preserve as current Stage 04 control:

- `src/pc/stage_04_incremental_bridge/fmpc_tf2.py`
- `experiments/stage_04_incremental_bridge/fmpc_tf2.py`
- `tests/stage_04_incremental_bridge/`

Success condition:

- no semantic changes to the adopted Stage 04 bridge package while the Stage 05 v2 charter is being defined

### 2. Comparison Is Now Complete

Completed outcome:

- the formal comparison entry now exists under `src/pc/stage_05_ef_core_probe/`
- the comparison ran on shared `digits` data splits, shared seeds, and shared batch protocol
- the resulting evidence says Stage 05 clears the multiseed mechanism-first rule for a v2 charter
- the same evidence does not support replacing the frozen bridge result on `main`

Key artifact:

- `outputs/stage_05_ef_core_probe/frozen_bridge_vs_corrected_core_comparison/`

### 3. Draft Stage 05 V2 Charter

Objective:

- define the next narrow Stage 05 step now that the corrected residual core has cleared the mechanism-first comparison gate

Required framing:

- keep Stage 04 frozen on `main`
- do not reopen Stage 04 package-internal work
- do not treat the comparison result as a default-replacement claim
- keep Stage 05 mechanism-first

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
