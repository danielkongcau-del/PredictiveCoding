# PLANS.md

This file tracks the current forward plan only.

- Historical plan detail has moved to [archive/PLANS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/PLANS_HISTORY.md).
- The goal of this file is to stay short enough to be read at the start of a new session.

## Active-State Sync

- Active branch:
  - `main`
- Active algorithmic line:
  - `FMPC Stage 04 Incremental Bridge`
- Current adopted Stage 04 default:
  - `tf2_corrective_transport_terminal_angleclip_default`
- Current exploratory line:
  - `FMPC Stage 05 EF Core Probe`

## FMPC Stage Naming

- `stage_01_reference_prep/` -> `FMPC Stage 01 Reference Prep`
- `stage_02_interval_velocity/` -> `FMPC Stage 02 Interval Velocity Exploration`
- `stage_03_transport_core_v1/` -> `FMPC Stage 03 Transport Core v1`
- `stage_04_incremental_bridge/` -> `FMPC Stage 04 Incremental Bridge`
- `stage_05_ef_core_probe/` -> `FMPC Stage 05 EF Core Probe`

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

Evaluate whether the post-bridge exploratory core has enough mechanism signal to justify a v2 charter outside the saturated Stage 04 package.

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
- it is artifact-independent in target construction
- it shows positive mechanism signal on energy and fixed-point residual
- its task accuracy is still report-only and well below the frozen Stage 04 bridge result

## Immediate Execution Queue

### 1. Keep The Frozen Bridge Result Stable

Files to preserve as current Stage 04 control:

- `src/pc/stage_04_incremental_bridge/fmpc_tf2.py`
- `experiments/stage_04_incremental_bridge/fmpc_tf2.py`
- `tests/stage_04_incremental_bridge/`

Success condition:

- no semantic changes to the adopted Stage 04 bridge package while the Stage 05 comparison decision is pending

### 2. Run Frozen-Bridge Vs Exploratory-Core Comparison

Objective:

- compare the frozen Stage 04 bridge result against the Stage 05 exploratory core on a mechanism-first basis

Required comparison outputs:

- energy-side comparison
- fixed-point residual comparison
- deterministic artifact checks
- report-only task metrics

Primary decision question:

- does the exploratory line have enough mechanism signal to justify a v2 charter?

### 3. Decide Stage 05 V2 Charter

Possible outcomes:

- if comparison is favorable:
  - draft a narrow Stage 05 v2 charter
- if comparison is weak:
  - keep Stage 04 frozen and stop new code until a different strategic path is chosen

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

## Documentation Architecture

Current repository document layers are:

- low-context prompt entry:
  - [LLM_BRIEF.md](/e:/CodeSpace/PredictiveCoding/LLM_BRIEF.md)
- current frozen result summary:
  - [RESULTS.md](/e:/CodeSpace/PredictiveCoding/RESULTS.md)
- current operational state:
  - [CURRENT_STATE.md](/e:/CodeSpace/PredictiveCoding/CURRENT_STATE.md)
- current forward plan:
  - [PLANS.md](/e:/CodeSpace/PredictiveCoding/PLANS.md)
- current validation contract:
  - [validation.md](/e:/CodeSpace/PredictiveCoding/validation.md)
- historical long-form results, plan, and validation detail:
  - [archive/AGENTS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/AGENTS_HISTORY.md)
  - [archive/README_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/README_HISTORY.md)
  - [archive/RESULTS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/RESULTS_HISTORY.md)
  - [archive/PLANS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/PLANS_HISTORY.md)
  - [archive/validation_history.md](/e:/CodeSpace/PredictiveCoding/archive/validation_history.md)

## This Turn: Documentation Context-Thinning Refactor

Objective:

- make the repository more usable for low-context web GPT prompt drafting

Changes:

- add `LLM_BRIEF.md`
- move historical AGENTS detail to `archive/AGENTS_HISTORY.md`
- move historical README detail to `archive/README_HISTORY.md`
- keep `RESULTS.md` as a short frozen-result summary
- move historical results detail to `archive/RESULTS_HISTORY.md`
- move historical plan detail to `archive/PLANS_HISTORY.md`
- move historical validation detail to `archive/validation_history.md`
- keep `CURRENT_STATE.md`, `PLANS.md`, `RESULTS.md`, and `validation.md` short and current

Validation:

- markdown consistency pass across:
  - `LLM_BRIEF.md`
  - `CURRENT_STATE.md`
  - `PLANS.md`
  - `RESULTS.md`
  - `validation.md`
  - `README.md`
  - `AGENTS.md`
- ensure the Google Drive sync script includes the new briefing and archive docs
