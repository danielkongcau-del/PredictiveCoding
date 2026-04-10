# Predictive Coding from Scratch

A NumPy-first research codebase for building a reliable, extensible, near-paper-grade predictive coding implementation.

This repository is developed as an explicit research codebase rather than a loose prototype. The goals are:

- mathematical fidelity to [spec_math.md](/e:/CodeSpace/PredictiveCoding/spec_math.md)
- readable, inspectable NumPy implementations
- reproducible experiments and artifact traces
- staged extension from baseline predictive coding into FMPC research

## Current Repository State

- active branch:
  - `main`
- current active algorithmic line:
  - `FMPC Stage 04 Incremental Bridge`
- current adopted bridge default on `main`:
  - `tf2_corrective_transport_terminal_angleclip_default`
- current canonical bridge identity default:
  - `feature_aware_tangents = false`
- current exploratory line:
  - `FMPC Stage 05 EF Core Probe`

Current working interpretation:

- Stage 04 is frozen as the current bridge result on `main`
- Stage 04 package-internal digging is treated as closed from the current state
- the current open work is no longer another Stage 04 repair
- the next active question is whether the Stage 05 exploratory line earns a v2 charter

## Current Result Snapshot

- Phase 2 toy-benchmark methodology phase is frozen
- strongest current Phase 2 conclusion remains benchmark-dependent:
  - `toy_regression`: PC stays ahead
  - `toy_sine_regression`: MLP stays ahead
- Phase 3 currently provides standalone `digits_mlp` and `digits_pc` baselines on `sklearn.datasets.load_digits`
- the first Stage 05 exploratory probe has positive mechanism signal on energy and fixed-point residual, but its task accuracy remains report-only and well below the frozen Stage 04 bridge result

Use these short current documents for details:

- [CURRENT_STATE.md](/e:/CodeSpace/PredictiveCoding/CURRENT_STATE.md)
- [PLANS.md](/e:/CodeSpace/PredictiveCoding/PLANS.md)
- [RESULTS.md](/e:/CodeSpace/PredictiveCoding/RESULTS.md)
- [validation.md](/e:/CodeSpace/PredictiveCoding/validation.md)

## Read Order And Precedence

Before making code changes, read these files in order:

1. `README.md`
2. `AGENTS.md`
3. `CURRENT_STATE.md`
4. `PLANS.md`
5. `spec_math.md`
6. the applicable stage addendum under `specs/` if the task touches FMPC stage-specific logic
7. `validation.md`

If documents conflict, repository precedence is:

- `spec_math.md > validation.md > AGENTS.md > CURRENT_STATE.md > PLANS.md > README.md`

For low-context web GPT prompt drafting, start with:

1. `LLM_BRIEF.md`
2. `AGENTS.md`
3. `CURRENT_STATE.md`

Then open `RESULTS.md`, `spec_math.md`, `PLANS.md`, `validation.md`, or `archive/` only as needed.

## Document Guide

The repository now uses a layered documentation structure:

- low-context entry:
  - [LLM_BRIEF.md](/e:/CodeSpace/PredictiveCoding/LLM_BRIEF.md)
- current operational docs:
  - [CURRENT_STATE.md](/e:/CodeSpace/PredictiveCoding/CURRENT_STATE.md)
  - [PLANS.md](/e:/CodeSpace/PredictiveCoding/PLANS.md)
  - [RESULTS.md](/e:/CodeSpace/PredictiveCoding/RESULTS.md)
  - [validation.md](/e:/CodeSpace/PredictiveCoding/validation.md)
- long historical detail:
  - [archive/AGENTS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/AGENTS_HISTORY.md)
  - [archive/README_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/README_HISTORY.md)
  - [archive/RESULTS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/RESULTS_HISTORY.md)
  - [archive/PLANS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/PLANS_HISTORY.md)
  - [archive/validation_history.md](/e:/CodeSpace/PredictiveCoding/archive/validation_history.md)

If you only need the current project picture, do not open the archive first.

## Which Document Answers What

Use this lookup rule when deciding where to read next:

- if you want current active line, defaults, frozen/open status, or the immediate recommended move:
  - [CURRENT_STATE.md](/e:/CodeSpace/PredictiveCoding/CURRENT_STATE.md)
- if you want the current execution queue or current exploratory charter:
  - [PLANS.md](/e:/CodeSpace/PredictiveCoding/PLANS.md)
- if you want the current frozen result snapshot:
  - [RESULTS.md](/e:/CodeSpace/PredictiveCoding/RESULTS.md)
- if you want current acceptance rules or validation semantics:
  - [validation.md](/e:/CodeSpace/PredictiveCoding/validation.md)
- if you want the authoritative math:
  - [spec_math.md](/e:/CodeSpace/PredictiveCoding/spec_math.md)
  - plus the applicable stage addendum under [specs/](/e:/CodeSpace/PredictiveCoding/specs)
- if you want standing coding constraints:
  - [AGENTS.md](/e:/CodeSpace/PredictiveCoding/AGENTS.md)
- if you want older narratives and historical evidence chains:
  - `archive/`

For web GPT, [LLM_BRIEF.md](/e:/CodeSpace/PredictiveCoding/LLM_BRIEF.md) is the cheapest starting point.

## FMPC Stage Map

- `stage_01_reference_prep/` -> `FMPC Stage 01 Reference Prep`
- `stage_02_interval_velocity/` -> `FMPC Stage 02 Interval Velocity Exploration`
- `stage_03_transport_core_v1/` -> `FMPC Stage 03 Transport Core v1`
- `stage_04_incremental_bridge/` -> `FMPC Stage 04 Incremental Bridge`
- `stage_05_ef_core_probe/` -> `FMPC Stage 05 EF Core Probe`

Interpretation:

- Stage 01 is a frozen reference-prep checkpoint
- Stage 02 is interval/velocity exploration
- Stage 03 is the first sealed artifact-independent transport stage
- Stage 04 is the current frozen bridge result on `main`
- Stage 05 is the current exploratory line

## Repository Layout

```text
predictive-coding/
  LLM_BRIEF.md
  README.md
  AGENTS.md
  CURRENT_STATE.md
  PLANS.md
  RESULTS.md
  spec_math.md
  specs/
  validation.md
  archive/
  src/pc/
  experiments/
  tests/
  outputs/
```

Important lookup rules:

- shared predictive-coding substrate stays under:
  - `src/pc/`
- stage-specific FMPC implementations live under:
  - `src/pc/stage_01_reference_prep/`
  - `src/pc/stage_02_interval_velocity/`
  - `src/pc/stage_03_transport_core_v1/`
  - `src/pc/stage_04_incremental_bridge/`
  - `src/pc/stage_05_ef_core_probe/`
- stage-specific mathematical addenda live under:
  - `specs/stage_03_transport_core_v1.md`
  - `specs/stage_04_incremental_bridge.md`
- matching stage experiment entry points live under:
  - `experiments/stage_04_incremental_bridge/`
  - `experiments/stage_05_ef_core_probe/`
- matching tests live under:
  - `tests/stage_04_incremental_bridge/`
  - `tests/stage_05_ef_core_probe/`
- newer FMPC artifact trees live under:
  - `outputs/stage_04_incremental_bridge/`
  - `outputs/stage_05_ef_core_probe/`

Historical baseline outputs may still keep stable legacy names such as:

- `outputs/digits_mlp/`
- `outputs/digits_pc/`
- `outputs/digits_baselines/`

## Where To Start In Code

If you are working on the frozen bridge result:

- implementation:
  - `src/pc/stage_04_incremental_bridge/fmpc_tf2.py`
- experiment entry:
  - `experiments/stage_04_incremental_bridge/fmpc_tf2.py`
- tests:
  - `tests/stage_04_incremental_bridge/`

If you are working on the current exploratory line:

- implementation:
  - `src/pc/stage_05_ef_core_probe/fmpc_ef_exploratory_probe.py`
- experiment entry:
  - `experiments/stage_05_ef_core_probe/fmpc_ef_exploratory_probe.py`
- tests:
  - `tests/stage_05_ef_core_probe/`

## Working Rules

- keep the repository NumPy-first unless explicitly directed otherwise
- do not silently change mathematics without updating [spec_math.md](/e:/CodeSpace/PredictiveCoding/spec_math.md)
- keep public APIs batch-first
- keep randomness explicit with seeds
- preserve backward compatibility for named presets unless the task explicitly permits otherwise

For FMPC work:

- teacher-based FMPC remains frozen as baseline / diagnostic reference
- new FMPC work should not depend on teacher trajectories, teacher fixed points, or teacher-generated regression targets unless explicitly labeled as baseline or diagnostic
- do not casually reopen Stage 04 package-internal cone, successor-value, successor-increment, bootstrap, curriculum, or readout-alignment sweeps

## History And Older Guidance

This short README is now a landing page, not the full historical narrative.

Use these archives when you need older material:

- earlier long-form README content:
  - [archive/README_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/README_HISTORY.md)
- earlier long-form AGENTS wording:
  - [archive/AGENTS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/AGENTS_HISTORY.md)
- earlier result narrative:
  - [archive/RESULTS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/RESULTS_HISTORY.md)
- earlier plan chain:
  - [archive/PLANS_HISTORY.md](/e:/CodeSpace/PredictiveCoding/archive/PLANS_HISTORY.md)
- earlier validation chain:
  - [archive/validation_history.md](/e:/CodeSpace/PredictiveCoding/archive/validation_history.md)

## One-Sentence Project Summary

The repository currently treats `FMPC Stage 04 Incremental Bridge` as the frozen bridge result on `main`, while `FMPC Stage 05 EF Core Probe` is the active exploratory line being evaluated on mechanism-first evidence.
