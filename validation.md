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

- the exploratory line has enough multiseed mechanism-first signal to justify a narrow Stage 05 v2 charter
- it still does not justify replacing the frozen bridge result on `main`

## Current Recommended Validation Move

The next validation move is:

- draft the next narrow Stage 05 v2 charter from the completed comparison evidence

That next step should preserve:

- the frozen Stage 04 bridge result on `main`
- the mechanism-first validation contract for Stage 05

It should not be framed as:

- a Stage 04 package-internal reopening
- a new adoption sweep inside the saturated corrective bridge package

## Relevant Active Artifacts

- frozen Stage 04 authority artifact tree:
  - `outputs/stage_04_incremental_bridge/`
- Stage 05 exploratory probe:
  - `outputs/stage_05_ef_core_probe/fmpc_ef_exploratory_probe/`
- Stage 05 frozen-bridge comparison:
  - `outputs/stage_05_ef_core_probe/frozen_bridge_vs_corrected_core_comparison/`

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
