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
- Current Stage 05 note:
  - there is no separate Stage 05 math addendum yet
  - the current probe reuses the baseline substrate plus the Stage 03 average-velocity transport contract where applicable

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
- target construction remains artifact-independent
- the first canonical probe shows positive mechanism signal:
  - one-step validation energy improves vs identity
  - configured two-step validation energy improves vs identity
  - configured two-step fixed-point residual improves vs identity
- Stage 05 evaluation remains mechanism-first:
  - task accuracy is report-only and is not the current acceptance gate
- task accuracy remains report-only and is still well below the frozen Stage 04 bridge result

## Current Recommendation

- Keep the Stage 04 bridge result frozen on `main`.
- Do not open another package-internal Stage 04 diagnostic suite from this state.
- Run the frozen-bridge vs exploratory-core comparison next.
- Use that comparison to decide whether Stage 05 earns a v2 charter.

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
