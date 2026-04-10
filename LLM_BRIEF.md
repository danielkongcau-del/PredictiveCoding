# LLM_BRIEF.md

This file is the low-context entry point for web GPT sessions and prompt drafting.

- It is a convenience summary, not the highest-precedence authority.
- If anything here conflicts with the main repository documents, follow:
  - `spec_math.md > validation.md > AGENTS.md > CURRENT_STATE.md > PLANS.md > README.md`

## Suggested Read Order For Web GPT

Start here, then expand only as needed:

1. `LLM_BRIEF.md`
2. `AGENTS.md`
3. `CURRENT_STATE.md`
4. the relevant section of `spec_math.md`
5. `PLANS.md` or `validation.md` only if the task needs current plan or evidence detail
6. `archive/` only if historical context is required

## Active Repository State

- Active branch:
  - `main`
- Active algorithmic line:
  - `FMPC Stage 04 Incremental Bridge`
- Current adopted bridge default on `main`:
  - `tf2_corrective_transport_terminal_angleclip_default`
- Current canonical bridge identity default:
  - `feature_aware_tangents = false`

## FMPC Stage Map

- `stage_01_reference_prep/` -> `FMPC Stage 01 Reference Prep`
- `stage_02_interval_velocity/` -> `FMPC Stage 02 Interval Velocity Exploration`
- `stage_03_transport_core_v1/` -> `FMPC Stage 03 Transport Core v1`
- `stage_04_incremental_bridge/` -> `FMPC Stage 04 Incremental Bridge`
- `stage_05_ef_core_probe/` -> `FMPC Stage 05 EF Core Probe`

## What Is Closed

Inside the current Stage 04 corrective bridge package:

- package-internal digging is closed from the current state
- unified-cone / terminal-cone family is locally saturated
- selector / checkpoint is not the main limiter
- simple head-fit / separability is not the main limiter
- readout alignment is sealed as a no-op
- bootstrap-source bias, target lag, and bootstrap-to-identity curriculum do not provide a credible reopen path
- the late-rollout successor-value / successor-increment line ends in a strengthened formulation-level blocker

Working interpretation:

- the adopted corrective bridge package is locally saturated under the current selector/gate contract
- do not reopen package-internal Stage 04 cone, successor-value, successor-increment, curriculum, or bootstrap sweeps unless genuinely new evidence appears

## What Is Open

The active open work is no longer a Stage 04 package-internal fix.

The current exploratory line is:

- `FMPC Stage 05 EF Core Probe`

Current exploratory probe status:

- implementation exists
- target construction is artifact-independent
- mechanism metrics are positive on the first probe:
  - one-step validation energy improves vs identity
  - configured two-step validation energy improves vs identity
  - configured two-step fixed-point residual improves vs identity
- task accuracy is still low and remains report-only

## Current Recommended Next Move

- keep the Stage 04 bridge result frozen on `main`
- run the frozen-bridge vs exploratory-core comparison next
- use that comparison to decide whether Stage 05 earns a v2 charter

## Forbidden Moves From Current State

Do not do these by default:

- reopen Stage 04 package-internal cone-family work
- reopen successor-value / successor-increment internal sweeps
- reopen readout-alignment, bootstrap-source, target-lag, or curriculum follow-ups
- introduce AlphaFlow, `muPC`-style bridge-stage scaling, or TF3 code as a casual next step
- rewrite the current active line on `main` as if Stage 05 has already replaced Stage 04

## Key Files

- active bridge implementation:
  - `src/pc/stage_04_incremental_bridge/fmpc_tf2.py`
- exploratory core probe:
  - `src/pc/stage_05_ef_core_probe/fmpc_ef_exploratory_probe.py`
- current frozen results summary:
  - `RESULTS.md`
- current operational summary:
  - `CURRENT_STATE.md`
- current active plan:
  - `PLANS.md`
- current active validation contract:
  - `validation.md`

## Key Artifacts

- Stage 04 authority artifacts:
  - `outputs/stage_04_incremental_bridge/`
- Stage 05 exploratory artifact:
  - `outputs/stage_05_ef_core_probe/fmpc_ef_exploratory_probe/`

## Where History Went

Historical long-form context now lives in:

- `archive/AGENTS_HISTORY.md`
- `archive/README_HISTORY.md`
- `archive/RESULTS_HISTORY.md`
- `archive/PLANS_HISTORY.md`
- `archive/validation_history.md`

Use those only when a prompt truly needs older experiment chains or legacy plan history.
