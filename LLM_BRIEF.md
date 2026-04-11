# LLM_BRIEF.md

This file is the low-context starting point for web GPT sessions.

If the task is specifically to draft a prompt for Codex or web GPT, read `PROMPT_CONTEXT.md` first.

- It is a convenience map, not the highest-precedence authority.
- If documents conflict, follow:
  - `spec_math.md > validation.md > AGENTS.md > CURRENT_STATE.md > PLANS.md > README.md`

Treat this GitHub repository as the authority for:

- code
- current normative docs
- current project state
- current decision chain

If Google Drive materials are also available, treat them as supplementary:

- `PredictiveCoding_papers` for external references
- `PredictiveCoding_notes` for supplementary notes
- `PredictiveCoding_results` for heavy result materials

## Minimal Read Order

Start here, then expand only as needed:

1. `LLM_BRIEF.md`
2. `AGENTS.md`
3. `CURRENT_STATE.md`
4. the relevant part of `spec_math.md` and the applicable addendum under `specs/`
5. `PLANS.md`, `RESULTS.md`, or `validation.md` only if the task needs them
6. `archive/` only if historical context is truly required

For most new-session orientation, `LLM_BRIEF.md` plus `CURRENT_STATE.md` is enough.

Open `AGENTS.md` if the task may lead to code changes.
Open `spec_math.md` and `specs/` only if the task is math-sensitive.
Open `validation.md` only if acceptance rules matter.

## Document Finder

If you want a specific kind of information, read this file:

- shortest prompt-writing context:
  - `PROMPT_CONTEXT.md`
- current active line, current defaults, current next move:
  - `CURRENT_STATE.md`
- current forward plan and immediate queue:
  - `PLANS.md`
- current frozen result snapshot:
  - `RESULTS.md`
- current validation / acceptance contract:
  - `validation.md`
- mathematical definitions and update equations:
  - `spec_math.md`
  - plus the applicable file under `specs/`
- standing coding rules and non-negotiable constraints:
  - `AGENTS.md`
- repository layout, stage map, and entry navigation:
  - `README.md`
- older long-form history:
  - `archive/`

## Current Repository State

- active branch:
  - `main`
- active algorithmic line:
  - `FMPC Stage 04 Incremental Bridge`
- current adopted bridge default:
  - `tf2_corrective_transport_terminal_angleclip_default`
- current canonical bridge identity default:
  - `feature_aware_tangents = false`
- current exploratory line:
  - `FMPC Stage 05 EF Core Probe`

## What Is Closed

Inside Stage 04:

- package-internal digging is closed from the current state
- cone-family work is locally saturated
- selector / checkpoint is not the main limiter
- simple head-fit / separability is not the main limiter
- readout alignment is sealed as a no-op
- bootstrap-source, target-lag, and curriculum follow-ups do not reopen the package
- the late-rollout successor-value / successor-increment line ends in a strengthened formulation-level blocker

Working interpretation:

- the adopted corrective bridge package is locally saturated under the current selector/gate contract

## What Is Open

The current open work is:

- `FMPC Stage 05 EF Core Probe`

Current known state:

- implementation exists
- the current Stage 05 baseline contract is the corrected residual MeanFlow v1 probe
- the current narrow Stage 05 candidate is the two-branch corrected residual MeanFlow v2 core
- target construction is artifact-independent
- the first probe shows positive mechanism signal on energy and fixed-point residual
- the formal frozen-bridge vs corrected-core comparison now exists
- the formal Stage 05 v1 vs v2 comparison now exists
- that comparison says Stage 05 v2 improves mechanism magnitude over v1 on mechanism-first evidence
- the refreshed frozen-bridge vs Stage 05 v2 comparison now exists
- that refreshed comparison supports using Stage 05 v2 as the new exploratory reference
- task accuracy is still low and remains report-only

## Current Recommended Next Move

- keep the Stage 04 bridge result frozen on `main`
- do not treat any current Stage 05 comparison as a replacement claim against the frozen bridge result
- use the Stage 05 v2 two-branch core as the current exploratory reference
- define the next narrow Stage 05 mechanism-first step from the v2 reference

## Current Code Entry Points

- frozen bridge implementation:
  - `src/pc/stage_04_incremental_bridge/fmpc_tf2.py`
- exploratory core probe:
  - `src/pc/stage_05_ef_core_probe/fmpc_ef_exploratory_probe.py`

## Current Math Entry Points

- baseline root spec:
  - `spec_math.md`
- Stage 03 transport addendum:
  - `specs/stage_03_transport_core_v1.md`
- Stage 04 bridge addendum:
  - `specs/stage_04_incremental_bridge.md`
- Stage 05 corrected residual MeanFlow addendum:
  - `specs/stage_05_ef_core_probe.md`

## Current Artifact Roots

- Stage 04 authority artifacts:
  - `outputs/stage_04_incremental_bridge/`
- Stage 05 exploratory probe:
  - `outputs/stage_05_ef_core_probe/fmpc_ef_exploratory_probe/`
- Stage 05 comparison:
  - `outputs/stage_05_ef_core_probe/frozen_bridge_vs_corrected_core_comparison/`
- Stage 05 refreshed bridge vs v2 comparison:
  - `outputs/stage_05_ef_core_probe/frozen_bridge_vs_two_branch_corrected_core_comparison/`

## History

Long historical context now lives in:

- `archive/AGENTS_HISTORY.md`
- `archive/README_HISTORY.md`
- `archive/RESULTS_HISTORY.md`
- `archive/PLANS_HISTORY.md`
- `archive/validation_history.md`
