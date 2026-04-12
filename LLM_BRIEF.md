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
- the fixed-budget Stage 05 control is the two-branch corrected residual MeanFlow v2 core
- target construction is artifact-independent
- the first probe shows positive mechanism signal on energy and fixed-point residual
- the formal frozen-bridge vs corrected-core comparison now exists
- the formal Stage 05 v1 vs v2 comparison now exists
- that comparison says Stage 05 v2 improves mechanism magnitude over v1 on mechanism-first evidence
- the refreshed frozen-bridge vs Stage 05 v2 comparison now exists
- that refreshed comparison supports using Stage 05 v2 as the new exploratory reference
- the dedicated Stage 05 v2 diagnostics now say the current narrow diagnosis is `likely_undertrained`
- the completed Stage 05 v2 longer-training validation now says the stronger same-family budget materially improves configured-step mechanism and report-only accuracy, but still hits the final training boundary on every seed
- the completed Stage 05 v2 efficiency diagnostic at the fixed `1536`-epoch ceiling now says the tested same-family schedule change does not materially improve configured-step mechanism or report-only accuracy and does not materially narrow the gap to the contextual `3072`-epoch reference
- the completed fixed-budget Stage 05 `v2 vs v3-A` comparison now says the v3-A contract materially improves configured-step mechanism over the fixed-budget v2 reference
- the completed refined fixed-budget recompare now says `stage05_v3b_stronger_traj_curr_weight` materially improves configured-step mechanism over both the fixed-budget v2 control and the fixed-budget v3-A reference
- task accuracy is still low and remains report-only

## Current Recommended Next Move

- keep the Stage 04 bridge result frozen on `main`
- do not treat any current Stage 05 comparison as a replacement claim against the frozen bridge result
- use the fixed-budget v2 result as the immediate control and `stage05_v3b_stronger_traj_curr_weight` as the current fixed-budget improvement reference
- do not continue pure same-family budget escalation from the current state
- treat the fixed-budget `stage05_v3a_explicit_transport_drift_contract` result as the previous comparison reference
- do not open a Stage 05 v3-C charter automatically from this recompare alone

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
- Stage 05 v2 diagnostics:
  - `outputs/stage_05_ef_core_probe/stage05_v2_diagnostics/`
- Stage 05 v2 longer-training validation:
  - `outputs/stage_05_ef_core_probe/stage05_v2_longer_training_validation/`

## History

Long historical context now lives in:

- `archive/AGENTS_HISTORY.md`
- `archive/README_HISTORY.md`
- `archive/RESULTS_HISTORY.md`
- `archive/PLANS_HISTORY.md`
- `archive/validation_history.md`
