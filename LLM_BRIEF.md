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

Default completion workflow:

- after any completed code or documentation change, sync to GitHub and Google Drive by default
- GitHub is the authoritative code/doc update surface
- Google Drive is the supplementary report/artifact update surface

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
- adopted implemented line on `main`:
  - `FMPC Stage 04 Incremental Bridge`
- current adopted bridge default:
  - `tf2_corrective_transport_terminal_angleclip_default`
- current canonical bridge identity default:
  - `feature_aware_tangents = false`
- current frozen mechanism-reference stage:
  - `FMPC Stage 05 EF Core Probe`
- current active forward charter:
  - `FMPC Stage 06 Low-Budget Efficiency`

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

The current open forward work is no longer another Stage 05 geometry pass.

The current active forward charter is:

- `FMPC Stage 06 Low-Budget Efficiency`

Current known state:

- implementation exists
- Stage 05 remains the mechanism evidence base
- the `1536 -> 3072` Stage 05 v2 budget push still improved configured-step mechanism and still hit the final epoch on every seed
- the fixed-`1536` same-family efficiency tweak was nearly a no-op
- `stage05_v3c_stronger_semigroup_weight` is the strongest high-budget Stage 05 mechanism reference
- the active refined v3-C line is expensive enough that it should not be treated as an efficiency win by default
- the narrow v3-C continuation / midpoint / coupled / precision / scaled micro-family is locally saturated
- Stage 06 is now opened because the remaining question is:
  - low-budget
  - low-compute
  - matched-budget
  - viability
- the first implemented Stage 06 line is:
  - `stage06_v1_objective_curriculum_energydrop_default`
- the current authoritative Stage 06 baseline artifact is:
  - `outputs/stage_06_low_budget_efficiency/stage06_v1_low_budget_comparison/stage06_v1_post_semantic_alignment_rebaseline/`
- the current authoritative Stage 06 v2 follow-up artifact is:
  - `outputs/stage_06_low_budget_efficiency/stage06_v2_low_budget_comparison/stage06_v2_initial_authoritative_comparison/`
- the post-semantic-alignment Stage 06 baseline passed Tier 1 viability but still failed the Tier 2 main gate against the matched-budget Stage 05 control
- task accuracy remains secondary, but cost is no longer secondary
- the current single-axis implemented follow-up surface is:
  - `stage06_v2_persistent_overlap_objective_curriculum_energydrop_default`
- the changed axis is:
  - replace the Stage 06 v1 hard late `L_traj -> L_semi` handoff with a persistent overlap objective contract
- current status:
  - v2 authoritative comparison artifact now exists
  - v2 remains non-promoted after the first authoritative low-budget comparison
  - v1 remains the authoritative Stage 06 baseline
  - do not auto-open a `v2.1` / `v2.2` retuning pass from the current state
- current post-v2 planning conclusion is:
  - `A2_aggregate_contract_itself_is_now_the_more_likely_bottleneck`
  - `open_new_stage06_contract_family`
  - planning-only provisional family:
    - `stage06_B1_split_update_contract_family`
  - planning-only first probe:
    - `stage06_v3_split_update_objective_contract_default`
- this does not mean:
  - restoring Stage 05 v3-A branchwise supervision
  - reopening the saturated Stage 05 geometry micro-family

## Current Recommended Next Move

- keep the Stage 04 bridge result frozen on `main`
- do not treat any current Stage 05 comparison as a replacement claim against the frozen bridge result
- keep Stage 05 as the high-budget mechanism reference stage, not the efficiency mainline
- use the fixed-budget v2 result and `stage05_v3c_stronger_semigroup_weight` as the Stage 05 control/reference pair when needed
- do not continue pure same-family budget escalation from the current state
- do not continue another continuation / midpoint / coupled / precision / scaled Stage 05 micro-variant
- move the next forward pass into Stage 06:
  - low-budget-first
  - matched-budget
  - efficiency-first
  - above the saturated Stage 05 micro-family
- current tested Stage 06 follow-up:
  - `stage06_v2_persistent_overlap_objective_curriculum_energydrop_default`
  - remained inside `stay_within_stage06_A2_family`
  - keeps A2 two-branch parameterization and target-builder reuse
  - does not restore Stage 05 v3-A branchwise supervision
  - is artifact-tested but not promoted over the v1 baseline
- current planning-only next family:
  - `open_new_stage06_contract_family`
  - provisional family:
    - `stage06_B1_split_update_contract_family`
  - single primary changed axis:
    - replace the simultaneous weighted-sum aggregate objective update path with a split-update supervision contract over the same reused Stage 05 targets

## Current Code Entry Points

- frozen bridge implementation:
  - `src/pc/stage_04_incremental_bridge/fmpc_tf2.py`
- exploratory core probe:
  - `src/pc/stage_05_ef_core_probe/fmpc_ef_exploratory_probe.py`
- Stage 06 low-budget efficiency probe:
  - `src/pc/stage_06_low_budget_efficiency/fmpc_stage06_objective_curriculum.py`

## Current Math Entry Points

- baseline root spec:
  - `spec_math.md`
- Stage 03 transport addendum:
  - `specs/stage_03_transport_core_v1.md`
- Stage 04 bridge addendum:
  - `specs/stage_04_incremental_bridge.md`
- Stage 05 corrected residual MeanFlow addendum:
  - `specs/stage_05_ef_core_probe.md`
- Stage 06 low-budget efficiency addendum:
  - `specs/stage_06_low_budget_efficiency.md`
- Stage 06 planning-only next-family addendum:
  - `specs/stage_06_split_update_contract_family.md`

## Current Artifact Roots

- Stage 04 authority artifacts:
  - `outputs/stage_04_incremental_bridge/`
- Stage 05 key evidence artifacts:
  - `outputs/stage_05_ef_core_probe/stage05_v2_budget_push_validation_1536_to_3072/`
  - `outputs/stage_05_ef_core_probe/stage05_v2_efficiency_diagnostic_at_1536/`
  - `outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_refined_v3c_fixed_budget_recompare/`
  - `outputs/stage_05_ef_core_probe/stage05_v3c_continuation_strength_diagnostic/`
- Stage 06 authoritative baseline artifact:
  - `outputs/stage_06_low_budget_efficiency/stage06_v1_low_budget_comparison/stage06_v1_post_semantic_alignment_rebaseline/`
- Stage 06 v2 authoritative follow-up artifact:
  - `outputs/stage_06_low_budget_efficiency/stage06_v2_low_budget_comparison/stage06_v2_initial_authoritative_comparison/`

## History

Long historical context now lives in:

- `archive/AGENTS_HISTORY.md`
- `archive/README_HISTORY.md`
- `archive/RESULTS_HISTORY.md`
- `archive/PLANS_HISTORY.md`
- `archive/validation_history.md`
