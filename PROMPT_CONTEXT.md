# PROMPT_CONTEXT.md

This file is the shortest context sheet for drafting prompts to Codex or web GPT for this repository.

- It is a prompt-writing aid, not the highest-precedence authority.
- If documents conflict, follow:
  - `spec_math.md > validation.md > AGENTS.md > CURRENT_STATE.md > PLANS.md > README.md`

For prompt writing, treat this GitHub repository as the only authority for:

- code
- tests
- current defaults
- current state
- current decision chain

If the task also references Google Drive materials, treat them only as supplementary:

- `PredictiveCoding_papers` for papers and external references
- `PredictiveCoding_notes` for notes and design sketches
- `PredictiveCoding_results` for tables, charts, and reports

Do not let those Drive materials override the repository docs.

## Use This File For

Read this file first when you want GPT to help write a task prompt.

It is optimized for:

- current project state
- current forbidden moves
- current math read path
- prompt-writing guardrails

It is not meant to replace:

- `LLM_BRIEF.md` for low-context project orientation
- `CURRENT_STATE.md` for current active status
- `AGENTS.md` for standing repository rules
- `spec_math.md` and `specs/` for authoritative math

## Minimal Prompt-Drafting Read Order

1. `PROMPT_CONTEXT.md`
2. `LLM_BRIEF.md`
3. `AGENTS.md`
4. `CURRENT_STATE.md`
5. `spec_math.md` and the applicable file under `specs/` only if the task is math-sensitive
6. `PLANS.md`, `RESULTS.md`, or `validation.md` only if the task needs them
7. `archive/` only if historical evidence is truly necessary

For most prompt-drafting sessions, this is enough:

- `PROMPT_CONTEXT.md`
- `CURRENT_STATE.md`

Add `AGENTS.md` if the prompt may cause code changes.
Add `spec_math.md` and `specs/` only if the prompt is math-sensitive.
Add `validation.md` only if acceptance rules or gating language matter.
Add `PLANS.md` only if the prompt needs an explicit execution queue or charter.

## Current Project Reality

- active branch:
  - `main`
- active algorithmic line on `main`:
  - `FMPC Stage 04 Incremental Bridge`
- frozen bridge result on `main`:
  - `FMPC Stage 04 Incremental Bridge`
- current adopted bridge default:
  - `tf2_corrective_transport_terminal_angleclip_default`
- current canonical bridge identity default:
  - `feature_aware_tangents = false`
- current open exploratory line:
  - `FMPC Stage 05 EF Core Probe`

## Current Working Interpretation

- Stage 04 package-internal digging is closed from the current state
- Stage 04 is treated as the frozen bridge result on `main`
- the current Stage 05 baseline contract is the corrected residual MeanFlow v1 probe
- the fixed-budget Stage 05 control is the two-branch corrected residual MeanFlow v2 core
- the frozen-bridge vs corrected-core comparison is now complete
- the Stage 05 v1 vs v2 comparison is now complete
- the current comparison result says Stage 05 v2 improves mechanism magnitude over v1 on mechanism-first grounds
- the refreshed frozen-bridge vs Stage 05 v2 comparison is now complete
- the refreshed comparison supports using Stage 05 v2 as the current exploratory reference
- the dedicated Stage 05 v2 diagnostics now conclude:
  - `likely_undertrained`
  - the current low report-only accuracy is not primarily a selection-rule artifact
- the completed Stage 05 v2 longer-training validation now concludes:
  - the stronger same-family budget materially improves configured-step mechanism and report-only accuracy
  - the stronger budget still hits the final training boundary on every seed
- the completed Stage 05 v2 efficiency diagnostic at the fixed `1536`-epoch ceiling now concludes:
  - the tested same-family schedule change does not materially improve configured-step mechanism
  - it does not materially improve report-only accuracy
  - it does not materially narrow the gap to the contextual `3072`-epoch reference
- the first minimal working-hypothesis-driven `Stage 05 v3-A` candidate now exists:
  - `explicit transport-drift contract`
  - motivated by the hypothesis that the current residual target may entangle transport residual and anchor-drift residual too tightly
- the fixed-budget `v2 vs v3-A` comparison is now complete:
  - the v3-A candidate materially improves configured-step mechanism over the fixed-budget v2 reference
  - the fixed-budget `stage05_v3a_explicit_transport_drift_contract` result became the first active Stage 05 implementation branch
- the fixed-budget `v2 vs v3-A vs v3-B` comparison is now also complete:
  - the original v3-B candidate improves configured-step mechanism over the fixed-budget v2 control and directionally over v3-A
  - that original v3-B candidate did not materially improve enough over v3-A for promotion
  - `v3-B = trajectory curriculum contract`
- the narrow fixed-budget `v3-B` refinement diagnostic is now also complete:
  - the strongest tested refinement is `stage05_v3b_stronger_traj_curr_weight`
  - that refined v3-B candidate materially improves configured-step mechanism over the fixed-budget v3-A reference under the current threshold
- the fresh refined fixed-budget recompare is now complete:
  - `stage05_v3b_stronger_traj_curr_weight` is now the active fixed-budget Stage 05 improvement reference
  - the `v3-A -> refined v3-B` promotion question is closed
  - the repo no longer treats `Stage 05 v3-C` as planning-only
  - `v3-C = endpoint / semigroup consistency contract`
  - this charter is motivated by the working hypothesis that the current refined v3-B scaffold still lacks an explicit endpoint / semigroup consistency contract across split horizons
  - the first minimal diagnostic-only v3-C probe now also exists:
    - `stage05_v3c_endpoint_semigroup_consistency_contract`
  - the smoke-ready `v2 vs promoted-v3B vs v3-C` artifact now also exists under:
    - `outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_v3c_comparison/`
  - that smoke artifact only verifies wiring and deterministic comparison readiness
  - the real fixed-budget `v2 vs promoted-v3B vs v3-C` comparison is now also complete under:
    - `outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_v3c_fixed_budget_comparison/`
  - that fixed-budget comparison shows:
    - v3-C directionally improves configured-step mechanism over the promoted refined v3-B reference
    - v3-C improves contextual gap closure
    - v3-C does not yet materially displace the promoted refined v3-B reference
- the narrow fixed-budget `v3-C` refinement diagnostic is now also complete under:
  - `outputs/stage_05_ef_core_probe/stage05_v3c_refinement_diagnostic/`
- that refinement diagnostic shows:
  - the strongest tested refinement is `stage05_v3c_stronger_semigroup_weight`
  - that refined v3-C candidate materially improves configured-step mechanism over the promoted refined v3-B reference under the current threshold
  - it does not materially improve configured-step mechanism over the current v3-C control under the same threshold
  - it avoids an obvious report-only accuracy regression under the current Stage 05 rule
- the fresh fixed-budget `v2 vs promoted-v3B vs refined-v3C` recompare is now also complete under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_promoted_v3b_refined_v3c_fixed_budget_recompare/`
- that refined v3-C recompare shows:
  - `stage05_v3c_stronger_semigroup_weight` materially improves configured-step mechanism over the promoted refined v3-B reference under the current threshold
  - it materially improves configured-step mechanism over the fixed-budget v2 control
  - it avoids an obvious report-only accuracy regression
  - `stage05_v3c_stronger_semigroup_weight` is now the active fixed-budget Stage 05 improvement reference
  - the current repository-level structural interpretation is now:
    - `absorb_semigroup_into_main_trajectory_contract`
    - do not keep semigroup consistency framed as a permanently attached auxiliary-only term
    - do not yet refactor the main contract around endpoint / semigroup consistency alone
- the first fixed-budget fused-contract comparison now also exists under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_fused_contract_comparison/`
- that fused-contract comparison shows:
  - `stage05_v3c_fused_trajectory_semigroup_contract` keeps one-step and configured-step mechanism positive
  - it avoids an obvious report-only accuracy regression relative to `stage05_v3c_stronger_semigroup_weight`
  - it improves contextual gap closure slightly beyond the active refined v3-C reference
  - it does not materially displace `stage05_v3c_stronger_semigroup_weight` under the current threshold
- the fixed-budget endpoint-line midpoint comparison now also exists under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_endpoint_line_midpoint_contract_comparison/`
- that endpoint-line midpoint comparison shows:
  - `stage05_v3c_endpoint_line_midpoint_trajectory_contract` directionally improves configured-step mechanism over `stage05_v3c_stronger_semigroup_weight`
  - it improves contextual gap closure beyond the active refined v3-C reference
  - it avoids an obvious report-only accuracy regression
  - it does not materially displace `stage05_v3c_stronger_semigroup_weight` under the current threshold
- the fixed-budget endpoint-line continuation-blend comparison now also exists under:
  - `outputs/stage_05_ef_core_probe/stage05_v2_active_v3c_endpoint_line_continuation_blend_contract_comparison/`
- that endpoint-line continuation-blend comparison shows:
  - `stage05_v3c_endpoint_line_continuation_blend_trajectory_contract` directionally improves configured-step mechanism over `stage05_v3c_stronger_semigroup_weight`
  - it improves contextual gap closure beyond the active refined v3-C reference
  - it avoids an obvious report-only accuracy regression
  - it does not materially displace `stage05_v3c_stronger_semigroup_weight` under the current threshold
- do not write prompts as if Stage 05 has already replaced the active Stage 04 line on `main`

## Current Recommended Next Move

- keep the Stage 04 bridge result frozen on `main`
- do not write prompts that treat any current Stage 05 comparison as a default-replacement claim
- use the fixed-budget v2 result as the immediate control and `stage05_v3c_stronger_semigroup_weight` as the current fixed-budget improvement reference
- do not draft the next Stage 05 step as another pure same-family budget push
- do not draft the next Stage 05 step as another pure same-family efficiency tweak
- do not draft the next Stage 05 step as a repeat of the already-closed refined v3-C versus promoted refined v3-B promotion question
- do not draft the next Stage 05 step as if Stage 05 has already replaced frozen Stage 04 on `main`
- if Stage 05 continues, draft the next step as a narrow implementation-first continuation-target refinement above `stage05_v3c_endpoint_line_continuation_blend_trajectory_contract`, while keeping `stage05_v3c_stronger_semigroup_weight` as the active reference
- keep:
  - the fixed-budget v2 control
  - `stage05_v3c_stronger_semigroup_weight` as the active fixed-budget improvement reference
- do not frame the next step as another recompare or as an automatic new charter

## What Prompt Writers Must Not Reopen

Do not write prompts that reopen Stage 04 package-internal work unless new repo evidence explicitly does so.

That includes:

- cone-family / terminal-geometry follow-ups
- successor-value / successor-increment internal diagnostics
- readout-alignment follow-ups
- bootstrap-source bias follow-ups
- target-lag follow-ups
- bootstrap-to-identity curriculum follow-ups
- broad Stage 04 sweep requests after the saturation decision

Also do not casually open:

- AlphaFlow as a new training family
- `muPC`-style scaling in the mainline
- TF3

## Current Math Read Path

- baseline math root:
  - `spec_math.md`
- Stage 03 transport addendum:
  - `specs/stage_03_transport_core_v1.md`
- Stage 04 bridge addendum:
  - `specs/stage_04_incremental_bridge.md`
- Stage 05 corrected residual MeanFlow addendum:
  - `specs/stage_05_ef_core_probe.md`

## Current Prompt-Writing Style

Good prompts for this repository should:

- keep scope narrow
- say whether the pass is planning-only, docs-only, or implementation
- say whether code changes are allowed
- say which files should be updated first if planning is required
- keep experiments minimal and evidence-backed
- preserve the current frozen bridge result unless the task explicitly says otherwise
- distinguish clearly between frozen Stage 04 reality and Stage 05 exploration

## Common Prompt Mistakes To Avoid

- writing as if Stage 04 is still open for package-internal repair
- writing as if Stage 05 already replaced the active line on `main`
- using `teacher-free` as the stage name instead of the current stage naming scheme
- asking GPT to read the entire archive by default
- opening a broad experimental zoo when the repo state only justifies one narrow pass
- treating report-only accuracy as the Stage 05 acceptance gate

## Copy-Paste Prompt Skeleton

Use the skeleton below when you want web GPT to draft a prompt for Codex.

Rules for using it:

- keep only the blocks that matter for the current task
- replace bracketed placeholders with concrete repo facts
- do not add broad sweeps if the repo state only justifies a narrow pass
- if the task is docs-only or planning-only, say that explicitly
- if the task should not change defaults, say that explicitly

```text
Goal:
Use branch `main` only. [Describe the exact task in one sentence.]

Read order:
1. README.md
2. AGENTS.md
3. CURRENT_STATE.md
4. PLANS.md
5. spec_math.md
6. [applicable file under specs/ if stage-specific math matters]
7. validation.md

Repository precedence:
spec_math.md > validation.md > AGENTS.md > CURRENT_STATE.md > PLANS.md > README.md

Current repo state to respect:
- Active branch: `main`
- Active algorithmic line on `main` remains:
  - `FMPC Stage 04 Incremental Bridge`
- Frozen Stage 04 bridge result on `main`:
  - `tf2_corrective_transport_terminal_angleclip_default`
- Current canonical bridge identity default:
  - `feature_aware_tangents = false`
- Current exploratory line:
  - `FMPC Stage 05 EF Core Probe`
- [Add only the few current facts that materially constrain this task.]

Do not reopen:
- [List only the blocked or sealed families relevant to this task.]

Task:
1. [First concrete step.]
2. [Second concrete step.]
3. [If code is allowed, list the smallest files to touch.]
4. [If metrics or outputs are required, list them explicitly.]

Constraints:
- NumPy-first only
- preserve backward compatibility for existing named presets
- keep the patch narrow and reversible
- [Add task-specific constraints only if they are real.]

Validation:
- [If docs-only: run a markdown consistency pass and report it.]
- [If code changes: run the narrowest relevant pytest subset.]
- [If experiments are needed: run only the smallest suite needed for a real answer.]

Done when:
- [State the exact question that must be answered.]
- [State which docs must agree afterward, if applicable.]
- [State the minimum required deliverables.]
```

## Recommended Task Modes

Use one of these three headers near the top of the prompt so Codex can choose the right level of action immediately.

### Implementation Pass

Use this when you want code changes.

```text
This pass is implementation-first.
- Code changes are allowed.
- Keep experiments minimal.
- Update docs only if needed for factual consistency.
```

### Planning-Only Pass

Use this when you want a decision memo, charter, or next-step design without new experiment code.

```text
This pass is planning-only unless a tiny factual doc sync is required.
- Do not add new experimental code.
- Prefer existing evidence over new search.
```

### Docs-Only Pass

Use this when you want cleanup, consistency, naming, or structure changes.

```text
This pass is docs-only.
- Do not change algorithmic code.
- Do not change validation semantics unless the task explicitly asks for it.
```

## Minimal Fill-In Checklist

Before sending a prompt drafted from this file, make sure it answers these:

- Is the task implementation, planning-only, or docs-only?
- Does it stay inside Stage 04, or does it explicitly leave the frozen bridge package?
- Does it respect the current frozen Stage 04 result?
- Does it avoid reopening sealed Stage 04 families without new evidence?
- If math matters, did it point to the right `specs/` addendum?
- Did it define a narrow success condition instead of a broad search?

## If You Need More Than This File

- project map and document finder:
  - `README.md`
- low-context repository orientation:
  - `LLM_BRIEF.md`
- standing coding and workflow rules:
  - `AGENTS.md`
- current active state and next move:
  - `CURRENT_STATE.md`
- current forward plan:
  - `PLANS.md`
- current result snapshot:
  - `RESULTS.md`
- current validation semantics:
  - `validation.md`
