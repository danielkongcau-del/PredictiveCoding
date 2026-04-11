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
- the current narrow Stage 05 candidate is the two-branch corrected residual MeanFlow v2 core
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
- do not write prompts as if Stage 05 has already replaced the active Stage 04 line on `main`

## Current Recommended Next Move

- keep the Stage 04 bridge result frozen on `main`
- do not write prompts that treat any current Stage 05 comparison as a default-replacement claim
- use the Stage 05 v2 two-branch core as the current exploratory reference
- if Stage 05 continues, draft the next narrow step as another budget push on the current v2 reference before inventing a new family or opening a true v3 charter

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
