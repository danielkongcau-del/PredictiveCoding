# PROMPT_CONTEXT.md

This file is the shortest context sheet for drafting prompts to Codex or web GPT for this repository.

- It is a prompt-writing aid, not the highest-precedence authority.
- If documents conflict, follow:
  - `spec_math.md > validation.md > AGENTS.md > CURRENT_STATE.md > PLANS.md > README.md`

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

## Current Project Reality

- active branch:
  - `main`
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
- the current open question is whether Stage 05 earns a v2 charter
- do not write prompts as if Stage 05 has already replaced the active Stage 04 line on `main`

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
- current Stage 05 note:
  - no separate Stage 05 addendum exists yet
  - the current probe reuses the baseline substrate plus the Stage 03 average-velocity contract where applicable

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
