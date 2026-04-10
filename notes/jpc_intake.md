# JPC Intake for Phase TF2 / iFMPC Bridge

Status: read-only intake of `third_party/jpc` as a reference codebase. No files in
`third_party/jpc` were modified. No PredictiveCoding mainline code was changed in
this intake pass.

## Scope and stance

The current PredictiveCoding repo remains NumPy-first and does not migrate its
mainline to JAX in this stage. JPC is being treated as:

- a reference implementation for predictive-coding API design
- a reference source for muPC-related parameterisation and inference diagnostics
- a possible probe target for TF2 bridge experiments

It is **not** being adopted as a new mainline substrate.

## Files inspected

Top level:

- `third_party/jpc/README.md`
- `third_party/jpc/pyproject.toml`

Docs:

- `third_party/jpc/docs/index.md`
- `third_party/jpc/docs/basic_usage.md`
- `third_party/jpc/docs/advanced_usage.md`
- `third_party/jpc/docs/api/Initialisation.md`
- `third_party/jpc/docs/api/Discrete updates.md`
- `third_party/jpc/docs/api/Continuous-time Inference.md`
- `third_party/jpc/docs/api/Training.md`
- `third_party/jpc/docs/api/Theoretical tools.md`
- `third_party/jpc/docs/examples`

Core package:

- `third_party/jpc/jpc/__init__.py`
- `third_party/jpc/jpc/_utils.py`
- `third_party/jpc/jpc/_train.py`
- `third_party/jpc/jpc/_test.py`
- `third_party/jpc/jpc/_core/_init.py`
- `third_party/jpc/jpc/_core/_infer.py`
- `third_party/jpc/jpc/_core/_updates.py`
- `third_party/jpc/jpc/_core/_grads.py`
- `third_party/jpc/jpc/_core/_energies.py`
- `third_party/jpc/jpc/_core/_analytical.py`

Examples / experiments:

- `third_party/jpc/examples/`
  - `discriminative_pc.ipynb`
  - `supervised_generative_pc.ipynb`
  - `unsupervised_generative_pc.ipynb`
  - `hybrid_pc.ipynb`
  - `mupc.ipynb`
  - `epc.ipynb`
  - `bidirectional_pc.ipynb`
- `third_party/jpc/experiments/library_paper/train_mlp.py`
- `third_party/jpc/experiments/library_paper/utils.py`
- `third_party/jpc/experiments/mupc_paper/README.md`
- `third_party/jpc/experiments/mupc_paper/train_pcn.py`
- `third_party/jpc/experiments/mupc_paper/train_pcn_no_metrics.py`
- `third_party/jpc/experiments/mupc_paper/analyse_activity_hessian.py`
- `third_party/jpc/experiments/mupc_paper/utils.py`
- `third_party/jpc/experiments/limits_paper/README.md`
- `third_party/jpc/experiments/limits_paper/train.py`
- `third_party/jpc/experiments/limits_paper/theory_utils.py`
- `third_party/jpc/experiments/limits_paper/plot_infer_convergence.py`

Tests:

- `third_party/jpc/tests/README.md`
- `third_party/jpc/tests/test_init.py`
- `third_party/jpc/tests/test_infer.py`
- `third_party/jpc/tests/test_updates.py`
- `third_party/jpc/tests/test_train.py`
- `third_party/jpc/tests/test_utils.py`
- `third_party/jpc/tests/test_analytical.py`
- `third_party/jpc/tests/test_energies.py`
- `third_party/jpc/tests/test_grads.py`

## High-level library picture

JPC is a JAX + Equinox + Optax + Diffrax predictive-coding library. Its design
is strongly functional:

- model and optimiser state are explicit inputs / outputs
- activity inference can be solved either by discrete activity updates or by
  continuous-time ODE integration
- parameter updates are cleanly separated from activity updates
- theoretical and numerical diagnostics are exposed as first-class utilities

This is relevant to TF2 because the bridge stage likely wants:

- clean separation between feedforward init, inference, and param updates
- explicit support for multiple inference schedulers
- explicit diagnostics for inference conditioning

## Live probe update

The completed local probe in `outputs/tf2/tf2_jpc_probe/summary.json` should now be
treated as the operative TF2 interpretation aid.

What the live probe currently supports:

- JPC remains reference-only for TF2
- many-step inference remains materially better than 1-step in:
  - the current PredictiveCoding layered PC substrate
  - JPC standard PC
- the probe does not provide strong evidence that muPC-style scaling should now
  replace incremental scheduling as the main TF2 focus

What this means for the current TF2 stage:

- TF2 should continue to prioritize incremental scheduling and corrective
  transport
- muPC-style scaling should remain a future candidate substrate mechanism
- the current TF2 mainline should not introduce JPC runtime dependence or a new
  scaling mechanism yet

## 1. Relevant JPC modules and exact APIs

### Feedforward activity initialisation

Primary APIs:

- `jpc._core._init.init_activities_with_ffwd`
- `jpc._core._init.init_activities_from_normal`
- `jpc._core._init.init_activities_with_amort`
- `jpc._core._init.init_epc_errors`

Exposed via:

- `jpc.init_activities_with_ffwd`
- `jpc.init_activities_from_normal`
- `jpc.init_activities_with_amort`
- `jpc.init_epc_errors`

Notes:

- `init_activities_with_ffwd(...)` is the standard feedforward activity initialiser.
- It explicitly supports `param_type in {"sp", "mupc", "ntp"}` and optional
  `skip_model`.
- `init_activities_with_amort(...)` is hybrid-PC specific and relevant only as a
  reference for amortised initialisation, not as a direct TF2 dependency.

### Activity inference updates

Primary APIs:

- `jpc._core._updates.update_pc_activities`
- `jpc._core._grads.compute_pc_activity_grad`
- `jpc._core._grads.neg_pc_activity_grad`

Also relevant variant APIs:

- `update_bpc_activities`
- `update_epc_errors`
- `update_pdm_activities`

Notes:

- `update_pc_activities(...)` applies a single discrete optimiser step to
  activities using an Optax optimiser.
- `compute_pc_activity_grad(...)` computes the energy and `dF/dz`.
- `neg_pc_activity_grad(...)` defines the ODE vector field `-dF/dz`.
- `update_epc_errors(...)` shows an alternative inference variable choice:
  error variables instead of activity variables.

### Parameter updates

Primary APIs:

- `jpc._core._updates.update_pc_params`
- `jpc._core._grads.compute_pc_param_grads`

High-level wrapper:

- `jpc._train.make_pc_step`

Related variants:

- `update_bpc_params`
- `update_epc_params`
- `update_pdm_params`
- `compute_hpc_param_grads`

Notes:

- JPC cleanly separates parameter updates from activity updates.
- In the standard path, parameters are updated **after** inference reaches a
  numerical solution.

### ODE / solver-based inference

Primary APIs:

- `jpc._core._infer.solve_inference`
- `jpc._core._grads.neg_pc_activity_grad`

Utility / experiment support:

- `third_party/jpc/experiments/library_paper/utils.py:get_ode_solver`
- `third_party/jpc/experiments/library_paper/train_mlp.py`

Notes:

- `solve_inference(...)` wraps `diffrax.diffeqsolve(...)`.
- Default solver is `Heun`, with configurable solver, step size, and step-size
  controller.
- The library paper experiment explicitly compares solver families such as
  Euler, Heun, Midpoint, Ralston, Bosh3, Tsit5, Dopri5, and Dopri8.

### muPC parameterisation or experiments

Core parameterisation support:

- `jpc._core._energies._get_param_scalings`
- `jpc._utils.make_mlp`
- `jpc._utils.make_skip_model`
- many core functions accept `param_type="mupc"`

Main muPC experiment files:

- `third_party/jpc/experiments/mupc_paper/train_pcn.py`
- `third_party/jpc/experiments/mupc_paper/train_pcn_no_metrics.py`
- `third_party/jpc/experiments/mupc_paper/analyse_activity_hessian.py`
- `third_party/jpc/experiments/mupc_paper/utils.py`
- `third_party/jpc/examples/mupc.ipynb`

Notes:

- `make_mlp(..., param_type="mupc")` changes the layer parameterisation to align
  with the muPC scaling conventions.
- `_get_param_scalings(...)` is the exact scaling switch for `sp`, `mupc`, and
  `ntp`.
- `make_skip_model(...)` is directly tied to the skip-connection pattern used in
  the muPC experiments.

### Diagnostics relevant to forward-init stability or inference conditioning

Core utility / theory APIs:

- `jpc._utils.compute_infer_energies`
- `jpc._utils.compute_activity_norms`
- `jpc._core._analytical.compute_linear_activity_hessian`
- `jpc._core._analytical.compute_linear_activity_solution`
- `jpc._core._analytical.linear_equilib_energy`
- `jpc._core._analytical.update_linear_equilib_energy_params`

Experiment-side diagnostics:

- `third_party/jpc/experiments/mupc_paper/utils.py:compute_hessian_eigens`
- `third_party/jpc/experiments/mupc_paper/utils.py:compute_cond_num`
- `third_party/jpc/experiments/mupc_paper/analyse_activity_hessian.py`
- `third_party/jpc/experiments/mupc_paper/train_pcn.py`
- `third_party/jpc/experiments/limits_paper/plot_infer_convergence.py`
- `third_party/jpc/experiments/library_paper/train_mlp.py`

Notes:

- JPC already has a concrete diagnostics stack for:
  - per-step inference energies
  - activity norms
  - Hessian eigenvalues
  - Hessian condition numbers
  - analytical activity solutions for linear nets
  - closed-form equilibrium energy comparisons
- This is the most obviously useful part of JPC for TF2 bridge work.

## 2. Does JPC already contain iPC-like scheduling?

### Short answer

For the standard PC path, **no**. JPC does not currently expose an iPC-like
training schedule in which state and weight updates are interleaved online
during the same inference rollout.

### What the standard path actually does

The standard high-level schedule in `jpc._train.make_pc_step(...)` is:

1. initialise activities
2. solve inference
3. update parameters at the inferred activities

The same separation appears in the lower-level experimental loops:

- `experiments/library_paper/train_mlp.py`
- `experiments/limits_paper/train.py`
- `experiments/mupc_paper/train_pcn.py`

In those files, the pattern is consistently:

- reset / initialise activities
- run many activity updates or a continuous solver
- call `update_pc_params(...)` once after inference

### Online or per-step parameter updates during inference

No direct implementation was found for:

- per-inference-step parameter updates
- state/weight interleaving inside a single standard PC inference rollout
- an explicit "incremental predictive coding" scheduler

### Closest things in JPC

Closest but not equivalent:

- `update_pc_activities(...)`
  - this gives a single discrete activity-update primitive
- `update_pc_params(...)`
  - this gives a single post-inference parameter update primitive
- `solve_inference(...)`
  - this gives a continuous-time solver interface
- `update_epc_errors(...)`
  - this changes the inference variable from activities to errors, which is
    relevant to conditioning, but still does not interleave parameter updates
- `make_hpc_step(...)`
  - hybrid PC with amortised initialisation, but still not iPC-style

### Extension points if TF2 wants iFMPC / iPC-style probing

The cleanest extension points on the JPC side would be:

- compose `init_activities_with_ffwd(...)`
- then repeatedly call `update_pc_activities(...)`
- and optionally insert `update_pc_params(...)` inside that outer loop

That is, JPC exposes the right low-level bricks, but not the iPC schedule as a
first-class API.

## 3. Reuse classification for TF2

### Direct reference only

Useful to read, mirror conceptually, or cite, but not worth importing:

- `jpc._train.make_pc_step`
  - good reference for explicit high-level PC step decomposition
  - not appropriate to import because it assumes JAX, Diffrax, Optax, Equinox
- `jpc._test.*`
  - good reference for separating discriminative / generative / hybrid eval
  - not useful as an import into our NumPy codebase
- `jpc._core._analytical.*`
  - very useful as theory and diagnostics reference
  - not directly compatible with our current numerical stack
- `experiments/mupc_paper/*`
  - strong reference for muPC scaling, Hessian monitoring, and condition-number
    studies
- `experiments/limits_paper/*`
  - strong reference for inference-mode comparisons and conditioning studies

### Thin adapter candidate

These are the pieces most worth mirroring or lightly wrapping in TF2 bridge
experiments:

- the API decomposition:
  - feedforward init
  - inference update primitive
  - parameter update primitive
- inference-solver naming / registry ideas from:
  - `experiments/library_paper/utils.py:get_ode_solver`
- diagnostics ideas from:
  - `compute_infer_energies`
  - `compute_activity_norms`
  - `compute_hessian_eigens`
  - `compute_cond_num`
- parameterisation naming and scaling interface:
  - `_get_param_scalings(...)`
  - `param_type in {"sp", "mupc", "ntp"}`

For our repo this likely means "adapter by re-expression", not literal code import.

### Not worth importing into mainline

- anything that hard-depends on JAX / Equinox / Diffrax / Optax
- JPC high-level training wrappers as runtime dependencies
- notebook-only example code
- BPC, ePC, PDM, and hybrid-PC paths as immediate TF2 scope

Reason:

- TF2 bridge wants to stay small and non-invasive
- our main repo is still architecturally distinct
- literal import would create a cross-framework maintenance burden too early

## 4. What is useful for muPC-inspired substrate work

Most useful:

- `jpc._utils.make_mlp(..., param_type="mupc")`
- `jpc._utils.make_skip_model(...)`
- `jpc._core._energies._get_param_scalings(...)`
- `experiments/mupc_paper/train_pcn.py`
- `experiments/mupc_paper/analyse_activity_hessian.py`
- `experiments/mupc_paper/utils.py:compute_hessian_eigens`
- `experiments/mupc_paper/utils.py:compute_cond_num`

Why useful:

- muPC in JPC is not just a paper note; it is threaded through init, energy,
  inference, and update calls via `param_type`.
- the experiments explicitly monitor activity Hessian spectra and condition
  numbers during training and at initialisation.
- this is directly relevant if TF2 wants a muPC-inspired hidden-state substrate
  that is more stable under deeper or stiffer inference.

## 5. What is useful for iPC-inspired scheduling work

Most useful:

- `jpc._core._updates.update_pc_activities`
- `jpc._core._updates.update_pc_params`
- `jpc._core._infer.solve_inference`
- `jpc._core._grads.compute_pc_activity_grad`
- `jpc._core._grads.compute_pc_param_grads`
- `experiments/library_paper/train_mlp.py`
- `experiments/limits_paper/train.py`

Why useful:

- they expose exactly the state update and parameter update primitives an iPC or
  iFMPC bridge scheduler would need
- they show the existing separation between:
  - discrete state optimisation
  - continuous ODE-based state optimisation
  - post-inference parameter updates

What is still missing for direct reuse:

- a first-class interleaved schedule
- a per-step parameter update API
- an "incremental PC" trainer that updates weights inside the inference loop

## 6. What should remain outside our mainline for now

Keep outside mainline TF2 for now:

- any direct dependency on JAX / Equinox / Optax / Diffrax
- direct reuse of JPC model objects
- BPC / ePC / PDM / hybrid-PC branches
- closed-form linear theory code as runtime dependency
- notebook-driven workflows

Reason:

- TF2 is a bridge stage, not a framework migration
- the immediate need is mapping and probing, not replacing the repo substrate
- we should keep the mainline legible and phase-bounded

## 7. JPC vs our current repo: bridge-relevant mismatches

These are the important mismatches to keep explicit:

- JPC is JAX-first and fully functional; our mainline remains NumPy-first.
- JPC activity containers are PyTrees of per-layer arrays, not our current
  flattened hidden-state transport format.
- JPC standard training path is "infer then update params", not interleaved
  iPC-style scheduling.
- JPC's `param_type` abstraction includes `mupc` and `ntp`; our current repo
  does not yet expose an equivalent substrate switch in mainline TF code.
- JPC examples and tests focus on models compatible with its layer-list
  conventions and Equinox modules.

None of these mismatches block a TF2 bridge note, but they argue against
directly importing JPC logic into our mainline right now.

## 8. Smallest non-invasive bridge plan for our repo

### Required now

- `notes/jpc_intake.md`
  - this file

### Optional next step 1: `src/pc/jpc_bridge.py`

Only if TF2 needs a lightweight reference shim. Keep it small and read-only in
spirit:

- define pure-Python metadata structures that mirror JPC concepts:
  - inference mode names
  - parameterisation names
  - diagnostic bundle names
- no hard runtime dependency on JPC in mainline code
- if imported at all, it should degrade cleanly when JAX/JPC is unavailable

Suggested scope:

- string constants / dataclasses for:
  - `param_type`
  - `inference_mode`
  - `diagnostic_fields`
- helper for turning a TF2 probe config into a comparable JPC-style descriptor

### Optional next step 2: `experiments/tf2/tf2_jpc_probe.py`

This is the most attractive next bridge artifact.

Purpose:

- run a very small, isolated, explicitly optional comparison or probe
- compare our TF2 bridge ideas against JPC-inspired scheduler / diagnostics
  conventions
- keep all JAX/JPC contact quarantined to one experiment entrypoint

Suggested initial responsibilities:

- inspect whether a TF2 candidate schedule can reproduce a JPC-like decomposition:
  - feedforward init
  - state-update loop
  - param-update point
- record a minimal diagnostics bundle inspired by JPC:
  - activity / state norm
  - per-step energy
  - gate / conditioning proxy

Not for first pass:

- no import of JPC models into mainline training
- no attempt to align numerical outputs exactly across frameworks
- no rewrite of PredictiveCoding core APIs around JAX

## 9. Bottom line for Phase TF2

For TF2 / iFMPC bridge planning, the most valuable lessons from JPC are:

- explicit decomposition of init / infer / param-update
- clean support for both discrete and continuous inference
- strong inference-conditioning diagnostics
- a concrete `mupc` parameterisation interface

The most important negative result is:

- JPC does **not** already provide the iPC-like interleaved state/weight update
  schedule we would want to bridge toward

So the right TF2 stance is:

- use JPC as a reference and probe target
- do not import it into mainline
- if we build a bridge, keep it isolated and optional
