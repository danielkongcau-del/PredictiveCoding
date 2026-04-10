# FMPC Stage 04 Incremental Bridge Addendum

This file extends the baseline math in [spec_math.md](/e:/CodeSpace/PredictiveCoding/spec_math.md) and the Stage 03 transport addendum in [specs/stage_03_transport_core_v1.md](/e:/CodeSpace/PredictiveCoding/specs/stage_03_transport_core_v1.md).

To build a complete mathematical understanding for Stage 04 work, read in this order:

1. [spec_math.md](/e:/CodeSpace/PredictiveCoding/spec_math.md)
2. [specs/stage_03_transport_core_v1.md](/e:/CodeSpace/PredictiveCoding/specs/stage_03_transport_core_v1.md)
3. this addendum

Scope and precedence:

- this addendum overrides the Stage 03 addendum only inside the explicit Incremental Bridge scope
- outside that scope, the Stage 03 addendum and baseline remain authoritative
- this addendum does not silently redefine the baseline energy, baseline hidden-state gradient, baseline local parameter-update equations, or the slow iterative predict/eval path

## 17. Incremental Bridge addendum

This addendum defines an experimental **training-time scheduling extension** on top of
FMPC Stage 03 Transport Core v1 without changing:

- the baseline predictive-coding energy
- the baseline hidden-state gradient definition
- the baseline local parameter-update equations
- the slow iterative predict/eval path

### 17.1 Scope

FMPC Stage 04 Incremental Bridge remains teacher-free and layered-PC-specific.

- it does not introduce a new substrate class
- it does not depend on JPC runtime
- it does not redefine predict-mode inference
- it does not add a new scaling mechanism in this first bridge pass

### 17.2 Micro-step schedule

Let `H = micro_steps` and define uniform rollout knots:

- `t_k = k / H`
- `Δt = 1 / H`
- `r_k = 1 - t_k`

FMPC Stage 04 Incremental Bridge maintains two training-time hidden-state streams:

- `z_on_k`: learned on-policy hidden state
- `z_lf_k`: detached local-field-only shadow state

The state advances remain:

- `z_on_{k+1} = z_on_k + Δt * u_psi(z_on_k, r_k, t_k; c)`
- `z_lf_{k+1} = z_lf_k + Δt * g_theta(z_lf_k; c)`

where:

- `u_psi` is the same teacher-free average-velocity model from the Transport Core v1 addendum
- `g_theta(z; c) = -∇_z E_theta(z; c)` is unchanged

### 17.3 Frozen-within-micro-step semantics

Within a single micro-step `k`, all supervision targets and state advances must be
computed under one frozen parameter snapshot `(theta_k, psi_k)`.

This includes:

- `u_boot`
- `u_id`
- learned transport outputs
- `z_on_{k+1}`
- `z_lf_{k+1}`

Only after these quantities have been computed may parameter updates be applied.

The required order is:

1. compute supervision targets and learned transport under frozen `(theta_k, psi_k)`
2. advance `z_on` and `z_lf`
3. apply one immediate local `theta` update when enabled
4. apply one `psi` update

### 17.4 Mixed-policy teacher-free supervision

FMPC Stage 04 Incremental Bridge uses one of:

- `supervision_policy = "local_only"`
- `supervision_policy = "mixed"`

For `local_only`, `psi` is supervised only on detached `z_lf_k`.

For `mixed`, `psi` is supervised on the concatenation of detached:

- `z_lf_k`
- `z_on_k`

Targets remain the same Transport Core v1 teacher-free targets:

- `u_boot` from local self-bootstrap
- `u_id = g_t + r_k * D_T u_psi(...)`
- `L = L_boot + lambda_id * L_id`

If FMPC Stage 04 Incremental Bridge uses appended teacher-free current-state features in the psi input, then the
same two identity-tangent semantics from Section 16.5.1 apply:

- `feature_aware_tangents = true`
  - approximate the full augmented-input total derivative by injecting the feature
    directional-derivative block
- `feature_aware_tangents = false`
  - use the explicit truncated identity approximation that freezes the appended
    feature block inside the JVP path

### 17.5 Matched theta-update budget

Incremental Bridge introduces an explicit scheduling control:

- `theta_update_budget in {"matched", "unmatched"}`

If `incremental_weight_updates = true` and the budget is `matched`, the per-micro-step
parameter learning rates are normalized by the number of theta updates that are
actually applied under the active cadence for that batch:

- `theta_micro_lr = base_theta_lr / N_theta`
- `theta_micro_bias_lr = base_theta_bias_lr / N_theta`

where `N_theta` is:

- `1` for `terminal_only`
- `micro_steps` for `every_micro_step`
- the number of due theta-update events inside the micro-step loop for
  `every_2_micro_steps`

If `incremental_weight_updates = true` and the budget is `unmatched`, then:

- `theta_micro_lr = base_theta_lr`
- `theta_micro_bias_lr = base_theta_bias_lr`

If `incremental_weight_updates = false`, no theta updates occur inside the micro-step
loop and one terminal theta update is applied after the final micro-step using the
existing base learning rates.

This addendum changes only the **training-time schedule**. It does not redefine the
baseline local parameter-update rule itself.

### 17.6 Terminal local-field direction intervention

Incremental Bridge may optionally apply a **terminal-step teacher-free direction intervention**
during training:

- `terminal_local_field_direction_intervention in {`
  - `"none"`
  - `"local_field_direction_angle_clip_keep_live_norm"`
  - `"local_field_direction_smooth_unified_cone_projection_keep_live_norm"`
  - `"local_field_direction_hard_replace_keep_live_norm"`
  - `"local_field_direction_angle_clip_keep_live_norm_rowspace_only"`
  - `"local_field_direction_hard_replace_keep_live_norm_rowspace_only"`
  - `"local_field_direction_angle_clip_keep_live_norm_orthogonal_only"`
  - `"local_field_direction_angle_clip_keep_live_norm_split_threshold"`
  - `}`

By default, this intervention is defined only for the **final micro-step** of the
true closed-loop training rollout. Diagnostic configurations may expose
`terminal_local_field_intervention_step_offsets`, a set of late-rollout step
offsets relative to the final micro-step, to apply the same intervention on a
configured subset of late micro-steps. The adopted preset keeps:

- `terminal_local_field_intervention_step_offsets = (-1,)`

This does **not** change:

- the remaining-horizon contract for `u_psi(z_t, r, t; c)`
- the bootstrap target formula
- the identity target formula
- the evaluation-time transport operator

Let the raw learned terminal action be:

- `u_live = (z_{k+1}^{live} - z_k) / Δt`

and let the teacher-free local-field anchor direction be extracted from the current
terminal psi input:

- `d_lf = normalize(g_t)`

where `g_t` is the existing detached teacher-free local-flow block already present in
the Incremental Bridge input features.

If `terminal_local_field_direction_intervention = "none"`, Incremental Bridge uses:

- `u_term = u_live`

If `terminal_local_field_direction_intervention = "local_field_direction_hard_replace_keep_live_norm"`,
Incremental Bridge keeps the learned terminal norm but replaces the direction:

- `u_term = ||u_live|| * d_lf`

If `terminal_local_field_direction_intervention = "local_field_direction_angle_clip_keep_live_norm"`,
Incremental Bridge keeps the learned terminal norm but clips the learned terminal direction into a
cone around `d_lf` with half-angle `terminal_local_field_angle_clip_degrees`:

- `u_term = ||u_live|| * clip_dir(normalize(u_live), d_lf; theta_clip)`

If `terminal_local_field_direction_intervention = "local_field_direction_smooth_unified_cone_projection_keep_live_norm"`,
Incremental Bridge stays in the same full-space local-field cone family, keeps the learned
terminal norm, leaves in-cone actions unchanged, and applies a smooth interior
projection toward `d_lf` only when the learned terminal direction falls outside
the cone:

- `u_term = ||u_live|| * smooth_clip_dir(normalize(u_live), d_lf; theta_clip)`

Incremental Bridge may also use the readout-relevant row-space of the current output layer:

- `P_row = projector(rowspace(W_out))`
- `P_orth = I - P_row`

Define:

- `u_live^row = P_row u_live`
- `u_live^orth = P_orth u_live`
- `d_lf^row = normalize(P_row d_lf)`
- `d_lf^orth = normalize(P_orth d_lf)`

If `terminal_local_field_direction_intervention = "local_field_direction_hard_replace_keep_live_norm_rowspace_only"`,
Incremental Bridge keeps the live orthogonal component unchanged and replaces only the readout-row-space
component direction while preserving its live norm:

- `u_term = ||u_live^row|| * d_lf^row + u_live^orth`

If `terminal_local_field_direction_intervention = "local_field_direction_angle_clip_keep_live_norm_rowspace_only"`,
Incremental Bridge keeps the live orthogonal component unchanged and clips only the readout-row-space
component direction into a cone around `d_lf^row` with half-angle
`terminal_local_field_angle_clip_degrees`:

- `u_term = ||u_live^row|| * clip_dir(normalize(u_live^row), d_lf^row; theta_clip) + u_live^orth`

If `terminal_local_field_direction_intervention = "local_field_direction_angle_clip_keep_live_norm_orthogonal_only"`,
Incremental Bridge keeps the live row-space component unchanged and clips only the orthogonal
component direction into a cone around `d_lf^orth` with half-angle
`terminal_local_field_angle_clip_degrees`:

- `u_term = u_live^row + ||u_live^orth|| * clip_dir(normalize(u_live^orth), d_lf^orth; theta_clip)`

If `terminal_local_field_direction_intervention = "local_field_direction_angle_clip_keep_live_norm_split_threshold"`,
Incremental Bridge keeps both components active but clips them with separate half-angles:

- `theta_row = terminal_local_field_rowspace_angle_clip_degrees`
- `theta_orth = terminal_local_field_orthogonal_angle_clip_degrees`
- `u_term = ||u_live^row|| * clip_dir(normalize(u_live^row), d_lf^row; theta_row)
           + ||u_live^orth|| * clip_dir(normalize(u_live^orth), d_lf^orth; theta_orth)`

If either the live row-space component or the anchor row-space component is degenerate,
Incremental Bridge falls back to leaving the row-space component unchanged for that sample.

If either the live orthogonal component or the anchor orthogonal component is degenerate,
Incremental Bridge falls back to leaving the orthogonal component unchanged for that sample.

The transported terminal state used for the immediate terminal theta update is then:

- `z_{k+1} = z_k + Δt * u_term`

This is a **training-time stabilization option** only. It preserves the Incremental Bridge teacher-free
target construction and keeps the historical corrective preset available as an
unstabilized reference.

Repository-level preset note:

- the named preset `tf2_corrective_transport_default` remains the historical plain
  corrective-family reference
- the named preset `tf2_corrective_transport_terminal_angleclip_default` denotes the
  corrective-family package:
  - `psi_family = "residualized_local_field"`
  - `time_encoding_variant = "poly_rt2"`
  - `terminal_local_field_direction_intervention = "local_field_direction_angle_clip_keep_live_norm"`
  - `terminal_local_field_angle_clip_degrees = 30`

This preset naming is an **implementation / validation contract**, not a new
mathematical family. The normative mathematical change in this section is only the
availability of the optional terminal local-field direction intervention during
training.

### 17.7 Optional transported output-alignment weighting

Incremental Bridge may also expose an optional **output-side readout-alignment aid** that does
not change the transport family or the teacher-free target construction.

Configuration:

- `transported_output_alignment_weight >= 0`
- `transported_output_alignment_schedule in {`
  - `"none"`,
  - `"final_micro_step_only"`,
  - `"every_micro_step"`,
  `}`

Normative meaning:

- build the transported hidden state `z_(k+1)` exactly as in the current Incremental Bridge
  corrective package
- build the target-clamped transported-state cache exactly as in the current Incremental Bridge
  theta-update path
- compute the standard local parameter gradients from that transported state
- if output alignment is active on the current micro-step, scale only the final
  predictive layer parameter gradients by:

```text
(1 + lambda_out)
```

where `lambda_out = transported_output_alignment_weight`

Schedule semantics:

- `"none"`:
  - no extra readout weighting
- `"final_micro_step_only"`:
  - apply the output-side gradient scaling only on the terminal micro-step
- `"every_micro_step"`:
  - apply the output-side gradient scaling on every micro-step where Incremental Bridge performs
    a theta update

This is a **train-time weighting extension** only:

- it does not alter `u_boot`
- it does not alter the identity target
- it does not alter the terminal local-field direction intervention
- it does not introduce a new Incremental Bridge transport family
