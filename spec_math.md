# Mathematical Specification

This file defines the **baseline mathematical formulation** for the repository.

It is intentionally narrow. The goal is to make implementation, testing, and later extension unambiguous.

---

## 1. Scope of this baseline

This baseline implements a **supervised, fully-connected, batch-first predictive coding network** with:

- layerwise deterministic activations
- squared-error energy
- iterative state inference for free layers
- local parameter updates
- batch-first arrays

This is **not** claimed to be the only or most general predictive coding formulation. It is the starting point for a clean implementation.

---

## 2. Notation

Let the network have layers indexed by `l = 0, 1, ..., L`.

- `X^l in R^(B x n_l)` is the state at layer `l`
- `B` is batch size
- `n_l` is the feature dimension of layer `l`
- `W^l in R^(n_l x n_(l-1))` maps layer `l-1 -> l`
- `b^l in R^(n_l,)` is the bias for layer `l`
- `A^l in R^(B x n_l)` is the pre-activation for layer `l`
- `MU^l in R^(B x n_l)` is the prediction of `X^l`
- `E^l in R^(B x n_l)` is the prediction error at layer `l`
- `phi_l(.)` is the activation for layer `l`
- `phi_l'` is its elementwise derivative
- `sigma_l^2 > 0` is the error variance scale for layer `l`

### Batch-first convention

All public-facing arrays use shape:

```text
(batch, features)
```

We use row-major sample organization.

---

## 3. Forward prediction equations

For each layer `l = 1, ..., L`:

```text
A^l = X^(l-1) @ (W^l)^T + b^l
MU^l = phi_l(A^l)
E^l = X^l - MU^l
```

Interpretation:

- the state at layer `l` is compared against the prediction made from layer `l-1`
- `E^l` is the local mismatch between the current state and its prediction

---

## 4. Energy / objective

For a batch, define the baseline energy as:

```text
F(X, W, b) = sum_{l=1}^L [ 1 / (2 B sigma_l^2) ] * || E^l ||_F^2
```

where `|| . ||_F` is the Frobenius norm.

Equivalent expanded form:

```text
F = sum_{l=1}^L [ 1 / (2 B sigma_l^2) ] * sum_{i=1}^B sum_{j=1}^{n_l} (E^l_{ij})^2
```

### Default simplification

In Phase 0, it is acceptable to set:

```text
sigma_l^2 = 1 for all l
```

but the implementation should keep the scaling explicit in the code structure.

---

## 5. Clamping rules

### Training mode

During supervised training:

- `X^0` is clamped to the input batch
- `X^L` is clamped to the target batch
- hidden layers `X^1 ... X^(L-1)` are free variables inferred by iterative dynamics

### Prediction / evaluation mode

During prediction:

- `X^0` is clamped to the input batch
- `X^1 ... X^L` are free variables, unless a specific experiment chooses to clamp additional layers
- the final prediction is read from `X^L` after inference

In early classification experiments, predicted class is:

```text
argmax(X^L, axis=1)
```

---

## 6. State initialization

State initialization is not part of the energy itself, but it matters numerically.

The baseline implementation should support at least these initialization modes for free layers:

1. `zeros`
   - initialize all free states to zero

2. `forward`
   - recursively initialize
   - `X^1 = phi_1(X^0 @ (W^1)^T + b^1)`
   - `X^2 = phi_2(X^1 @ (W^2)^T + b^2)`
   - etc.

Default recommendation for early experiments:

- use `forward` initialization when possible

---

## 7. Inference dynamics for free states

For a free hidden layer `l` with `1 <= l <= L-1`, update by gradient descent on `F` with respect to `X^l`.

The gradient is:

```text
dF/dX^l = (1 / (B sigma_l^2)) E^l
          - (1 / (B sigma_(l+1)^2)) (E^(l+1) ⊙ phi_(l+1)'(A^(l+1))) @ W^(l+1)
```

where `⊙` is elementwise multiplication.

The baseline explicit Euler update is:

```text
X^l <- X^l - eta_x * dF/dX^l
```

or equivalently:

```text
X^l <- X^l + eta_x * [
    -(1 / (B sigma_l^2)) E^l
    + (1 / (B sigma_(l+1)^2)) (E^(l+1) ⊙ phi_(l+1)'(A^(l+1))) @ W^(l+1)
]
```

### Output layer in prediction mode

If `X^L` is free during prediction, then:

```text
dF/dX^L = (1 / (B sigma_L^2)) E^L
```

and the update is:

```text
X^L <- X^L - eta_x * dF/dX^L
```

### Clamped layers

For any clamped layer, **do not update the state**.

### Inference schedule

A baseline inference pass runs for `T` steps:

```text
for t in 1..T:
    recompute A^l, MU^l, E^l for all l
    update each free layer according to the state rule
```

Synchronous versus in-place layer updates should be documented. The baseline recommendation is:

- recompute all errors from current states
- compute all state deltas
- apply the deltas layerwise after the full sweep

This avoids accidental order dependence.

---

## 8. Parameter updates

After inference has finished for the current batch, update parameters by descending `F` with respect to `W^l` and `b^l`.

For each layer `l = 1, ..., L`:

```text
dF/dW^l = -(1 / (B sigma_l^2)) (E^l ⊙ phi_l'(A^l))^T @ X^(l-1)
```

```text
dF/db^l = -(1 / (B sigma_l^2)) sum_rows(E^l ⊙ phi_l'(A^l))
```

where `sum_rows(.)` sums across the batch dimension and returns shape `(n_l,)`.

The baseline SGD-style updates are:

```text
W^l <- W^l - eta_w * dF/dW^l
b^l <- b^l - eta_b * dF/db^l
```

Equivalent additive form:

```text
W^l <- W^l + eta_w * (1 / (B sigma_l^2)) (E^l ⊙ phi_l'(A^l))^T @ X^(l-1)
```

```text
b^l <- b^l + eta_b * (1 / (B sigma_l^2)) sum_rows(E^l ⊙ phi_l'(A^l))
```

### Default simplification

In Phase 0–1, it is acceptable to use:

```text
eta_b = eta_w
```

but the code should not assume this permanently.

---

## 9. Recommended activation choices

Baseline defaults:

- hidden layers: `tanh` or `relu`
- output layer: `identity`

The activation derivative must be implemented explicitly.

### Important note

If `relu` is used, inference dynamics can become more brittle near zero. This is not forbidden, but experiments should document the choice.

---

## 10. Training algorithm (baseline)

For each batch `(x, y)`:

1. Clamp `X^0 = x`
2. Clamp `X^L = y`
3. Initialize free hidden states
4. Run `T_train` inference steps over free hidden layers
5. Recompute `A^l`, `MU^l`, `E^l`
6. Update all `W^l`, `b^l`
7. Log:
   - final energy
   - optional per-step energy
   - parameter norms

Pseudo-code:

```text
for each batch (x, y):
    X^0 = x
    X^L = y
    initialize X^1 ... X^(L-1)

    for t in 1..T_train:
        compute A, MU, E
        compute delta_X for free layers
        apply delta_X
        optionally record F_t

    compute A, MU, E one final time
    update W, b
```

---

## 11. Prediction algorithm (baseline)

For each input batch `x`:

1. Clamp `X^0 = x`
2. Initialize `X^1 ... X^L`
3. Run `T_eval` inference steps over free layers
4. Return `X^L`

Pseudo-code:

```text
X^0 = x
initialize X^1 ... X^L
for t in 1..T_eval:
    compute A, MU, E
    compute delta_X for free layers
    apply delta_X
return X^L
```

---

## 12. Numerical safeguards

The baseline implementation may include the following safeguards, but each must be documented in code:

- small weight initialization
- optional state clipping
- optional delta clipping
- optional damping factor on state updates
- NaN / Inf checks in tests or debug mode

Do not add undocumented stabilization tricks.

---

## 13. Phase 0 simplifications

The baseline implementation is allowed to make these simplifying choices:

- all `sigma_l^2 = 1`
- fixed inference step count `T`
- fixed learning rates
- no momentum / Adam
- no trainable recognition network
- no explicit probabilistic output likelihood beyond squared error

---

## 14. What must remain invariant unless the spec changes

The following are repository-level invariants for the baseline:

1. Arrays are batch-first
2. The baseline energy is squared-error over local prediction errors
3. Hidden states are inferred iteratively
4. Parameter updates are local functions of neighboring activities/errors under this spec
5. Clamped states are never updated during inference

---

## 15. Planned extension points

Likely future extensions include:

- separate recognition / initialization network
- alternative output likelihoods
- convolutional layers
- temporal predictive coding
- deeper-network stabilization variants
- alternative inference integrators

Those are not part of the baseline unless they are explicitly added in later versions of the spec.
