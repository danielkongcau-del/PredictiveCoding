from __future__ import annotations

from typing import Literal

import numpy as np

from ..interval_meanflow.fmpc_meanflow_jvp import MeanFlowMLPJVPResult, forward_mlp_with_jvp
from .fmpc_tf1_flow import FMPCTF1StateFeatures, FMPCTF1StateFeatureTangents
from ..mlp_baseline import MLPNetwork

TF1IdentityTangentMode = Literal[
    "base_total_derivative",
    "feature_aware_total_derivative_approx",
    "feature_frozen_truncated_identity_approx",
]


def _as_batch_first(name: str, array: np.ndarray) -> np.ndarray:
    array_float = np.asarray(array, dtype=np.float64)
    if array_float.ndim != 2:
        raise ValueError(f"{name} must be shaped (batch, features).")
    return array_float


def resolve_tf1_identity_tangent_mode(
    *,
    use_teacher_free_features: bool,
    feature_aware_tangents: bool,
) -> TF1IdentityTangentMode:
    """Return the repository's explicit TF1/TF2 identity-tangent semantics label."""

    if not use_teacher_free_features:
        return "base_total_derivative"
    if feature_aware_tangents:
        return "feature_aware_total_derivative_approx"
    return "feature_frozen_truncated_identity_approx"


def build_tf1_input(
    z_t: np.ndarray,
    target_onehot: np.ndarray,
    *,
    t: float,
    r: float,
    use_teacher_free_features: bool,
    features: FMPCTF1StateFeatures | None = None,
) -> np.ndarray:
    """Build the TF1 batch-first input block for `(z_t, r, t)`."""

    z_array = _as_batch_first("z_t", z_t)
    target_array = _as_batch_first("target_onehot", target_onehot)
    if z_array.shape[0] != target_array.shape[0]:
        raise ValueError("z_t and target_onehot must share the same batch size.")
    batch_size = int(z_array.shape[0])
    t_block = np.full((batch_size, 1), float(t), dtype=np.float64)
    r_block = np.full((batch_size, 1), float(r), dtype=np.float64)
    if not use_teacher_free_features:
        return np.concatenate([z_array, target_array, t_block, r_block], axis=1).astype(
            np.float64,
            copy=False,
        )
    if features is None:
        raise ValueError("features must be provided when use_teacher_free_features=True.")
    return np.concatenate(
        [z_array, target_array, t_block, r_block, features.g_t, features.e_out_t, features.F_t],
        axis=1,
    ).astype(np.float64, copy=False)


def build_tf1_input_tangent(
    g_t: np.ndarray,
    *,
    target_dim: int,
    use_teacher_free_features: bool,
    feature_aware_tangents: bool = False,
    feature_tangents: FMPCTF1StateFeatureTangents | None = None,
    d_t: float = 1.0,
    d_r: float = -1.0,
) -> np.ndarray:
    """Build the TF1 fixed-terminal-time input tangent.

    When `use_teacher_free_features=True` and `feature_aware_tangents=False`, the
    appended feature block is treated as frozen side information and receives a zero
    tangent. The resulting JVP therefore corresponds to the repository's explicit
    truncated identity approximation rather than the full augmented-input total
    derivative.
    """

    g_array = _as_batch_first("g_t", g_t)
    batch_size = int(g_array.shape[0])
    target_tangent = np.zeros((batch_size, int(target_dim)), dtype=np.float64)
    t_tangent = np.full((batch_size, 1), float(d_t), dtype=np.float64)
    r_tangent = np.full((batch_size, 1), float(d_r), dtype=np.float64)
    if not use_teacher_free_features:
        return np.concatenate([g_array, target_tangent, t_tangent, r_tangent], axis=1).astype(
            np.float64,
            copy=False,
        )

    if feature_aware_tangents:
        if feature_tangents is None:
            raise ValueError(
                "feature_tangents must be provided when feature_aware_tangents=True."
            )
        feature_tangent_block = np.concatenate(
            [
                feature_tangents.Dg_g_t,
                feature_tangents.Dg_e_out_t,
                feature_tangents.Dg_F_t,
            ],
            axis=1,
        ).astype(np.float64, copy=False)
    else:
        if feature_tangents is None:
            feature_dim = int(g_array.shape[1]) + int(target_dim) + 1
        else:
            feature_dim = (
                int(feature_tangents.Dg_g_t.shape[1])
                + int(feature_tangents.Dg_e_out_t.shape[1])
                + int(feature_tangents.Dg_F_t.shape[1])
            )
        feature_tangent_block = np.zeros((batch_size, feature_dim), dtype=np.float64)
    return np.concatenate(
        [g_array, target_tangent, t_tangent, r_tangent, feature_tangent_block],
        axis=1,
    ).astype(np.float64, copy=False)


def forward_tf1_mlp_with_jvp(
    network: MLPNetwork,
    inputs: np.ndarray,
    input_tangent: np.ndarray,
) -> MeanFlowMLPJVPResult:
    """Evaluate the TF1 MLP and its explicit forward-mode JVP."""

    return forward_mlp_with_jvp(network, inputs, input_tangent)
