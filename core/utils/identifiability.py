import jax.numpy as jnp
from typing import Dict
from core.utils.hankel import build_LTI_hankel
from core.models.linearCT import LinearSystem


def verify_identifiability_criteria(A: jnp.ndarray, B: jnp.ndarray, *, sigma_tol: float = 1e-12) -> Dict[str, bool]:
    n = A.shape[0]
    H = build_LTI_hankel(A, B, n)

    # --- Criterion 1 --- Rank deficiency
    rank_defect = jnp.linalg.matrix_rank(H) < n

    # --- Criterion 2 --- Almost zero Singular Value
    sigma_min = jnp.linalg.svd(H, compute_uv=False)[-1]
    almost_zero = sigma_min <= sigma_tol

    # --- Criterion 3 --- Kalman Uncontrollability
    model = LinearSystem(A, B, C=jnp.eye(n))
    kalman_uncontrollable = jnp.logical_not(model.is_controllable(method='kalman'))

    result = {
        "rank_defect": rank_defect,
        "almost_zero": almost_zero,
        "kalman_uncontrollable": kalman_uncontrollable,
    }

    return result