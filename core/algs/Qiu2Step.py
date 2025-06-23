import jax.numpy as jnp
from jax import Array
import numpy as np
from typing import Dict, Any


def two_stage_simple(y_data: Array, dt: float) -> Dict[str, Any]:
    """
    based on https://github.com/qiuxing/ode.ident
    """
    p, T = y_data.shape

    # S is the identity matrix in the simple two-stage method, representing
    # the assumption of uncorrelated measurement errors
    S = jnp.eye(T)

    # NOTE: The construction of the finite difference matrix L differs from the
    # `twostage1` function in the original `ode.ident` R package. This
    # implementation uses a standard central difference scheme for interior points
    # and forward/backward differences for the boundaries. This approach is
    # generally more numerically accurate but will produce different results
    # than the original R code.
    L_diag = jnp.diag(-jnp.ones(T - 1), k=-1) + jnp.diag(jnp.ones(T - 1), k=1)
    L = L_diag / (2 * dt)

    L = L.at[0, 0].set(-1 / dt)
    L = L.at[0, 1].set(1 / dt)
    L = L.at[-1, -2].set(-1 / dt)
    L = L.at[-1, -1].set(1 / dt)

    Y = y_data.T

    Y_YT_inv = np.linalg.pinv(Y @ Y.T)

    A_hat = Y @ L.T @ Y.T @ Y_YT_inv

    return {'A_hat': A_hat, 'S': S, 'L': L}
