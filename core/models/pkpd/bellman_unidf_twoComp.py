from core.models.linearCT import LinearSystem
import jax.numpy as jnp
from jax import Array
import diffrax
from typing import Dict, Any, Optional


class BellmanAstrom(LinearSystem):
    def __init__(self,
                 params: Dict[str, float],
                 solver: Optional[diffrax.AbstractSolver] = None,
                 solver_options: Optional[Dict[str, Any]] = None
                 ):
        k1, k2, k3, k4 = params['k1'], params['k2'], params['k3'], params['k4']
        u = params['u']

        A = jnp.array([
            [-(k1 + k2), k3],
            [-k2, -(k3 + k4)]
        ])

        C = jnp.array([
            [1.0, 0.0]
        ])

        super().__init__(A=A, C=C, x0=u, solver=solver, solver_options=solver_options)
