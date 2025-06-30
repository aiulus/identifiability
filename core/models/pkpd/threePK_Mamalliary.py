from core.models.linearCT import LinearSystem
import jax.numpy as jnp
from jax import Array
import diffrax
from typing import Dict, Any, Optional

class threePKModel(LinearSystem):
    def __init__(self,
                 params: Dict[str, float],
                 solver: Optional[diffrax.AbstractSolver] = None,
                 solver_options: Optional[Dict[str, Any]] = None
                 ):
        k10, k12, k13 = params['k10'], params['k12'], params['k13']
        k21, k31 = params['k21'], params['k31']
        V1 = params['V1']

        A = jnp.array([
            [-(k10 + k12 + k13), k21, k31],
            [k12,                -k21, 0.0],
            [k13,                0.0, -k31]
        ])

        B = jnp.array([
            [1.0],
            [0.0],
            [0.0]
        ])

        C = jnp.array([
            [1.0 / V1, 0.0, 0.0]
        ])
        
        
        super().__init__(A=A, B=B, C=C, solver=solver, solver_options=solver_options)