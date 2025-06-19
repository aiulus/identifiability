from typing import Dict, Any, Optional
import jax.numpy as jnp
from jax import Array
import diffrax

from ..linearCT import LinearSystem

class SimpleGlucoseInsulinModel(LinearSystem):
    """
    Implements a linearized 3-state model of the glucose-insulin regulatory system.

    This class constructs the A, B, and C state-space matrices from key
    physiological parameters.

    States (n=3):
        x1: G - Deviation of blood glucose concentration from baseline
        x2: I - Deviation of blood insulin concentration from baseline
        x3: X - Insulin action in a remote compartment
    
    Inputs (m=2):
        u1: External glucose infusion rate
        u2: External insulin infusion rate

    Outputs (p=2, by default):
        y1: Glucose deviation (G)
        y2: Insulin deviation (I)
    """
    def __init__(self, 
                 params: Dict[str, float],
                 C_matrix: Optional[Array] = None,
                 solver: Optional[diffrax.AbstractSolver] = None,
                 solver_options: Optional[Dict[str, Any]] = None):
        """
        Initializes the model from a dictionary of physiological parameters.
        """
        p1, p2, p3 = params['p1'], params['p2'], params['p3']
        n_param, k = params['n'], params['k']

        A = jnp.array([
            [-p1, -p2, -p3],
            [0.0, -n_param, 0.0],
            [0.0, k, -k]
        ])

        B = jnp.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0]
        ])

        if C_matrix is None:
            C = jnp.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0]
            ])
        else:
            C = C_matrix

        super().__init__(A=A, B=B, C=C, solver=solver, solver_options=solver_options)
