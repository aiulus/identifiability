from typing import Optional
from jax import Array
import jax.numpy as jnp
from .models.base import DynamicalSystem

class IdentificationProblem:
    """
    A container class that defines a complete system identification problem.
    """
    def __init__(self,
                 sys: DynamicalSystem,
                 time_steps: Array, 
                 u_meas: Optional[Array] = None,
                 y_meas: Optional[Array] = None
                 ):
        self.sys = sys
        self.time_steps = time_steps        
        if u_meas is None:
            self.u_meas = jnp.zeros((time_steps.shape[0], sys.m))
        else:
            assert time_steps.shape[0] == u_meas.shape[0], \
                f"size(time_steps, 1) = {time_steps.shape[0]} != {u_meas.shape[0]} = size(u_meas, 1)"
            assert sys.m == u_meas.shape[1], \
                f"Measurement vector inherent dim. ({u_meas.shape[1]}) doesn't match m ({sys.m})!"
            self.u_meas = u_meas
        if y_meas is not None:
            assert time_steps.shape[0] == y_meas.shape[0], \
                "Time seps and measurement data must have the same length!"
            assert sys.p == y_meas.shape[1], \
                f"Measurement data must have dimension {sys.p}"
        self.y_meas = y_meas