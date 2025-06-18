from typing import Optional, List, Dict, Any
import jax.numpy as jnp
from jax import Array
import diffrax

from identifiability.core.models.base import DynamicalSystem

class HIVModel(DynamicalSystem):
    """
    Implements a classic model of HIV dynamics, describing the interaction
    between healthy T-cells, infected T-cells, and free virus particles.

    States (n=3):
      - T: Healthy CD4+ T-cell concentration
      - T_inf: Infected T-cell concentration
      - V: Free virus particle concentration
    """
    def __init__(self, 
                 params: Dict[str, float], 
                 initial_state: Array,
                 **kwargs: Any):
        
        self._keys = ["s", "d", "beta", "delta", "p", "c"]
        for key in self._keys:
            assert key in params, f"Parameter '{key}' is missing!"
            assert params[key] >= 0, f"Parameter '{key}' must be non-negative!"
            
        jnp_params = jnp.array([params[k] for k in self._keys])
        
        super().__init__(n=3, m=0, p=1, params=jnp_params, **kwargs)
        self.initial_state = initial_state

    def f(self, state: Array, u: Optional[Array], params: Array, t: float) -> Array:
        """Implements the HIV dynamics ODEs"""
        T, T_inf, V = state
        s, d, beta, delta, p, c = params
        
        dT = s - d * T - beta * T * V
        dT_inf = beta * T * V - delta * T_inf
        dV = p * T_inf - c * V
        
        return jnp.stack([dT, dT_inf, dV])

    def g(self, state: Array, params: Optional[Array], t: float) -> Array:
        """Observation function: viral load (V)"""
        _ , _, V = state
        return jnp.array([V])

