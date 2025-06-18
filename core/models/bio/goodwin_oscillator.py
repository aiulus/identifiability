from typing import Optional, List, Dict, Any
import jax.numpy as jnp
from jax import Array
import diffrax

from .base import DynamicalSystem

class GoodwinOscillator(DynamicalSystem):
    """
    States:
        - M: mRNA concentration
        - P: Protein concentration
        - I: Inhibitory protein concentration
    """
    def __init__(self, 
                 params: Dict[str, float],
                 initial_state: Array,
                 **kwargs: Any):
        self._keys = ["alpha1", "alpha2", "alpha3", "beta1", "beta2", "beta3", "n"]
        for key in self._keys:
            assert key in params, f"Parameter '{key}' is missing."
            assert params[key] >= 0, f"Parameter '{key}' must be non-negative."
            
        jnp_params = jnp.array([params[k] for k in self._keys])
        
        # The output y = P + I has dimension 1
        super().__init__(n=3, m=0, p=1, params=jnp_params, **kwargs)
        self.initial_state = initial_state

    def f(self, state: Array, u: Optional[Array], params: Array, t: float) -> Array:
        M, P, I = state
        alpha1, alpha2, alpha3, beta1, beta2, beta3, n_hill = params
        
        dM = (alpha1 / (1 + I**n_hill)) - beta1 * M
        dP = alpha2 * M - beta2 * P
        dI = alpha3 * P - beta3 * I
        
        return jnp.stack([dM, dP, dI])

    def g(self, state: Array, params: Optional[Array], t: float) -> Array:
        """Observation function: total protein concentration (P + I)."""
        _ , P, I = state
        return jnp.array([P + I])