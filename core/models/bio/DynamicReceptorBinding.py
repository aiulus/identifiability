from identifiability.core.models.base import DynamicalSystem
import jax.numpy as jnp
from jax import Array

"""
Dynamic Receptor Binding Model with an Effect Component and Linear Transduction (Janzen et.al. 2004).

@article{janzen2016parameter,
  title={Parameter identifiability of fundamental pharmacodynamic models},
  author={Janz{\'e}n, David LI and Bergenholm, Linn{\'e}a and Jirstrand, Mats and Parkinson, Joanna and Yates, James and Evans, Neil D and Chappell, Michael J},
  journal={Frontiers in Physiology},
  volume={7},
  pages={590},
  year={2016},
  publisher={Frontiers Media SA}
}

"""

class DynamicReceptorBinding(DynamicalSystem):
    def __init__(self, params: dict, initial_dose: float):
        """
        System state:
            - Ce: Concentration of the hypothetical effect component
            - RC: Receptor Complex
        Control input:
            - Cp: Blood plasma concentration
        Output:
            - E: Observed effect
        Constants:
            - R_tot: Percentage of total number of receptors
            
        Model parameters:
            - ke_0: 
            - ke: 
            - k_on: 
            - k_off: 
        """        
        self._keys = [
            "ke_0", "ke", "k_on", "k_off", "R_tot"
        ]
            
        jnp_params = jnp.array([params[k] for k in self._keys])
        super().__init__(n=2, m=1, p=1, params=jnp_params)
        
        self.initial_state = jnp.array([
            0.0,           
            0.0            
        ])

    def f(self, state: Array, u: Array, params: Array, t: float) -> Array:
        Ce, RC = state
        ke_0, ke, k_on, k_off, R_tot = params
        Cp = u
        
        dCe = ke_0*(Cp - Ce)
        dRC = k_on*(R_tot - RC)*Ce - k_off*RC
        
        return jnp.stack([dCe, dRC])

    def g(self, state: Array, params: Array, t: float) -> Array:
        Ce, RC = state
        ke_0, ke, k_on, k_off, R_tot = params
        y = ke*RC
        return y