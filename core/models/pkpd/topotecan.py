from identifiability.core.models.base import DynamicalSystem
import jax.numpy as jnp
from jax import Array

"""
5-Compartment PK model for Topotecan (Evans et.al. 2004).

@article{evans2004mathematical,
  title={A mathematical model for the in vitro kinetics of the anti-cancer agent topotecan},
  author={Evans, Neil D and Errington, Rachel J and Shelley, Michael and Feeney, Graham P and Chapman, Michael J and Godfrey, Keith R and Smith, Paul J and Chappell, Michael J},
  journal={Mathematical Biosciences},
  volume={189},
  number={2},
  pages={185--217},
  year={2004},
  publisher={Elsevier}
}
"""

class TopotecanModel(DynamicalSystem):
    def __init__(self, params: dict, initial_dose: float):
        """
        Model parameters:
            - k_x: Flow rate constants between compartments (locations) and forms of drug
            - BT: Concentratino of DNA sites available for TPT binding
            - v1: Medium volume / cytoplasm volume
            - v2: Nucleus volume / cytoplasm volume
        """        
        self._keys = [
            "ko_m", "kc_m", "ki", "ke",
            "ko_c", "kc_c", "kb", "kd",
            "BT", "v1", "v2"
        ]
        
        # --- assert all parameters >= 0 ---
        for key in self._keys:
            val = params[key]
            assert val >= 0, f"Parameter '{key}' must be ≥ 0, got {val}"
            
        jnp_params = jnp.array([params[k] for k in self._keys])
        super().__init__(n=5, m=0, p=4, params=jnp_params)
        
        # initial state: all drug in extracellular lactone compartment
        self.initial_state = jnp.array([
            initial_dose,  # Lm: TPT-L in the medium
            0.0,           # Hm: TPT-H in the medium
            0.0,           # Lc: TPT-L in the cytoplasm
            0.0,           # Hc: TPT-H in the cytoplasm
            0.0            # Ln: TPT-L in the nucleus
        ])

    def f(self, state: Array, u: Array, params: Array, t: float) -> Array:
        Lm, Hm, Lc, Hc, Ln = state
        ko_m, kc_m, ki, ke, ko_c, kc_c, kb, kd, BT, v1, v2 = params
        dLm = - (ko_m + ki) * Lm + kc_m * Hm + (ke / v1) * Lc
        dHm = ko_m * Lm - kc_m * Hm
        dLc = (ki * v1) * Lm - (ke + ko_c) * Lc + kc_c * Hc - kb * (BT - Ln) * Lc + (v2 * kd) * Ln
        dHc = ko_c * Lc - kc_c * Hc
        dLn = (kb / v2) * (BT - Ln) * Lc - kd * Ln
        return jnp.stack([dLm, dHm, dLc, dHc, dLn])

    def g(self, state: Array, params: Array, t: float) -> Array:
        Lm, Hm, Lc, Hc, Ln = state
        ko_m, kc_m, ki, ke, ko_c, kc_c, kb, kd, BT, v1, v2 = params
        Hi = Hc /(1 + v2)
        Li = (Lc + v2*Ln) / (1 + v2)
        y = jnp.stack([Lm, Hm, Li, Hi])
        return y