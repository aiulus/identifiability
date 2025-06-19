import jax.numpy as jnp
from jax import Array
from typing import Tuple, Dict
from ..models.linearDT import LinearSystem

"""
Performs identifiability analysis on a linear system from a single trajectory, based on 
the ICIS score from Qiu et.al. (2021).
 
@article{qiu2022identifiability,
  title={Identifiability analysis of linear ordinary differential equation systems with a single trajectory},
  author={Qiu, Xing and Xu, Tao and Soltanalizadeh, Babak and Wu, Hulin},
  journal={Applied Mathematics and Computation},
  volume={430},
  pages={127260},
  year={2022},
  publisher={Elsevier}
}
"""

class QiuICIS:
    def analyze(
        self, 
        model: LinearSystem, 
        x0: Array
    ) -> Dict:
        A = model.A
        n = model.n
        
        # Step 1: Compute eigenvalues, eigenvectors
        eigenvalues, eigenvectors = jnp.linalg.eig(A)
        
        # Check for distinct eigenvalues
        if len(jnp.unique(jnp.round(eigenvalues, decimals=8))) < n:
            return {
                "identifiable": False,
                "status": "Failed",
                "message": "System matrix A has non-distinct eigenvalues.",
                "score": 0.0
            }
            
        # Step 2: Project x0 to the basis of eigenvectors
        try:
            inv_eigenvectors = jnp.linalg.inv(eigenvectors)
            coords = inv_eigenvectors @ x0
        except jnp.linalg.LinAlgError:
            return{
                "identifiable": False,
                "status": "Failed",
                "message": "Eigenvector matrix is singular. x0 projection failed.",
                "score": 0.0
            }
            
        # Step 3: Compute the Initial Condition-Based Identifiability Score (ICIS)
        icis_score = jnp.prod(jnp.abs(coords)**2)
        
        is_identifiable = icis_score > 1e-9 # Floating point tolerance constant
        
        return {
            "identifiable": bool(is_identifiable),
            "status": "Success",
            "message": "System is identifiable from this initial condition." if is_identifiable 
                       else "System is not identifiable from this initial condition.",
            "score": float(icis_score)
        }