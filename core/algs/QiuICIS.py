import jax
import jax.numpy as jnp
from jax import Array
import diffrax
from typing import Tuple, Dict, Callable
from scipy.optimize import minimize
from scipy.stats import chi2
from ..models.linearCT import LinearSystem
from ..idp import IdentificationProblem

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

TOL = 1e-12 # zero-tolerance
n_digits = 4 # accuracy for equality checks

class QiuICIS:
    def analyze(
        self, 
        model: LinearSystem, 
        x0: Array
    ) -> Dict:
        A = model.A
        n = model.n
        
        # Step 1: Compute eigenvalues, eigenvectors
        try:
            eigenvalues, eigenvectors = jnp.linalg.eig(A)
        except jnp.linalg.LinAlgError:
            return {
                "identifiable": False,
                "status": "Failed",
                "message": "Eigenvalue computation did not converge.",
                "score": 0.0
            }
        
        # Check for singularity
        cond_num = jnp.linalg.cond(eigenvectors)
        if cond_num > 1.0 / jnp.finfo(eigenvectors.dtype).eps:
            return {
                "identifiable": False,
                "status": "Failed",
                "message": "System matrix A may not be diagonalizable (eigenvector matrix is singular).",
                "score": 0.0
            }
            
        # Step 2.1: Check for repeated eigenvalues
        rounded_eigenvalues = jnp.around(eigenvalues, decimals=n_digits)
        unique_eigenvalues, counts = jnp.unique(rounded_eigenvalues, return_counts=True)
        if jnp.any(counts > 1):
            return {
                "identifiable": False,
                "status": "Success",
                "message": "System is not identifiable due to repeated eigenvalues.",
                "score": 0.0,
                "repeated_eigenvalues": True
            }

        # Step 2.2: Project x0 to the basis of eigenvectors
        try:
            coords = jnp.linalg.inv(eigenvectors) @ x0
        except jnp.linalg.LinAlgError:
            return {
                "identifiable": False,
                "status": "Failed",
                "message": "Could not invert the eigenvector matrix.",
                "score": 0.0
            }
            
        # Step 3: Compute the Initial Condition-Based Identifiability Score (ICIS)
        w0k_norms = []
        i = 0
        while i < n:
            if jnp.abs(jnp.imag(eigenvalues[i])) < TOL:
                w0k_norms.append(jnp.abs(coords[i]))
                i += 1
            else: # Complex values
                if i + 1 < n and jnp.isclose(eigenvalues[i], jnp.conj(eigenvalues[i+1])):
                    w0k_norms.append(jnp.sqrt(jnp.abs(coords[i]) ** 2 + jnp.abs(coords[i + 1]) ** 2))
                    i += 2
                else:
                    w0k_norms.append(jnp.abs(coords[i]))
                    i += 1

        if not w0k_norms:
            icis_score = 0.0
        else:
            icis_score = jnp.min(jnp.array(w0k_norms))

        is_identifiable_icis = icis_score > TOL
        identifiable = is_identifiable_icis

        message = ""
        if identifiable:
            message = f"System is identifiable from x0={x0}"
        elif not is_identifiable_icis:
            message = f"System is not identifiable from x0={x0} using the ICIS score."
        
        return {
            "identifiable": bool(identifiable),
            "status": "Success",
            "message": message,
            "score": float(icis_score),
            "repeated_eigenvalues": False
        }
