from typing import Optional, Dict, Any
import jax.numpy as jnp
from jax import Array
import diffrax

from .base import DynamicalSystem

class LinearSystem(DynamicalSystem):
    """
    Defines a Linear Time-Invariant (LTI) system of the form:
        dx/dt = Ax + Bu
        y = Cx
    """
    def __init__(self, A: Array, B: Array, C: Array, 
                 solver: Optional[diffrax.AbstractSolver] = None,
                 solver_options: Optional[Dict[str, Any]] = None):
        
        assert A.ndim == 2 and A.shape[0] == A.shape[1], "A must be a square matrix."
        n = A.shape[0]
        
        assert B.ndim == 2 and B.shape[0] == n, f"B must have {n} rows."
        m = B.shape[1]
        
        assert C.ndim == 2 and C.shape[1] == n, f"C must have {n} columns."
        p = C.shape[0]

        super().__init__(n=n, m=m, p=p, solver=solver, solver_options=solver_options)
        
        self.A = A
        self.B = B
        self.C = C

    def f(self, state: Array, u: Optional[Array], params: Optional[Array], t: float) -> Array:
        """Implements the linear dynamics: Ax + Bu."""
        return self.A @ state + self.B @ u

    def g(self, state: Array, params: Optional[Array], t: float) -> Array:
        """Implements the linear observation: Cx."""
        return self.C @ state

    def kalman_controllability_matrix(self) -> Array:
        """Constructs the Kalman controllability matrix."""
        n = self.n
        C_matrix_list = [self.B]
        for i in range(1, n):
            C_matrix_list.append(self.A @ C_matrix_list[-1])
        return jnp.hstack(C_matrix_list)

    def is_controllable(self, method: str = 'kalman') -> bool:
        """
        Checks system controllability using the specified method.

        Args:
            method (str): 'kalman' for Kalman rank test or 'pbh' for PBH test.
                > 'kalman': $\operatorname{rank}\eft(\begin{bmatrix} B & AB & A^2B & \cdots & A^{n-1}B \end{bmatrix}\right)\overset{?}{=}n$
                > 'pbh' (Popov-Belevitch-Hautus): For every eigenvalue $\lambda$ of $A$, $\text{rank}\left(\begin{bmatrix} A - \lambda I & B \end{bmatrix}\right) = n$
        """
        
        if method == 'kalman':
            C_matrix = self.kalman_controllability_matrix()
            return jnp.linalg.matrix_rank(C_matrix) == self.n
        elif method == 'pbh':
            eigenvalues = jnp.linalg.eigvals(self.A)
            for eig in eigenvalues:
                M = jnp.hstack([self.A - eig * jnp.eye(self.n), self.B])
                if jnp.linalg.matrix_rank(M) < self.n:
                    return False
            return True
        else:
            raise ValueError("Method must be 'kalman' or 'pbh'")

    def kalman_observability_matrix(self) -> Array:
        """Constructs the Kalman observability matrix."""
        n = self.n
        O_matrix_list = [self.C]
        for i in range(1, n):
            O_matrix_list.append(O_matrix_list[-1] @ self.A)
        return jnp.vstack(O_matrix_list)

    def is_observable(self, method: str = 'kalman') -> bool:
        """
        Checks system observability using the specified method.

        Args:
            method (str): 'kalman' for Kalman rank test or 'pbh' for PBH test.
            > 'kalman': Checks whether $\operatorname(rank)\left(\begin{bmatrix} C \ CA \ CA^2 \ \vdots \ CA^{n-1} \end{bmatrix}\right)=n
            > 'pbh': Checks whether $\operatorname{rank}\left(\begin{bmatrix} A - \lambda I \ C \end{bmatrix}\right) = n$ for every eigenvalue $\lambda$ of $A$
        """
        if method == 'kalman':
            O_matrix = self.kalman_observability_matrix()
            return jnp.linalg.matrix_rank(O_matrix) == self.n
        elif method == 'pbh':
            eigenvalues = jnp.linalg.eigvals(self.A)
            for eig in eigenvalues:
                M = jnp.vstack([self.A - eig * jnp.eye(self.n), self.C])
                if jnp.linalg.matrix_rank(M) < self.n:
                    return False
            return True
        else:
            raise ValueError("Method must be 'kalman' or 'pbh'")

