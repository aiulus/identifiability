from typing import Optional, List, Callable
import jax
import jax.numpy as jnp
from jax import Array

from .base import DynamicalSystem
from ..utils.algebra import lie_derivative

class ControlAffineSystem(DynamicalSystem):
    """
    Represents a nonlinear system in control-affine form:
        dx/dt = f(x) + Σ g_i(x)u_i
        y = h(x)
    """
    def __init__(self, 
                 n: int, 
                 m: int,
                 f_drift: Callable[[Array], Array],
                 g_control: List[Callable[[Array], Array]],
                 h_output: Callable[[Array], Array],
                 **kwargs):
        """
        Initializes the nonlinear system.

        Args:
            n: State dimension.
            m: Input dimension.
            f_drift: The drift vector field f(x).
            g_control: A list of m control vector fields [g_1(x), ..., g_m(x)].
            h_output: The output function h(x).
            **kwargs: Other arguments for the DynamicalSystem base class (solver, etc.).
        """
        super().__init__(n=n, m=m, p=h_output(jnp.zeros(n)).shape[0], **kwargs)
        
        assert len(g_control) == m, "Number of control fields must match input dimension m."
        
        self.f_drift = f_drift
        self.g_control = g_control
        self.h_output = h_output

    def f(self, state: Array, u: Array, params: Optional[Array], t: float) -> Array:
        """Implements the full dynamics dx/dt = f(x) + G(x)u."""
        control_effect = sum(g(state) * u[i] for i, g in enumerate(self.g_control))
        return self.f_drift(state) + control_effect

    def g(self, state: Array, params: Optional[Array], t: float) -> Array:
        """Implements the output mapping y = h(x)."""
        return self.h_output(state)

    def observability_codistribution_matrix(self, x: Array) -> Array:
        """
        Constructs the observability codistribution matrix O_NL(x).
        The rows of this matrix are the gradients of successive Lie derivatives
        of the output functions.
        """
        # Split the output function h(x) into its scalar components h_i(x)
        h_components = [lambda x, i=i: self.h_output(x)[i] for i in range(self.p)]
        
        codistribution = []
        for h_i in h_components:
            L_f_h = h_i
            # Add gradient of h_i itself (k=0)
            codistribution.append(jax.grad(L_f_h)(x))
            
            # Add gradients of successive Lie derivatives
            for _ in range(self.n - 1):
                L_f_h = lie_derivative(self.f_drift, L_f_h)
                codistribution.append(jax.grad(L_f_h)(x))
                
        return jnp.vstack(codistribution)

    def is_locally_observable(self, at_point: Array) -> bool:
        """
        Checks for local observability at a specific point using the
        Observability Rank Condition.
        
        Ref: Hermann & Krener (1977), "Nonlinear controllability and observability"
        """
        O_matrix = self.observability_codistribution_matrix(at_point)
        return jnp.linalg.matrix_rank(O_matrix) == self.n

