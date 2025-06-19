from typing import Optional, Dict, Any, Callable
import jax.numpy as jnp
from jax import Array

from .base import DynamicalSystem

class LinearParametricCT(DynamicalSystem):
    """
    LTI system with (A(theta), B(theta), C(theta)):
        dx/dt = A(theta)x + B(theta)u
        y = C(theta)x
    """
    def __init__(self, 
                 n: int, m: int, p: int,
                 params: Array,
                 A_func: Callable[[Array], Array],
                 B_func: Callable[[Array], Array],
                 C_func: Callable[[Array], Array],
                 **kwargs
    ):
        super().__init__(n=n, m=m, p=p, params=params, **kwargs)
        self.A_func = A_func
        self.B_func = B_func
        self.C_func = C_func
        
    def f(self, state: Array, u: Array, params: Array, t: float) -> Array:
        A = self.A_func(params)
        B = self.B_func(params)
        return A @ state + B @ u
    
    def g(self, state: Array, params: Array, t: float) -> Array:
        C = self.C_func(params)
        return C @ state