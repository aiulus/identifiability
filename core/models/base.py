from abc import ABC, abstractmethod
from typing import Optional, Callable

import jax.numpy as jnp
from jax import Array

class DynamicalSystem(ABC):
    """
    Abstract base class for dynamical systems.
    
    Subclasses should define:
    > System dimensions
        - n (int): dim(x)
        - m (int): dim(u)
        - p (int): dim(y)
        - d (int): dim(\theta)
    > Dynamics equations
        - f: X x U x \Theta \mapsto X
        - g: X x \Theta \mapsto Y
    """
    
    def __init__(self, n: int, m: int, p: Optional[int] = None, d: Optional[int] = None):
        assert isinstance(n, int) and n > 0
        self.n = n
        if d is not None:
            assert isinstance(d, int) and d > 0
        self.d = d
        self.p = p or n
        assert self.p <= self.n        

    
    @abstractmethod
    def f(self, state:Array, input: Array, t:float) -> Array:
        """
        System dynamics. Computes the derivative of the state vector at time t.
        
        Defaults to zero dynamics.
        
        Args:
            state: x(t) - shape: (n, 1)
            input: u(t) - shape: (m, 1)
            params: \theta - shape: (d, 1)
            t: Scalar time \in \mathbb{R}
            
        Returns:
            \dot{x}(t) \in \mathbb{R}^{n \times 1}
        """
        
        return jnp.zeros_like(state)
        
    @abstractmethod
    def g(self, state:Array, params:Array, t:float) -> Array:
        """
        Measurement / observation dynamics. Maps the system state to a measurement.
        
        Defaults to the identity map (full observability) if not overridden.
        
        Args:
            state: x(t) - shape: (n, 1)
            params: \theta - shape: (d, 1)
            t: Scalar time \in \mathbb{R}
            
        Returns:
            \dot{x}(t) \in \mathbb{R}^{n \times 1}
        """
        