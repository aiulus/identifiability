from abc import ABC, abstractmethod
from typing import Optional, Callable, Union

import jax
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
    
    def __init__(self, 
                 n: int, # State dim.
                 m: int, # Input dim.
                 p: Optional[int] = None, #  Output dim.
                 params: Optional[Array] = None,
                 integrator: Optional[Callable]=None):
        assert isinstance(n, int) and n > 0
        self.n = n 
        self.m = m 
        assert self.p <= self.n 
        self.p = p or n
        if params:
            self.d = max(params.shape)
            self.params = params 
        else:
            self.d = 0
        # Custom (y0, t, \theta) -> y(t) or RK of order 4
        self.integrator = integrator or self._rk4_step 

    
    @abstractmethod
    def f(self, state:Array, input: Optional[Array], t:float) -> Array:
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
        
    def _rk4_step(
        self, 
        f: Callable, 
        x: Array,
        u: Optional[Array], 
        t: float, 
        dt: float,
    ):
        k1 = f(x, u, t)
        k2 = f(x +0.5*dt*k1, u, t + 0.5*dt)
        k3 = f(x + 0.5*dt*k2, u, t + 0.5*dt)
        k4 = f(x+ dt*k3, u, t + dt)
        
        return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    def simulate(self, x0: Array, time_steps: Array, controls: Optional[Array]) -> Array:
        T = time_steps.shape[0]
        if controls is None:
            controls = jnp.zeros((T, self.m))
        else:
            assert T == controls.shape[0], f"Length of the control sequence must match that of time steps!"
            assert self.m == controls.shape[1], f"Control inputs must have the dimension m = {self.m}! Provided sequence: ({controls.shape[0]}, {controls.shape[1]})"
        
        def step(carry, idx):
            x_prev, t_prev = carry
            t_curr = time_steps[idx]
            dt = t_curr - t_prev
            u_curr = controls[idx]
            x_next = self.integrator(self.f, x_prev, u_curr, self.params, t_prev, dt)
            return (x_next, t_curr), x_next
        
        _, xs = jax.lax.scan(step, (x0, time_steps[0]), jnp.arange(1, T))
        return jnp.vstack((x0[None, :], xs))
    
    def observe(self, states: Array, times : Array) -> Array:
        vg = jax.vmap(lambda x, t: self.g(x, self.params, t), in_axes=(0,0))
        return vg(states, times)
    
    def perturb_process(self, states: Array) -> Array:
        """
        Override to inject process noise.
        """
        return states
    
    def perturb_measurements(self, outputs: Array) -> Array:
        """
        Override to inject measurement noise.
        """
        return outputs