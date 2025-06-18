from abc import ABC, abstractmethod
from typing import Optional, Callable

import jax
import jax.numpy as jnp
from jax import Array
import diffrax

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
        self.p = p or n
        assert self.p <= self.n         
        if params is not None:
            self.d = max(params.shape)
            self.params = params 
        else:
            self.d = 0
        # Custom (y0, t, \theta) -> y(t) or RK of order 4
        self.integrator = integrator or self._rk4_step 

    
    @abstractmethod
    def f(self, state:Array, u: Optional[Array], params:Array, t:float) -> Array:
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
        params: Array, 
        t: float, 
        dt: float,
    ):
        k1 = f(x, u, params, t)
        k2 = f(x +0.5*dt*k1, u, params, t + 0.5*dt)
        k3 = f(x + 0.5*dt*k2, u, params, t + 0.5*dt)
        k4 = f(x+ dt*k3, u, params, t + dt)
        
        return x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    def simulate(self, x0: Array, time_steps: Array, u: Optional[Array]) -> Array:
        T = time_steps.shape[0]
        if u is None:#
            u = jnp.zeros((T, self.m))
        else:
            assert T == u.shape[0], f"Length of the control sequence must match that of time steps!"
            assert self.m == u.shape[1], f"Control inputs must have the dimension m = {self.m}! Provided sequence: ({controls.shape[0]}, {controls.shape[1]})"
        
        # DIFFRAX:
        #   - Requires continuous-time input signal
        #   - Convert the discrete control input sequence to a step function
        u_ct = diffrax.Step(ts=time_steps, ys=u)
        
        # Convert the dynamics function's signature to a Diffrax-compatible one
        def f_diffrax(t, y, args):
            params, controls = args
            u_t = controls.evaluate(t)
            return self.f(y, u_t, params, t)

        # Set up the ODE problem
        term = diffrax.ODETerm(f_diffrax)
        solver = diffrax.Tsit5() # General-purpose adaptive solver
        t0 = time_steps[0]
        t1 = time_steps[-1]
        
        args = (self.params, u_ct)
        saveat = diffrax.SaveAt(ts=time_steps)
        
        solution = diffrax.diffeqsolve(
            term, solver, t0, t1, dt0=None, y0=x0, args=args, saveat=saveat
        )
        
        return solution.ys
        
    
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