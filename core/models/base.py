from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any

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
        - params: Model parameters
        - solver: A Diffrax solver instance (e.g., diffrax.Tsit5())
        - solver_options: A dictionary of options to pass to diffrax.diffeqsolve
    > Dynamics equations
        - f: X x U x \Theta \mapsto X
        - g: X x \Theta \mapsto Y
    """

    def __init__(self,
                 n: int,  # State dim.
                 m: int,  # Input dim.
                 p: Optional[int] = None,  # Output dim.
                 params: Optional[Array] = None,
                 solver: Optional[diffrax.AbstractSolver] = None,
                 stepsize_ctrlr: Optional[diffrax.AbstractStepSizeController] = None,
                 solver_options: Optional[Dict[str, Any]] = None):
        assert isinstance(n, int) and n > 0
        self.n = n
        self.m = m
        self.p = p or n
        assert self.p <= self.n
        if params is not None:
            self.d = max(params.shape) if params.shape else 0
            self.params = params
        else:
            self.d = 0
            self.params = None
        self.solver = solver or diffrax.Tsit5()
        self.solver_options = solver_options or {}
        # Ensure a robust adaptive controller is used by default.
        self.stepsize_controller = stepsize_ctrlr or diffrax.PIDController(rtol=1e-5, atol=1e-5)

    @abstractmethod
    def f(self, state: Array, u: Optional[Array], params: Array, t: float) -> Array:
        """System dynamics: dx/dt = f(x, u, params, t)"""
        raise NotImplementedError

    @abstractmethod
    def g(self, state: Array, params: Array, t: float) -> Array:
        """Measurement dynamics: y = g(x, params, t)"""
        raise NotImplementedError

    def simulate(self, x0: Array, time_steps: Array, u: Optional[Array] = None) -> Array:
        """Simulates the system trajectory using the configured Diffrax solver."""
        T = time_steps.shape[0]
        if u is None:
            u = jnp.zeros((T, self.m))
        else:
            assert T == u.shape[0], f"Length of the control sequence must match that of time steps!"
            assert self.m == u.shape[
                1], f"Control inputs must have the dimension m = {self.m}! Provided sequence: ({u.shape[0]}, {u.shape[1]})"

        # DIFFRAX:
        #   - Requires continuous-time input signal
        #   - Apply linear interpolation to obtain it
        u_ct = diffrax.LinearInterpolation(ts=time_steps, ys=u)

        # Convert the dynamics function's signature to a Diffrax-compatible one
        def f_diffrax(t, y, args):
            params, controls = args
            u_t = controls.evaluate(t)
            return self.f(y, u_t, params, t)

        # Set up the ODE problem
        term = diffrax.ODETerm(f_diffrax)
        t0, t1 = time_steps[0], time_steps[-1]
        args = (self.params, u_ct)
        saveat = diffrax.SaveAt(ts=time_steps)

        dt0 = self.solver_options.get('dt0')
        if dt0 is None and len(time_steps) > 1:
            dt0 = time_steps[1] - time_steps[0]

        solution = diffrax.diffeqsolve(
            term,
            self.solver,
            t0,
            t1,
            dt0=dt0,
            y0=x0,
            args=args,
            saveat=saveat,
            stepsize_controller=self.stepsize_controller,
            **self.solver_options
        )

        return solution.ys

    def observe(self, states: Array, times: Array) -> Array:
        """Applies the observation function to a trajectory of states."""
        params = self.params if self.params is not None else jnp.array([])
        vg = jax.vmap(lambda x, t, p: self.g(x, params, t), in_axes=(0, 0, None))
        return vg(states, times, params)

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

    # TODO: Redundant / remove    
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
        k2 = f(x + 0.5 * dt * k1, u, params, t + 0.5 * dt)
        k3 = f(x + 0.5 * dt * k2, u, params, t + 0.5 * dt)
        k4 = f(x + dt * k3, u, params, t + dt)

        return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
