# file: identifiability/core/tests/test_base.py

import pytest
import jax
import jax.numpy as jnp
from identifiability.core.models.base import DynamicalSystem
from typing import Optional

# =============================================================================
# Helper "Dummy" System for Testing the Base Class
# =============================================================================

class DummyLinearSystem(DynamicalSystem):
    """
    A simple, predictable linear system for testing base class features.
    The dynamic is dx/dt = -a*x, which has the analytical solution
    x(t) = x0 * exp(-a*t).
    The parameter 'a' is the first element of the params array.
    """
    def __init__(self, params: jnp.ndarray, initial_state: jnp.ndarray, m: int = 0):
        super().__init__(n=initial_state.shape[0], m=m, p=initial_state.shape[0], params=params)
        self.initial_state = initial_state

    def f(self, state: jnp.ndarray, u: Optional[jnp.ndarray], params: jnp.ndarray, t: float) -> jnp.ndarray:
        """Implements dx/dt = -a*x + b*u"""
        a = params[0]
        # Handle control input if present
        b_u = 0.0
        if self.m > 0:
            b = params[1]
            b_u = b * u
        return -a * state + b_u

    def g(self, state: jnp.ndarray, params: jnp.ndarray, t: float) -> jnp.ndarray:
        """Observation is simply the state (identity map)."""
        return state

# =============================================================================
# Tests for DynamicalSystem Methods
# =============================================================================

class TestDynamicalSystem:
    """
    Test suite for the generic methods of the DynamicalSystem base class.
    """

    # --- _rk4_step Method Tests (tested via simulate) ---
    
    def test_rk4_numerical_accuracy(self):
        """
        Tests RK4 accuracy for a single step against an analytical solution.
        """
        a = 0.1
        x0 = jnp.array([10.0])
        model = DummyLinearSystem(params=jnp.array([a]), initial_state=x0)
        
        dt = 0.5
        time_steps = jnp.array([0.0, dt])

        # Simulate one step
        trajectory = model.simulate(x0, time_steps, u=None)
        x_rk4 = trajectory[-1]
        
        # Analytical solution: x(t) = x0 * exp(-a*t)
        x_analytical = x0 * jnp.exp(-a * dt)
        
        assert jnp.allclose(x_rk4, x_analytical, atol=1e-5)

    def test_rk4_dt_zero(self):
        """
        Tests that a time step of zero (dt=0) results in no change to the state.
        """
        model = DummyLinearSystem(params=jnp.array([0.1]), initial_state=jnp.array([10.0, 5.0]))
        x0 = model.initial_state
        time_steps = jnp.array([0.0, 0.0, 0.0]) # A sequence of zero-dt steps

        trajectory = model.simulate(x0, time_steps, u=None)
        
        assert trajectory.shape == (3, 2)
        assert jnp.allclose(trajectory[0], x0)
        assert jnp.allclose(trajectory[1], x0)
        assert jnp.allclose(trajectory[2], x0)

    # --- simulate Method Tests ---

    @pytest.mark.parametrize("time_steps", [
        jnp.linspace(0, 1, 5),  # Uniform steps
        jnp.array([0.0, 0.2, 0.3, 0.7, 1.0]), # Non-uniform steps
    ])
    def test_simulate_trajectory_properties(self, time_steps):
        """
        Tests basic trajectory properties like initial state and output shape.
        """
        model = DummyLinearSystem(params=jnp.array([0.1]), initial_state=jnp.array([10.0]))
        x0 = model.initial_state
        
        trajectory = model.simulate(x0, time_steps, u=None)
        
        # Check that the first state of the trajectory is the initial state
        assert jnp.allclose(trajectory[0], x0)
        
        # Check that the output shape is (T, n)
        assert trajectory.shape == (time_steps.shape[0], model.n)

    def test_simulate_control_input_handling(self):
        """
        Tests that the `simulate` method correctly handles control inputs.
        """
        # System with control input: dx/dt = -ax + bu
        params = jnp.array([0.1, 2.0]) # a, b
        x0 = jnp.array([1.0])
        model = DummyLinearSystem(params=params, initial_state=x0, m=1)
        
        T = 5
        time_steps = jnp.linspace(0, 1, T)
        
        # 1. Test with controls=None (should default to zeros)
        traj_none = model.simulate(x0, time_steps, u=None)
        assert traj_none.shape == (T, model.n)

        # 2. Test with a valid controls array
        u = jnp.ones((T, model.m))
        traj_valid = model.simulate(x0, time_steps, u=u)
        assert traj_valid.shape == (T, model.n)
        # With positive control, final state should be higher than with no control
        assert traj_valid[-1] > traj_none[-1]

        # 3. Test with controls of incorrect dimension `m`
        with pytest.raises(AssertionError, match=r"dimension m = 1"):
            bad_controls_m = jnp.ones((T, model.m + 1))
            model.simulate(x0, time_steps, u=bad_controls_m)

        # 4. Test with controls of incorrect length `T`
        with pytest.raises(AssertionError, match=r"match that of time steps"):
            bad_controls_t = jnp.ones((T - 1, model.m))
            model.simulate(x0, time_steps, u=bad_controls_t)

    def test_simulate_edge_case_single_timestep(self):
        """
        Tests simulation with only one time step, which should just return the initial state.
        """
        model = DummyLinearSystem(params=jnp.array([0.1]), initial_state=jnp.array([10.0]))
        x0 = model.initial_state
        time_steps = jnp.array([0.0]) # Single time point
        
        trajectory = model.simulate(x0, time_steps, u=None)
        
        assert trajectory.shape == (1, model.n)
        assert jnp.allclose(trajectory[0], x0)

    # --- observe Method Tests ---

    def test_observe_method(self):
        """
        Tests the observe method for correct output shape and vmap behavior.
        """
        model = DummyLinearSystem(params=jnp.array([0.1]), initial_state=jnp.array([10.0, 5.0]))
        
        # Generate a sample trajectory
        time_steps = jnp.linspace(0, 1, 10)
        states = model.simulate(model.initial_state, time_steps, u=None)
        
        # Get the observations
        outputs = model.observe(states, time_steps)
        
        # Check output dimensions are (T, p)
        assert outputs.shape == (time_steps.shape[0], model.p)
        
        # For DummyLinearSystem, g is identity, so states and outputs should be identical
        assert jnp.allclose(states, outputs)

    # --- perturb_* Method Tests ---

    def test_perturb_methods_default_behavior(self):
        """
        Tests that the default perturb methods return the input array unchanged.
        """
        model = DummyLinearSystem(params=jnp.array([0.1]), initial_state=jnp.array([10.0]))
        states = jnp.arange(10).reshape(5, 2)
        
        # Test process noise perturbation
        perturbed_states = model.perturb_process(states)
        assert jnp.all(perturbed_states == states)

        # Test measurement noise perturbation
        perturbed_outputs = model.perturb_measurements(states) # Use states as dummy outputs
        assert jnp.all(perturbed_outputs == states)
