# file: identifiability/core/tests/test_linear_system.py

import pytest
import jax.numpy as jnp
from identifiability.core.models.linearCT import LinearSystem

# --- Test System Fixtures ---

@pytest.fixture
def controllable_observable_system():
    """A simple double integrator, known to be controllable and observable."""
    A = jnp.array([[0., 1.], [0., 0.]])
    B = jnp.array([[0.], [1.]])
    C = jnp.array([[1., 0.]])
    return LinearSystem(A, B, C)

@pytest.fixture
def uncontrollable_system():
    """A system with an uncontrollable mode."""
    A = jnp.array([[1., 1., 0.], [0., 1., 0.], [0., 0., 2.]])
    B = jnp.array([[0.], [1.], [0.]])
    C = jnp.array([[1., 0., 0.]])
    return LinearSystem(A, B, C)

@pytest.fixture
def unobservable_system():
    """A system with an unobservable mode."""
    # This is the transpose of the uncontrollable system, a common way to create one.
    A = jnp.array([[1., 0., 0.], [1., 1., 0.], [0., 0., 2.]])
    B = jnp.array([[1.], [0.], [1.]])
    C = jnp.array([[0., 1., 0.]])
    return LinearSystem(A, B, C)

# --- Test Cases ---

class TestLinearSystemCriteria:

    @pytest.mark.parametrize("method", ["kalman", "pbh"])
    def test_is_controllable_true(self, controllable_observable_system, method):
        """Tests that a known controllable system is identified as such."""
        assert controllable_observable_system.is_controllable(method=method)

    @pytest.mark.parametrize("method", ["kalman", "pbh"])
    def test_is_controllable_false(self, uncontrollable_system, method):
        """Tests that a known uncontrollable system is identified as such."""
        assert not uncontrollable_system.is_controllable(method=method)

    @pytest.mark.parametrize("method", ["kalman", "pbh"])
    def test_is_observable_true(self, controllable_observable_system, method):
        """Tests that a known observable system is identified as such."""
        assert controllable_observable_system.is_observable(method=method)

    @pytest.mark.parametrize("method", ["kalman", "pbh"])
    def test_is_observable_false(self, unobservable_system, method):
        """Tests that a known unobservable system is identified as such."""
        assert not unobservable_system.is_observable(method=method)
        
    def test_simulation_with_control(self, controllable_observable_system):
        """
        Tests that simulation with a control input produces the expected change.
        """
        model = controllable_observable_system
        x0 = jnp.array([1.0, 0.0]) # Start at x=1, v=0
        time_steps = jnp.linspace(0, 1, 10)
        
        # Apply a constant positive control input (acceleration)
        u = jnp.ones((10, 1)) * 2.0 # u = 2.0
        
        trajectory = model.simulate(x0, time_steps, u=u)
        
        assert trajectory.shape == (10, 2)
        
        # For a double integrator starting at v=0 with constant positive acceleration,
        # the final position must be greater than the initial position.
        assert trajectory[-1, 0] > trajectory[0, 0]
        # The final velocity must be greater than the initial velocity.
        assert trajectory[-1, 1] > trajectory[0, 1]
