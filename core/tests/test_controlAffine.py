import pytest
import jax.numpy as jnp
from identifiability.core.models.controlAffine import ControlAffineSystem
from identifiability.core.utils.algebra import lie_derivative, lie_bracket

# --- Test Math Functions (Unchanged) ---

def test_lie_derivative():
    f = lambda x: jnp.array([x[1], -x[0]])
    h = lambda x: x[0]**2
    x_point = jnp.array([2.0, 3.0])
    Lf_h = lie_derivative(f, h)
    assert jnp.isclose(Lf_h(x_point), 12.0)

def test_lie_bracket():
    f = lambda x: jnp.array([x[1], 0.0])
    g = lambda x: jnp.array([0.0, x[0]])
    x_point = jnp.array([2.0, 3.0])
    bracket_fg = lie_bracket(f, g)
    expected_result = jnp.array([-2.0, 3.0])
    assert jnp.allclose(bracket_fg(x_point), expected_result)

# --- Test Nonlinear System Fixtures ---

@pytest.fixture
def observable_bilinear_system():
    n, m = 2, 1
    f_drift = lambda x: jnp.array([x[1]**2, -x[0]])
    g_control = [lambda x: jnp.array([0., 1.])]
    h_output = lambda x: jnp.array([x[0]])
    return ControlAffineSystem(n, m, f_drift, g_control, h_output)

@pytest.fixture
def unobservable_bilinear_system():
    n, m = 2, 1
    f_drift = lambda x: jnp.array([0., 0.])
    g_control = [lambda x: jnp.array([1., 0.])]
    h_output = lambda x: jnp.array([x[0]**2])
    return ControlAffineSystem(n, m, f_drift, g_control, h_output)

@pytest.fixture
def controllable_nonholonomic_system():
    """The 'nonholonomic integrator', a classic controllable nonlinear system."""
    n, m = 3, 2
    f_drift = lambda x: jnp.array([0., 0., 0.])
    g_control = [
        lambda x: jnp.array([1., 0., -x[1]]),
        lambda x: jnp.array([0., 1., 0.])
    ]
    # This system has no meaningful output, so we define a dummy one.
    h_output = lambda x: jnp.array([x[0]])
    return ControlAffineSystem(n, m, f_drift, g_control, h_output)

@pytest.fixture
def uncontrollable_simple_system():
    """A simple system where motion is restricted to one dimension."""
    n, m = 2, 1
    f_drift = lambda x: jnp.array([x[1], 0.])
    g_control = [lambda x: jnp.array([1., 0.])]
    h_output = lambda x: jnp.array([x[0]])
    return ControlAffineSystem(n, m, f_drift, g_control, h_output)


# --- Test Classes ---

class TestNonlinearObservability:
    def test_observability_true(self, observable_bilinear_system):
        test_point = jnp.array([1.0, 1.0])
        assert observable_bilinear_system.is_locally_observable(test_point)

    def test_observability_false(self, unobservable_bilinear_system):
        test_point = jnp.array([1.0, 1.0])
        assert not unobservable_bilinear_system.is_locally_observable(test_point)
        
    def test_simulation_runs(self, observable_bilinear_system):
        model = observable_bilinear_system
        x0 = jnp.array([1.0, 1.0])
        time_steps = jnp.linspace(0, 1, 10)
        trajectory = model.simulate(x0, time_steps)
        assert trajectory.shape == (10, model.n)

class TestNonlinearControllability:
    def test_controllability_true(self, controllable_nonholonomic_system):
        """Tests that a known controllable system is identified as such."""
        test_point = jnp.array([1.0, 2.0, 3.0])
        assert controllable_nonholonomic_system.is_locally_controllable(test_point)

    def test_controllability_false(self, uncontrollable_simple_system):
        """Tests that a known uncontrollable system is identified as such."""
        test_point = jnp.array([1.0, 1.0])
        assert not uncontrollable_simple_system.is_locally_controllable(test_point)

