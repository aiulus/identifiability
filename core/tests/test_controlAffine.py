# file: identifiability/core/tests/test_nonlinear_system.py

import pytest
import jax.numpy as jnp
from identifiability.core.models.controlAffine import ControlAffineSystem
from identifiability.core.utils.algebra import lie_derivative, lie_bracket

# --- Test Math Functions ---

def test_lie_derivative():
    """Tests the Lie derivative with a known example."""
    # Let f(x) = [x_2, -x_1] (rotation)
    # Let h(x) = x_1^2
    # L_f h(x) = ∇h · f = [2x_1, 0] · [x_2, -x_1] = 2*x_1*x_2
    f = lambda x: jnp.array([x[1], -x[0]])
    h = lambda x: x[0]**2
    x_point = jnp.array([2.0, 3.0])

    Lf_h = lie_derivative(f, h)
    
    assert jnp.isclose(Lf_h(x_point), 2 * 2.0 * 3.0)

def test_lie_bracket():
    """Tests the Lie bracket with a known example."""
    # Let f(x) = [x_2, 0], g(x) = [0, x_1]
    # ∇f = [[0, 1], [0, 0]], ∇g = [[0, 0], [1, 0]]
    # [f, g](x) = ∇g·f - ∇f·g = [[0,0],[1,0]]@[x2,0] - [[0,1],[0,0]]@[0,x1]
    #            = [0, x2] - [x1, 0] = [-x1, x2]
    f = lambda x: jnp.array([x[1], 0.0])
    g = lambda x: jnp.array([0.0, x[0]])
    x_point = jnp.array([2.0, 3.0])
    
    bracket_fg = lie_bracket(f, g)
    
    expected_result = jnp.array([-x_point[0], x_point[1]])
    assert jnp.allclose(bracket_fg(x_point), expected_result)


@pytest.fixture
def observable_bilinear_system():
    """A simple bilinear system known to be observable."""
    n, m = 2, 1
    f_drift = lambda x: jnp.array([x[1]**2, -x[0]])
    g_control = [lambda x: jnp.array([0., 1.])]
    h_output = lambda x: jnp.array([x[0]]) # y = x_1
    return ControlAffineSystem(n, m, f_drift, g_control, h_output)

@pytest.fixture
def unobservable_bilinear_system():
    """A system where the second state is not observable."""
    n, m = 2, 1
    f_drift = lambda x: jnp.array([0., 0.]) # No dynamics
    g_control = [lambda x: jnp.array([1., 0.])]
    h_output = lambda x: jnp.array([x[0]**2]) # y = x_1^2
    return ControlAffineSystem(n, m, f_drift, g_control, h_output)

class TestNonlinearObservability:

    def test_observability_true(self, observable_bilinear_system):
        """Tests that a known observable system is identified as such."""
        # Test away from the origin, where observability might be lost
        test_point = jnp.array([1.0, 1.0])
        assert observable_bilinear_system.is_locally_observable(test_point)

    def test_observability_false(self, unobservable_bilinear_system):
        """Tests that a known unobservable system is identified as such."""
        test_point = jnp.array([1.0, 1.0])
        assert not unobservable_bilinear_system.is_locally_observable(test_point)
        
    def test_simulation_runs(self, observable_bilinear_system):
        """A simple test to ensure simulation runs for the nonlinear class."""
        model = observable_bilinear_system
        x0 = jnp.array([1.0, 1.0])
        time_steps = jnp.linspace(0, 1, 10)
        trajectory = model.simulate(x0, time_steps)
        assert trajectory.shape == (10, model.n)

