import pytest
import jax.numpy as jnp

from identifiability.core.algs.QiuICIS import QiuICIS
from identifiability.core.models.linearCT import LinearSystem
from identifiability.core.models.linearParametricCT import LinearParametricCT
from identifiability.core.idp import IdentificationProblem

# --- Existing Fixtures and Tests for ICIS (Unchanged) ---
@pytest.fixture
def lti_system_distinct_eigs():
    A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
    B = jnp.zeros((2, 1))
    C = jnp.zeros((1, 2))
    return LinearSystem(A, B, C)

class TestLinearSingleTrajectoryAnalysis:
    # ... (all previous ICIS tests remain here) ...
    def test_identifiable_case(self, lti_system_distinct_eigs):
        # ...
        pass
    # ... etc

# --- New Tests for Sensitivity Analysis ---
@pytest.fixture
def make_simple_parametric_model():
    """A simple 1D system: dx/dt = -p*x, where theta = [p]"""
    A_func = lambda p: jnp.array([[-p[0]]])
    B_func = lambda p: jnp.zeros((1, 0)) # No input
    C_func = lambda p: jnp.array([[1.]])
    return lambda p_val: LinearParametricCT(1, 0, 1, jnp.array([p_val]), A_func, B_func, C_func)

class TestSensitivityAnalysis:
    def test_sensitivity_calculation(self, make_simple_parametric_model):
        """
        Tests the sensitivity calculation against a known analytical solution.
        For x' = -p*x, the analytical solution is x(t) = x0*exp(-p*t).
        The sensitivity S = dx/dp = -t*x0*exp(-p*t) = -t * x(t).
        """
        p = 2.0
        model = make_simple_parametric_model(p)
        x0 = jnp.array([10.0])
        time_steps = jnp.linspace(0, 2, 21)
        initial_state = jnp.zeros((model.n, 1))

        # Create the problem. Note: IdentificationProblem needs `initial_state`
        # We need to update the problem class or add it here.
        # For now, let's assume problem has an `initial_state` attribute.
        problem = IdentificationProblem(model, time_steps, initial_state=initial_state)
        problem.initial_state = x0

        analyzer = QiuICIS()
        x_traj, s_traj = analyzer.compute_sensitivity_trajectory(problem)

        # Calculate the analytical sensitivity
        s_analytical = -time_steps[:, None, None] * x_traj[:, :, None]
        
        assert x_traj.shape == (21, 1)
        assert s_traj.shape == (21, 1, 1)
        assert jnp.allclose(s_traj, s_analytical, atol=1e-4)

