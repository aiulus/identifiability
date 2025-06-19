import pytest
import jax.numpy as jnp
from identifiability.core.models.linearParametricCT import LinearParametricCT

@pytest.fixture
def make_parametric_lti():
    """Factory to create a simple parameterized LTI system."""
    def _make(params):
        # A(theta) = [[-theta_1, 0], [0, -theta_2]]
        A_func = lambda p: jnp.diag(-jnp.array(p))
        # B and C are constant for simplicity
        B_func = lambda p: jnp.array([[1.], [1.]])
        C_func = lambda p: jnp.array([[1., 0.]])
        
        n, m, p = 2, 1, 1
        return LinearParametricCT(n, m, p, params, A_func, B_func, C_func)
    return _make

class TestLinearParametricCT:
    def test_creation_and_dynamics(self, make_parametric_lti):
        """Tests that the system is created and its dynamics are correct."""
        params = jnp.array([0.5, 2.0])
        model = make_parametric_lti(params)
        
        state = jnp.array([10., 20.])
        u = jnp.array([1.])
        
        # Expected dx/dt = A*x + B*u
        # A = [[-0.5, 0], [0, -2.0]]
        # Ax = [-5, -40], Bu = [1, 1]
        # dx/dt = [-4, -39]
        expected_dxdt = jnp.array([-4., -39.])
        
        dxdt = model.f(state, u, model.params, t=0)
        
        assert jnp.allclose(dxdt, expected_dxdt)
