import pytest
import jax.numpy as jnp
from jax import Array

from ..algs.NonlinearIdentifiability import NonlinearIdentifiability
from ..models.controlAffine import ControlAffineSystem


@pytest.fixture
def lotka_volterra_model():
    """
    A Lotka-Volterra predator-prey model.
    States: x1 (prey), x2 (predator)
    Params: theta = [alpha, beta, gamma, delta]
    dx1/dt = alpha*x1 - beta*x1*x2
    dx2/dt = delta*x1*x2 - gamma*x2
    """
    n, m, p = 2, 0, 1  # 2 states, 0 inputs, 1 output

    # The drift field is a function of state x and params theta
    def f_drift(x, theta):
        alpha, beta, gamma, delta = theta
        return jnp.array([
            alpha * x[0] - beta * x[0] * x[1],
            delta * x[0] * x[1] - gamma * x[1]
        ])

    # No control inputs for this model
    g_control = []

    # The output function only measures the prey population
    h_output = lambda x: jnp.array([x[0]])

    # A nominal set of parameters to perform the analysis at
    params = jnp.array([1.1, 0.4, 1.1, 0.4])

    return ControlAffineSystem(n, m, p, params, f_drift, g_control, h_output)


class TestNonlinearStructuralIdentifiability:

    def test_lotka_volterra_unidentifiable(self, lotka_volterra_model):
        """
        Tests the Lotka-Volterra model. When only observing the prey (x1),
        the parameters governing the predator dynamics (gamma) should be
        unidentifiable. The analysis should reveal a rank deficiency.
        """
        model = lotka_volterra_model
        analyzer = NonlinearIdentifiability()

        # A nominal initial condition and parameter vector for the test
        x0 = jnp.array([20.0, 5.0])
        theta = model.params

        result = analyzer.analyze_diffalg(model, x0, theta)

        # The rank should be less than the number of parameters
        assert not result["identifiable"]
        assert result["rank"] < result["num_params"]
        assert result["status"] == "Success"

