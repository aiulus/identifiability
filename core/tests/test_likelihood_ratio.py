
import pytest
import jax.numpy as jnp
import jax

from identifiability.core.algs.QiuICIS import QiuICIS
from identifiability.core.models.linearParametricCT import LinearParametricCT
from identifiability.core.idp import IdentificationProblem


@pytest.fixture
def make_lr_test_problem():
    """
    Creates an identification problem for testing the Likelihood Ratio method.
    Model: dx/dt = [[-p0, 0], [p0, -p1]]x
    - p0 is influential and should be identifiable.
    - p1 is not influential (as x2 is not observed) and should be unidentifiable.
    """
    n, m, p = 2, 0, 1
    
    A_func = lambda p: jnp.array([[-p[0], 0.0], [p[0], -p[1]]])
    C_func = lambda p: jnp.array([[1.0, 0.0]]) # Only observe x1
    # Define B_func, which returns a zero matrix as m=0
    B_func = lambda p: jnp.zeros((n, m))

    true_params = jnp.array([2.0, 5.0]) # True p0, p1
    # Correctly pass all required function arguments to the constructor
    model = LinearParametricCT(n, m, p, true_params, A_func=A_func, B_func=B_func, C_func=C_func)

    # Generate synthetic noisy data
    time_steps = jnp.linspace(0, 5, 51)
    x0 = jnp.array([10.0, 0.0])
    
    x_true = model.simulate(x0, time_steps)
    y_true = model.observe(x_true, time_steps)
    
    # Add a small amount of noise
    key = jax.random.PRNGKey(0)
    noise = jax.random.normal(key, y_true.shape) * 0.1 # 10% noise level
    y_meas = y_true + noise
    
    return IdentificationProblem(model, time_steps, initial_state=x0, y_meas=y_meas)


class TestLikelihoodRatioAnalysis:

    def test_lr_analysis(self, make_lr_test_problem):
        """
        Tests that the likelihood ratio method correctly identifies an influential
        parameter and correctly flags a non-influential one.
        """
        problem = make_lr_test_problem
        analyzer = QiuICIS()
        
        # Start with a slightly perturbed guess for the parameters
        theta_guess = problem.sys.params + 0.5
        
        results = analyzer.analyze_LR(problem, theta_guess)

        # Check the results
        param0_result = results['parameter_analysis']['theta_0']
        param1_result = results['parameter_analysis']['theta_1']

        assert param0_result['identifiable'], "Parameter 0 should be identifiable"
        assert not param1_result['identifiable'], "Parameter 1 should be unidentifiable"
        assert param0_result['p_value'] < 0.05
        assert param1_result['p_value'] > 0.05

