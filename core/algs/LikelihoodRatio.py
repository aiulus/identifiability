import jax.numpy as jnp
from jax import Array
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2
from typing import Dict, Callable

from ..idp import IdentificationProblem


class LikelihoodRatioAnalyzer:
    """
    Implements practical identifiability analysis using the Likelihood Ratio Test.
    """

    def _create_cost_function(self, problem: IdentificationProblem) -> Callable[[np.ndarray], float]:
        """Creates the sum-of-squared-errors cost function for SciPy's optimizer."""

        def cost_function(theta_np: np.ndarray) -> float:
            theta_jax = jnp.array(theta_np)
            # This assumes the model has a `params` attribute that can be updated.
            problem.model.params = theta_jax

            x_sim = problem.model.simulate(
                problem.initial_state, problem.time_steps, problem.u_exp
            )
            y_sim = problem.model.observe(x_sim, problem.time_steps)

            sse = jnp.sum((y_sim - problem.y_exp) ** 2)
            return float(sse)

        return cost_function

    def analyze(
            self,
            problem: IdentificationProblem,
            theta_guess: Array,
            alpha: float = 0.05
    ) -> Dict:
        """
        Performs the Likelihood Ratio Test for each parameter.
        """
        cost_func = self._create_cost_function(problem)
        d = len(theta_guess)
        results = {}

        # Use NumPy array for SciPy optimizer
        theta_guess_np = np.array(theta_guess)

        full_model_fit = minimize(cost_func, theta_guess_np, method='Nelder-Mead')
        rss_full = full_model_fit.fun
        theta_hat_full = full_model_fit.x

        results['full_model_fit'] = {'theta_hat': jnp.array(theta_hat_full), 'rss': rss_full}
        results['parameter_analysis'] = {}

        for i in range(d):
            theta_i_name = f"theta_{i}"

            def cost_func_reduced(theta_reduced_np):
                theta_full_np = np.insert(theta_reduced_np, i, theta_hat_full[i])
                return cost_func(jnp.array(theta_full_np))

            theta_reduced_guess_np = np.delete(theta_hat_full, i)
            reduced_model_fit = minimize(cost_func_reduced, theta_reduced_guess_np, method='Nelder-Mead')
            rss_reduced = reduced_model_fit.fun

            # Handle case where the fit is perfect to avoid log(0)
            if rss_full < 1e-9:
                lr_statistic = np.inf if rss_reduced > rss_full else 0.0
            else:
                lr_statistic = len(problem.time_steps) * np.log(rss_reduced / rss_full)

            p_value = 1 - chi2.cdf(lr_statistic, df=1)
            is_identifiable = p_value < alpha

            results['parameter_analysis'][theta_i_name] = {
                'identifiable': is_identifiable,
                'p_value': float(p_value),
                'lr_statistic': float(lr_statistic)
            }
        return results
