import jax
import jax.numpy as jnp
from jax import Array
import diffrax
from typing import Tuple, Dict, Callable
from scipy.optimize import minimize
from scipy.stats import chi2
from ..models.linearCT import LinearSystem
from ..idp import IdentificationProblem

"""
Performs identifiability analysis on a linear system from a single trajectory, based on 
the ICIS score from Qiu et.al. (2021).
 
@article{qiu2022identifiability,
  title={Identifiability analysis of linear ordinary differential equation systems with a single trajectory},
  author={Qiu, Xing and Xu, Tao and Soltanalizadeh, Babak and Wu, Hulin},
  journal={Applied Mathematics and Computation},
  volume={430},
  pages={127260},
  year={2022},
  publisher={Elsevier}
}
"""

class QiuICIS:
    def analyze(
        self, 
        model: LinearSystem, 
        x0: Array
    ) -> Dict:
        A = model.A
        n = model.n
        
        # Step 1: Compute eigenvalues, eigenvectors
        eigenvalues, eigenvectors = jnp.linalg.eig(A)
        
        # Check for singularity
        cond_num = jnp.linalg.cond(eigenvectors)
        if cond_num > 1.0 / jnp.finfo(eigenvectors.dtype).eps:
            return {
                "identifiable": False,
                "status": "Failed",
                "message": "System matrix A isn't diagonalizable.",
                "score": 0.0
            }
            
        # Step 2: Project x0 to the basis of eigenvectors
        coords = jnp.linalg.inv(eigenvectors) @ x0
            
        # Step 3: Compute the Initial Condition-Based Identifiability Score (ICIS)
        icis_score = jnp.prod(jnp.abs(coords)**2)
        
        is_identifiable = icis_score > 1e-15 # Floating point tolerance constant
        
        return {
            "identifiable": bool(is_identifiable),
            "status": "Success",
            "message": "System is identifiable from this initial condition." if is_identifiable 
                       else "System is not identifiable from this initial condition.",
            "score": float(icis_score)
        }
        
    def compute_sensitivity_trajectory(
        self,
        problem: IdentificationProblem,
    ) -> Tuple[Array, Array]:
        """
        Solves the augmented state and sensitivity equations for a given problem.

        The augmented state is [x, S_flat], where S is the sensitivity matrix
        S_ij = dx_i / d_theta_j.

        Args:
            problem: An IdentificationProblem instance containing the model,
                     time steps, and control inputs.

        Returns:
            A tuple containing:
            - The state trajectory (T, n).
            - The sensitivity trajectory (T, n, d), where d is the number of parameters.
        """
        sys = problem.sys
        n, d = sys.n, sys.d
        x0 = problem.initial_state 
        
        S0_flat = jnp.zeros(n * d) # initial sensitivity
        aug_y0 = jnp.concatenate([x0, S0_flat])

        def augmented_dynamics(t, y_aug, args): # Augmented dyamics function
            time_steps, u_exp, params, jax_instance = args
            
            x = y_aug[:n]
            S = y_aug[n:].reshape((n, d))

            control_func = diffrax.LinearInterpolation(ts=time_steps, ys=u_exp)
            u_t = control_func.evaluate(t)

            f_for_jac = lambda s, p: sys.f(s, u_t, p, t)

            df_dx = jax_instance.jacobian(f_for_jac, argnums=0)(x, params)
            df_dtheta = jax_instance.jacobian(f_for_jac, argnums=1)(x, params)

            dx_dt = sys.f(x, u_t, params, t)
            dS_dt = df_dx @ S + df_dtheta
            
            return jnp.concatenate([dx_dt, dS_dt.flatten()])

        term = diffrax.ODETerm(augmented_dynamics)
        t0, t1 = problem.time_steps[0], problem.time_steps[-1]
        solver_args = (problem.time_steps, problem.u_meas, sys.params, jax)
        saveat = diffrax.SaveAt(ts=problem.time_steps)

        solution = diffrax.diffeqsolve(
            term, sys.solver, t0, t1, dt0=None, y0=aug_y0, args=solver_args,
            saveat=saveat, stepsize_controller=sys.stepsize_controller,
            **sys.solver_options
        )

        x_traj = solution.ys[:, :n]
        S_traj = solution.ys[:, n:].reshape(-1, n, d)
        
        return x_traj, S_traj
    
    def _create_cost_function(self, problem: IdentificationProblem) -> Callable[[Array], float]:
        def costSSE(theta: Array) -> float:
            problem.sys.params = theta
            x_sim = problem.sys.simulate(problem.initial_state, problem.time_steps, problem.u_meas)
            y_sim = problem.sys.observe(x_sim, problem.time_steps)
            return jnp.sum((y_sim - problem.y_meas)**2)
        
    def analyze_LR(
        self,
        problem: IdentificationProblem,
        theta_guess: Array,
        alpha: float = 0.05
    ) -> Dict:
        """Performs practical identifiability analysis using the Likelihood Ratio Test."""
        cost_func = self._create_cost_function(problem)
        d = len(theta_guess)
        results = {}

        full_model_fit = minimize(cost_func, theta_guess, method='Nelder-Mead')
        rss_full = full_model_fit.fun
        theta_hat_full = full_model_fit.x
        
        results['full_model_fit'] = {'theta_hat': theta_hat_full, 'rss': rss_full}
        results['parameter_analysis'] = {}

        for i in range(d):
            theta_i_name = f"theta_{i}"
            
            def cost_func_reduced(theta_reduced):
                theta_full = jnp.insert(jnp.array(theta_reduced), i, theta_hat_full[i])
                return cost_func(theta_full)

            theta_reduced_guess = jnp.delete(theta_hat_full, i)
            reduced_model_fit = minimize(cost_func_reduced, theta_reduced_guess, method='Nelder-Mead')
            rss_reduced = reduced_model_fit.fun

            lr_statistic = len(problem.time_steps) * jnp.log(rss_reduced / rss_full)
            p_value = 1 - chi2.cdf(lr_statistic, df=1)
            is_identifiable = p_value < alpha

            results['parameter_analysis'][theta_i_name] = {
                'identifiable': is_identifiable,
                'p_value': float(p_value),
                'lr_statistic': float(lr_statistic)
            }
        return results