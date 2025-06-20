import jax
import jax.numpy as jnp
from jax import Array
import diffrax
from typing import Tuple

from ..idp import IdentificationProblem


class SensitivityAnalyzer:
    """
    Provides methods for sensitivity analysis of dynamical systems.
    """

    def compute_sensitivity_trajectory(
            self,
            problem: IdentificationProblem,
    ) -> Tuple[Array, Array]:
        """
        Solves the augmented state and sensitivity equations for a given problem.
        The augmented state is [x, S_flat], where S is the sensitivity matrix
        S_ij = dx_i / d_theta_j.
        """
        model = problem.model
        n, d = model.n, model.d
        x0 = problem.initial_state

        # S0 is the initial sensitivity, which is zero since x0 does not depend on theta.
        S0_flat = jnp.zeros(n * d)
        aug_y0 = jnp.concatenate([x0, S0_flat])

        def augmented_dynamics(t, y_aug, args):
            time_steps, u_exp, params, jax_instance = args

            x = y_aug[:n]
            S = y_aug[n:].reshape((n, d))

            control_func = diffrax.LinearInterpolation(ts=time_steps, ys=u_exp)
            u_t = control_func.evaluate(t)

            f_for_jac = lambda s, p: model.f(s, u_t, p, t)
            df_dx = jax_instance.jacobian(f_for_jac, argnums=0)(x, params)
            df_dtheta = jax_instance.jacobian(f_for_jac, argnums=1)(x, params)

            dx_dt = model.f(x, u_t, params, t)
            dS_dt = df_dx @ S + df_dtheta

            return jnp.concatenate([dx_dt, dS_dt.flatten()])

        term = diffrax.ODETerm(augmented_dynamics)
        t0, t1 = problem.time_steps[0], problem.time_steps[-1]
        solver_args = (problem.time_steps, problem.u_exp, model.params, jax)
        saveat = diffrax.SaveAt(ts=problem.time_steps)

        solution = diffrax.diffeqsolve(
            term, model.solver, t0, t1, dt0=None, y0=aug_y0, args=solver_args,
            saveat=saveat, stepsize_controller=model.stepsize_controller,
            **model.solver_options
        )

        x_traj = solution.ys[:, :n]
        S_traj = solution.ys[:, n:].reshape(-1, n, d)

        return x_traj, S_traj
