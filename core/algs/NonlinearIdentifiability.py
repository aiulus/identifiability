import jax
import jax.numpy as jnp
from jax import Array
import numpy as np
from scipy.linalg import schur
from typing import Tuple, Dict, Callable

# from ..models.base import DynamicalSystem
from ..models.controlAffine import ControlAffineSystem
from ..idp import IdentificationProblem
from ..utils.algebra import lie_derivative


class NonlinearIdentifiability:
    def analyze_diffalg(self,
                        model: ControlAffineSystem,
                        x0: Array,
                        theta: Array,
                        num_samples: int = 10
                        ) -> Dict:
        """
        Performs structural identifiability analysis using a robust, multi-point
        sampling version of the differential algebra approach.

        Args:
            model: An instance of the ControlAffineSystem class.
            x0: The nominal initial condition.
            theta: The nominal parameter vector.
            num_samples: Number of random points to test around the nominal x0.
        """
        n, d = model.n, model.d
        num_derivatives = 2 * n
        h_components = [lambda x, i=i: model.h_output(x)[i] for i in range(model.p)]

        # JAX requires a key for random number generation
        key = jax.random.PRNGKey(0)

        # Generate random samples in the state space around x0
        random_points = jax.random.normal(key, shape=(num_samples, n)) * 0.5 + x0

        def get_rank_at_point(x_point: Array) -> int:
            """Helper function to compute the identifiability matrix rank at one point."""

            def output_derivatives_map(p: Array) -> Array:
                f_drift = lambda x: model.f_drift(x, p)
                derivatives = []
                for h_i in h_components:
                    L_f_h = lie_derivative(f_drift, h_i)
                    derivatives.append(L_f_h(x_point))
                    for _ in range(num_derivatives - 1):
                        L_f_h = lie_derivative(f_drift, L_f_h)
                        derivatives.append(L_f_h(x_point))
                return jnp.stack(derivatives)

            identifiability_matrix = jax.jacobian(output_derivatives_map)(theta)
            return jnp.linalg.matrix_rank(identifiability_matrix)

        # Compute the rank at each of the random points.
        ranks = jax.vmap(get_rank_at_point)(random_points)

        # --- LOGIC FIX ---
        # The most robust conclusion comes from the MINIMUM rank observed.
        # If the rank drops below d for any sample, it indicates a structural
        # deficiency. A max rank could be a numerical fluke.
        rank = jnp.min(ranks)
        is_identifiable = rank == d

        return {
            "identifiable": bool(is_identifiable),
            "status": "Success",
            "rank": int(rank),
            "num_params": d,
            "message": "All parameters are structurally locally identifiable." if is_identifiable
            else f"{d - int(rank)} parameter(s) or combination(s) are unidentifiable."
        }
