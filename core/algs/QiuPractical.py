import jax.numpy as jnp
from jax import Array
import numpy as np
from typing import Dict

from .Qiu2Step import two_stage_simple


class PracticalIdentifiabilityAnalyzer:
    """
    Implements the practical, data-driven identifiability scores from Qiu et al. (2022).
    """

    def _compute_pis(self, Y: Array, S: Array, L: Array, A_hat: Array) -> float:
        """
        Computes the Practical Identifiability Score (PIS)
        """
        p, T = Y.shape
        N = np.linalg.pinv(Y @ S @ Y.T)
        A = A_hat
        N2 = N @ N
        tAA = A.T @ A

        Y_L = Y @ L.T
        Y_S = Y @ S.T

        term1_vec = Y @ L @ L.T @ Y.T + p * (Y_L.T @ Y_L) + Y_L @ Y_L.T \
                    - 2 * A @ Y_S @ L.T @ Y.T \
                    - 2 * jnp.trace(A) * Y_S @ L.T @ Y.T \
                    - 2 * Y @ S @ L.T @ Y.T @ A.T \
                    + A @ Y_S @ S.T @ Y.T @ tAA \
                    + jnp.trace(tAA) * (Y_S.T @ Y_S) \
                    + Y @ S @ S.T @ Y.T @ tAA

        term1 = jnp.sum(N2.T.flatten('F') * term1_vec.flatten('F'))

        term2_mat = Y_L.T @ Y_L - 2 * Y_S.T @ A.T @ Y_L + Y_S.T @ tAA @ Y_S
        term2 = jnp.trace(term2_mat) * jnp.trace(N2)

        return float(term1 + term2)

    def _compute_scn(self, Y: Array, S: Array) -> float:
        """
        Computes the Smoothed Condition Number (SCN).
        """
        M = Y @ S @ Y.T
        return float(jnp.linalg.cond(M))

    def _compute_kappa(self, Y: Array) -> float:
        """
        Computes the kappa score from Stanhope.
        """
        p, T = Y.shape
        return float(jnp.linalg.cond(Y[:, :p]))

    def analyze(self, y_meas: Array, dt: float) -> Dict[str, float]:
        """
        Performs a full practical identifiability analysis on a given dataset.

        Args:
            y_meas: A (T, p) array of measurement data
            dt: Time step

        Returns:
            A dictionary containing the three practical identifiability scores.
        """
        two_stage_results = two_stage_simple(y_meas, dt)
        A_hat = two_stage_results['A_hat']
        S = two_stage_results['S']
        L = two_stage_results['L']

        Y_T = y_meas.T

        pis_score = self._compute_pis(Y_T, S, L, A_hat)
        scn_score = self._compute_scn(Y_T, S)
        kappa_score = self._compute_kappa(Y_T)

        return {
            "PIS": pis_score,
            "SCN": scn_score,
            "kappa": kappa_score
        }
