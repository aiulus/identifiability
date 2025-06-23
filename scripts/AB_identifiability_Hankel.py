import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import json
from typing import Dict

from core.utils.hankel import build_LTI_hankel
from core.utils.random_matrices import sparse_matrix
from core.models.linearCT import LinearSystem


def verify_identifiability_criteria(A: jnp.ndarray, B: jnp.ndarray, *, sigma_tol: float = 1e-12) -> Dict[str, bool]:
    n = A.shape[0]
    H = build_LTI_hankel(A, B, n)

    # --- Criterion 1 --- Rank deficiency
    rank_defect = jnp.linalg.matrix_rank(H) < n

    # --- Criterion 2 --- Almost zero Singular Value
    sigma_min = jnp.linalg.svd(H, compute_uv=False)[-1]
    almost_zero = sigma_min <= sigma_tol

    # --- Criterion 3 --- Kalman Uncontrollability
    model = LinearSystem(A, B, C=jnp.eye(n))
    kalman_uncontrollable = jnp.logical_not(model.is_controllable(method='kalman'))

    result = {
        "rank_defect": rank_defect,
        "almost_zero": almost_zero,
        "kalman_uncontrollable": kalman_uncontrollable,
    }

    return result


def run_single_instance(n: int, m: int, pA: float, pB: float, key: jax.random.PRNGKey) -> dict:
    """
    Wrapper that samples (A, B) and evaluates all three criteria.
    """
    key_A, key_B = jax.random.split(key)

    A = sparse_matrix((n, n), pA, key_A)
    B = sparse_matrix((n, m), pB, key_B)
    return verify_identifiability_criteria(A, B)


def compute_grid(n_list, m_list, pB_list, pA: float, n_samples: int, master_key: jax.random.PRNGKey):
    shape = (len(n_list), len(m_list), len(pB_list))
    results = {
        "rank_defect": np.zeros(shape),
        "almost_zero": np.zeros(shape),
        "kalman_uncontrollable": np.zeros(shape),
    }
    key = master_key
    total_combinations = len(n_list) * len(m_list)
    with tqdm(total=total_combinations, desc="Total Grid Progress") as pbar:
        for i, n in enumerate(n_list, desc="State dim. (n)"):
            for j, m in enumerate(m_list, desc="Input dim. (m)", leave=False):
                for k, pB in enumerate(pB_list):
                    key, sub = jax.random.split(key)
                    trial_keys = jax.random.split(sub, n_samples)

                    batch_results = jax.vmap(
                        lambda k_: run_single_instance(n, m, pA, pB, k_)
                    )(trial_keys)

                    for crit in results:
                        results[crit][i, j, k] = jnp.mean(batch_results[crit])

    return results


def plot_heatmaps(results: Dict[str, np.ndarray],
                  n_list, m_list, pB_list,
                  pA: float, pB_slice_idx: int):
    """Side-by-side heat-maps for the three criteria at fixed p_B."""
    pB_val = pB_list[pB_slice_idx]
    crit_titles = {
        "rank_defect": "Rank defect  (rank(H) < n)",
        "almost_zero": r"Near-zero $\sigma_{\min}$",
        "kalman_uncontrollable": "Kalman uncontrollable",
    }

    fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)
    fig.suptitle(f"Identifiability vs (n, m)  |  p_A={pA:.2f}, p_B={pB_val:.2f}",
                 fontsize=16)

    for ax, crit in zip(axes, results):
        data = results[crit][:, :, pB_slice_idx].T  # m (rows) × n (cols)
        im = ax.imshow(data, origin="lower", aspect="auto",
                       cmap="viridis", vmin=0, vmax=1)
        ax.set_title(crit_titles[crit])
        ax.set_xlabel("state dim n")
        ax.set_xticks(range(len(n_list)), labels=n_list)

        fig.colorbar(im, ax=ax, label="fail prob.")

    axes[0].set_ylabel("input dim m")
    axes[0].set_yticks(range(len(m_list)), labels=m_list)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def main():
    n_list = [3, 5, 10, 20, 30, 40, 50]  # [3, 5, 10, 20, 30, 40, 50]
    m_list = [3, 5, 10, 20, 30, 40, 50]

    pB_list = jnp.linspace(0.05, 1.0, 11)
    pA = 0.5
    n_samples = 10  # 100
    master_key = jax.random.PRNGKey(0)

    results = compute_grid(n_list, m_list, pB_list, pA, n_samples, master_key)

    # --- Save the Results ---
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    results_filepath = os.path.join(output_dir, "hankel_grid_results_2.npz")
    np.savez(
        results_filepath,
        n_list=n_list,
        m_list=m_list,
        pB_list=pB_list,
        pA=pA,
        **results
    )
    print(f"\nExperiment results saved to: {results_filepath}")

    # --- Plotting ---
    middle_idx = len(pB_list) // 2
    plot_heatmaps(results, n_list, m_list, pB_list, pA, pB_slice_idx=middle_idx)


if __name__ == '__main__':
    main()
