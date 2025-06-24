import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
import json
from typing import Dict, Sequence

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
        for i, n in enumerate(n_list):
            for j, m in enumerate(m_list):
                pbar.set_description(f"Processing (n={n}, m={m})")
                for k, pB in enumerate(pB_list):
                    key, sub = jax.random.split(key)
                    trial_keys = jax.random.split(sub, n_samples)

                    batch_results = jax.vmap(
                        lambda k_: run_single_instance(n, m, pA, pB, k_)
                    )(trial_keys)

                    for crit in results:
                        results[crit][i, j, k] = jnp.mean(batch_results[crit])
                pbar.update(1)
    return results


# -------------------------------------------------------------------
# 1)  Heat-maps with axes swapped  (m on y-axis, p_B on x-axis)
# -------------------------------------------------------------------
def plot_heatmaps(results: Dict[str, np.ndarray],
                  n_list, m_list, pB_list,
                  pA: float,
                  cmap: str = "viridis"):
    """
    Grid of heat-maps:
        rows    -> fixed n (state dim)
        cols    -> identifiability criteria
        x-axis  -> sparsity p_B
        y-axis  -> input dimension m
    """
    crit_keys = ["rank_defect", "almost_zero", "kalman_uncontrollable"]
    crit_titles = {
        "rank_defect":           "Rank defect\nrank(H) < n",
        "almost_zero":           r"Near-zero $\sigma_{\min}$",
        "kalman_uncontrollable": "Kalman\nuncontrollable",
    }

    n_rows, n_cols = len(n_list), 3
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.5 * n_cols, 3.0 * n_rows),
        sharex="row",
        sharey="col",
    )

    # iterate over (row=n, col=criterion)
    for r, n in enumerate(n_list):
        for c, crit in enumerate(crit_keys):
            ax = axes[r, c] if n_rows > 1 else axes[c]

            # results[crit][r]  shape = (m, pB)
            data = results[crit][r]            # no transpose now!

            im = ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=0.0, vmax=1.0,
                extent=[pB_list[0], pB_list[-1], m_list[0], m_list[-1]],
            )

            # column titles (top row only)
            if r == 0:
                ax.set_title(crit_titles[crit], fontsize=14, pad=10)

            # row label (left column only)
            if c == 0:
                ax.set_ylabel("input dim $m$\n"
                               + rf"$n={int(n)}$", rotation=0,
                               labelpad=45, fontsize=12)

            # axes ticks
            ax.set_xticks(pB_list)
            ax.set_yticks([int(m) for m in m_list])

            # x-label only on bottom row
            if r == n_rows - 1:
                ax.set_xlabel("sparsity $p_B$", fontsize=12)

            # colour-bar on rightmost column
            if c == n_cols - 1:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("failure prob.", rotation=270, labelpad=15)

    fig.suptitle(rf"Identifiability failure maps — fixed $p_A={pA}$",
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# 2)  Same plot, but with bilinear interpolation => visible gradients
# -------------------------------------------------------------------
def plot_heatmaps_gradient(results: Dict[str, np.ndarray],
                           n_list, m_list, pB_list,
                           pA: float,
                           cmap: str = "viridis"):
    """
    Variant of `plot_heatmaps` that sets `interpolation="bilinear"` so
    Matplotlib blends neighbouring colour cells.  The gradient you see is
    simply the continuous mapping of *data values* (failure probabilities in
    [0,1]) through the chosen colormap.

    Mechanics
    ----------
    * **Data values**   Each entry in `results[...]` is a mean failure
      probability for a (n, m, p_B) triple.
    * **Colormap**      `cmap` (default *viridis*) defines how numbers map to
      colours.
    * **Mapping**       `imshow` looks up the colour for every matrix entry.
    * **Interpolation** Setting `interpolation="bilinear"` makes Matplotlib
      smooth the transitions between cell centres, generating the visible
      gradient.
    * **vmin/vmax**     Fixed at (0,1) so the full range of the colormap
      corresponds to probabilities 0 % … 100 %.
    """
    crit_keys = ["rank_defect", "almost_zero", "kalman_uncontrollable"]
    crit_titles = {
        "rank_defect":           "Rank defect\nrank(H) < n",
        "almost_zero":           r"Near-zero $\sigma_{\min}$",
        "kalman_uncontrollable": "Kalman\nuncontrollable",
    }

    n_rows, n_cols = len(n_list), 3
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6.5 * n_cols, 3.0 * n_rows),
        sharex="row",
        sharey="col",
    )

    for r, n in enumerate(n_list):
        for c, crit in enumerate(crit_keys):
            ax = axes[r, c] if n_rows > 1 else axes[c]

            data = results[crit][r]  # (m, pB)

            im = ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                interpolation="bilinear",   # <-- smooth gradient
                cmap=cmap,
                vmin=0.0, vmax=1.0,
                extent=[pB_list[0], pB_list[-1], m_list[0], m_list[-1]],
            )

            if r == 0:
                ax.set_title(crit_titles[crit], fontsize=14, pad=10)
            if c == 0:
                ax.set_ylabel("input dim $m$\n"
                               + rf"$n={int(n)}$", rotation=0,
                               labelpad=45, fontsize=12)

            ax.set_xticks(pB_list)
            ax.set_yticks([int(m) for m in m_list])

            if r == n_rows - 1:
                ax.set_xlabel("sparsity $p_B$", fontsize=12)

            if c == n_cols - 1:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("failure prob.", rotation=270, labelpad=15)

    fig.suptitle(rf"Gradient heat-maps — fixed $p_A={pA}$",
                 fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()



def main():
    n_list = [3, 5, 10, 20]  # [3, 5, 10, 20, 30, 40, 50]
    m_list = [3, 5, 10, 20] # [3, 5, 10, 20, 30, 40, 50]

    pB_list = jnp.linspace(0.05, 1.0, 11)
    pA = 0.5
    n_samples = 100  # 100
    master_key = jax.random.PRNGKey(0)

    results = compute_grid(n_list, m_list, pB_list, pA, n_samples, master_key)

    # --- Save the Results ---
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    results_filepath = os.path.join(output_dir, "hankel_grid_results_ntrials_1.npz")
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
    plot_heatmaps(results, n_list, m_list, pB_list, pA)
    plot_heatmaps_gradient(results, n_list, m_list, pB_list, pA)


if __name__ == '__main__':
    main()
