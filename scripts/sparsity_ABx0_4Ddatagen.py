import os

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
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


def compute_grid_4d(n_list, m_list,
                    pA_list, pB_list,
                    n_samples: int,
                    master_key: jax.random.PRNGKey):
    """
    Monte-Carlo estimate of failure probability tensor

        P[crit, n_idx, m_idx, pA_idx, pB_idx]

    crit ∈ {"rank_defect", "almost_zero", "kalman_uncontrollable"}
    """
    shape_4d = (len(n_list), len(m_list), len(pA_list), len(pB_list))
    results = {
        "rank_defect": np.zeros(shape_4d),
        "almost_zero": np.zeros(shape_4d),
        "kalman_uncontrollable": np.zeros(shape_4d),
    }

    key = master_key
    total = len(n_list) * len(m_list) * len(pA_list)
    with tqdm(total=total, desc="4-D grid progress") as pbar:
        for i, n in enumerate(n_list):
            for j, m in enumerate(m_list):
                for a, pA in enumerate(pA_list):
                    pbar.set_description(f"(n={n}, m={m}, pA={pA:0.2f})")
                    for b, pB in enumerate(pB_list):
                        key, sub = jax.random.split(key)
                        trial_keys = jax.random.split(sub, n_samples)

                        batch = jax.vmap(
                            lambda k_: run_single_instance(int(n), int(m),
                                                           float(pA), float(pB),
                                                           k_)
                        )(trial_keys)

                        for crit in results:
                            results[crit][i, j, a, b] = jnp.mean(batch[crit])
                    pbar.update(1)
    return results


def threshold_contour(S,
                      pA_list,
                      pB_list,
                      *,
                      level: float = 0.5,
                      sigma: float = 1.0):
    """
    Return a list of NumPy arrays, each array = one poly-line of the
    iso-contour  S(p_A,p_B)=level.

    The function is backend-agnostic: it first tries `.collections`, then
    falls back to `.allsegs` when the former is absent (as with the
    QuadContourSet you observed).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    # 1) smooth Monte-Carlo noise (optional)
    if sigma > 0:
        S = gaussian_filter(S, sigma=sigma)

    # 2) build the contour (quietly – no axes needed)
    CS = plt.contour(np.asarray(pB_list, float),
                     np.asarray(pA_list, float),
                     np.asarray(S,   float),
                     levels=[level],
                     colors="r")

    # 3) extract path vertices in a version-robust way
    curves = []

    # Newer Matplotlib: artists live in CS.collections[i].get_paths()
    if hasattr(CS, "collections") and CS.collections:
        for coll in CS.collections:
            for path in coll.get_paths():
                curves.append(path.vertices)          # (N,2) array
    # Fallback: use the raw segment list
    elif hasattr(CS, "allsegs"):
        for seg in CS.allsegs[0]:                     # first (= only) level
            curves.append(np.vstack(seg))             # list of arrays → (N,2)
    else:
        raise RuntimeError("Could not extract contour vertices from Matplotlib object")

    plt.close()
    return curves        # each vertex array has columns [p_B, p_A]



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
        "rank_defect": "Rank defect\nrank(H) < n",
        "almost_zero": r"Near-zero $\sigma_{\min}$",
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
            data = results[crit][r]  # no transpose now!

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


# Same plot with bilinear interpolation
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
        "rank_defect": "Rank defect\nrank(H) < n",
        "almost_zero": r"Near-zero $\sigma_{\min}$",
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
                interpolation="bilinear",  # <-- smooth gradient
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
    n_list = [3, 5]
    m_list = [3, 5]
    # pA_list = jnp.linspace(0.05, 1.0, 11)
    # pB_list = jnp.linspace(0.05, 1.0, 11)
    pA_list = jnp.linspace(0.1, 1.0, 11)
    pB_list = jnp.linspace(0.1, 1.0, 11)

    n_samples = 10
    master_key = jax.random.PRNGKey(0)

    P = compute_grid_4d(n_list, m_list,
                        pA_list, pB_list,
                        n_samples, master_key)

    # ---------- persist -----------------------------------------------------
    os.makedirs("outputs", exist_ok=True)
    np.savez(
        "outputs/4d_hankel_results_2.npz",
        n_list=n_list, m_list=m_list,
        pA_list=pA_list, pB_list=pB_list,
        P_rank=P["rank_defect"],
        P_gap=P["almost_zero"],
        P_ctrl=P["kalman_uncontrollable"],
    )
    print("4-D tensor saved to outputs/4d_hankel_results_2.npz")

    # ---- analyse one slice  ----------------------------------------------
    n_idx, m_idx = 0, 1  # e.g. n=5 , m=10
    S = P["rank_defect"][n_idx, m_idx]  # shape (K,L) over pA × pB
    curves = threshold_contour(S, pA_list, pB_list)

    # quick visual check ----------------------------------------------------
    plt.figure(figsize=(6, 4))
    extent = (
        float(pB_list[0]), float(pB_list[-1]),  # x-min, x-max   (p_B)
        float(pA_list[0]), float(pA_list[-1]),  # y-min, y-max   (p_A)
    )

    plt.imshow(
        np.asarray(S),  # make sure the data itself is NumPy, too
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
        vmin=0, vmax=1,
    )

    for curve in curves:
        plt.plot(curve[:, 0], curve[:, 1], 'r', lw=2)
    plt.xlabel(r"sparsity $p_B$")
    plt.ylabel(r"sparsity $p_A$")
    plt.title(r"$\;P_{\text{fail}}$ surface  +  0.5 contour  ($n=5,m=10$)")
    plt.colorbar(label="failure prob.")
    plt.show()


if __name__ == '__main__':
    main()
