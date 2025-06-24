import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Sequence


def plot_heatmap_grid(results: Dict[str, np.ndarray],
                      n_list: Sequence[int],
                      m_list: Sequence[int],
                      pB_list: Sequence[float],
                      pA: float,
                      cmap: str = "viridis"):
    """
    Make a grid with `len(n_list)` rows (fixed state dim n) and three columns
    (criteria).  Inside each heat-map:
        x-axis  = input dimension m
        y-axis  = sparsity p_B
        colour  = mean failure probability.

    Parameters
    ----------
    results : dict  (key -> 3-D array shaped (n, m, pB))
    n_list, m_list, pB_list : sequences with the grid values
    pA      : sparsity of A (for title only)
    cmap    : matplotlib colour-map
    """
    crit_titles = {
        "rank_defect": "Rank defect\nrank(H) < n",
        "almost_zero": r"Near-zero $\sigma_{\min}$",
        "kalman_uncontrollable": "Kalman\nuncontrollable",
    }
    crit_keys = ["rank_defect", "almost_zero", "kalman_uncontrollable"]

    n_rows = len(n_list)
    n_cols = 3
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7 * n_cols, 3.2 * n_rows),
        sharex="col", sharey="row"
    )

    # --- iterate over rows (fixed n) and columns (criteria) -----------------
    for row_idx, n in enumerate(n_list):
        for col_idx, crit in enumerate(crit_keys):
            ax = axes[row_idx, col_idx] if n_rows > 1 else axes[col_idx]

            # slice data: shape (m, pB)  ->  transpose to (pB, m)
            data = results[crit][row_idx].T  # (pB, m)

            im = ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                cmap=cmap,
                vmin=0.0, vmax=1.0,
                extent=[m_list[0], m_list[-1], pB_list[0], pB_list[-1]],
            )

            # axis labels / titles
            if row_idx == 0:
                ax.set_title(crit_titles[crit], fontsize=14, pad=12)
            if col_idx == 0:
                ax.set_ylabel(r"$p_B$  (sparsity)", fontsize=12)
            if row_idx == n_rows - 1:
                ax.set_xlabel("input dim  $m$", fontsize=12)

            # tick formatting
            ax.set_xticks(m_list)
            ax.set_yticks(pB_list)

            # annotate left-hand side with the value of n
            if col_idx == 0:
                ax.text(
                    -0.25, 0.5, f"$n={n}$",
                    transform=ax.transAxes,
                    rotation=90,
                    va="center", ha="right",
                    fontsize=13, weight="bold"
                )

            # colour-bar only on the rightmost column
            if col_idx == n_cols - 1:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("failure prob.", rotation=270, labelpad=15)

    fig.suptitle(
        rf"Identifiability failure maps  |  fixed $p_A={pA}$",
        fontsize=16,
        y=1.02
    )
    plt.tight_layout()
    plt.show()


def main():
    data = np.load("outputs/hankel_grid_results_2.npz", allow_pickle=True)
    n_list = data["n_list"].tolist()
    m_list = data["m_list"].tolist()
    pB_list = data["pB_list"]
    pA = float(data["pA"])
    results = {k: data[k] for k in ("rank_defect", "almost_zero", "kalman_uncontrollable")}

    plot_heatmap_grid(results, n_list, m_list, pB_list, pA)


if __name__ == '__main__':
    main()
