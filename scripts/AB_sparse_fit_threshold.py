#!/usr/bin/env python
# run_threshold_models.py
# ---------------------------------------------------------------
# Train OR load two surrogate models for P_fail(pA,pB,n,m):
#   1) logistic polynomial (degree-2)
#   2) Gaussian-process regression on grid means
# Then visualise a slice & export helper functions.
# ---------------------------------------------------------------

import argparse, os, joblib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# ------------------------------------------------------------------
# ----------  0. utilities -----------------------------------------
# ------------------------------------------------------------------
def make_dirs():
    os.makedirs("models",  exist_ok=True)
    os.makedirs("figures", exist_ok=True)

def load_tensor(path: str):
    data = np.load(path)
    return (data["n_list"], data["m_list"],
            data["pA_list"], data["pB_list"],
            data["P_rank"])                      # tensor means


# ------------------------------------------------------------------
# ----------  1. logistic-polynomial surface -----------------------
# ------------------------------------------------------------------
def build_or_load_logistic(X_full, y_full, fname="models/logistic_poly.joblib"):
    if Path(fname).exists():
        return joblib.load(fname)

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LogisticRegression

    log_poly = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False),
        LogisticRegression(max_iter=500, solver="lbfgs")
    ).fit(X_full, y_full)

    joblib.dump(log_poly, fname)
    return log_poly


# ------------------------------------------------------------------
# ----------  2. GP regression on grid means -----------------------
# ------------------------------------------------------------------
def build_or_load_gp(X_grid, y_grid, alpha, enc,
                     fname="models/gp_surface_reg.joblib"):
    if Path(fname).exists():
        return joblib.load(fname)

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel

    d_cat = X_grid.shape[1] - 2
    kernel = 1.0 * RBF(length_scale=[0.1, 0.1] + [1.0] * d_cat) \
             + WhiteKernel(1e-4)

    gpr = GaussianProcessRegressor(kernel, alpha=alpha,
                                   normalize_y=True).fit(X_grid, y_grid)

    joblib.dump((gpr, enc), fname)
    return gpr, enc


# ------------------------------------------------------------------
# ----------  3. helper wrappers -----------------------------------
# ------------------------------------------------------------------
def logistic_prob(model, pA, pB, n, m):
    return model.predict_proba([[pA, pB, n, m]])[0, 1]

def logistic_grad(model, pA, pB, n, m):
    # analytic derivative w.r.t pA,pB (rows) ; we only need those two dims
    coef = model[-1].coef_.ravel()
    z    = model[-1].decision_function([[pA, pB, n, m]])[0]
    s    = 1 / (1 + np.exp(-z))
    β = coef
    dz_dpA = β[0] + 2*β[4]*pA + β[5]*pB + β[6]*n + β[7]*m
    dz_dpB = β[1] + 2*β[5]*pB + β[8]*pA + β[9]*n + β[10]*m
    grad   = s*(1-s) * np.array([dz_dpB, dz_dpA])      # ∂/∂(pB,m)
    return grad

def gp_prob(gpr, enc, pA, pB, n, m):
    x_cat = enc.transform([[float(n), float(m)]])
    x     = np.hstack([[pA, pB], x_cat.ravel()])
    return gpr.predict(x.reshape(1, -1))[0]


# ------------------------------------------------------------------
# ----------  4. main analysis -------------------------------------
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor", default="outputs/4d_hankel_results_2.npz")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--slice_pA",  type=float, default=0.5)
    parser.add_argument("--slice_n",   type=int,   default=5)
    args = parser.parse_args()

    make_dirs()
    n_list, m_list, pA_list, pB_list, P_rank = load_tensor(args.tensor)

    # full tall table for logistic
    rows, labels = [], []
    for i,n in enumerate(n_list):
        for j,m in enumerate(m_list):
            for a,pA in enumerate(pA_list):
                for b,pB in enumerate(pB_list):
                    p_fail = P_rank[i,j,a,b]
                    k_fail = int(round(p_fail * args.n_samples))
                    rows.extend([[pA, pB, n, m]] * args.n_samples)
                    labels.extend([1]*k_fail + [0]*(args.n_samples-k_fail))
    X_full = np.asarray(rows, float)
    y_full = np.asarray(labels, int)

    log_poly = build_or_load_logistic(X_full, y_full)

    # grid means for GP
    rows, targets, alphas = [], [], []
    for i,n in enumerate(n_list):
        for j,m in enumerate(m_list):
            for a,pA in enumerate(pA_list):
                for b,pB in enumerate(pB_list):
                    mean = P_rank[i,j,a,b]
                    rows.append([pA,pB,n,m])
                    targets.append(mean)
                    alphas.append(mean*(1-mean)/args.n_samples + 1e-4)
    X_grid = np.asarray(rows, float)
    y_grid = np.asarray(targets, float)
    alpha  = np.asarray(alphas, float)

    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(sparse_output=False).fit(X_grid[:,2:])
    Xg  = np.hstack([X_grid[:,:2], enc.transform(X_grid[:,2:])])
    gpr, enc = build_or_load_gp(Xg, y_grid, alpha, enc)

    # ---------------------------------------------------------------
    # pick a slice (pA*, n*) and build dense mesh for (pB,m)
    # ---------------------------------------------------------------
    pA_star = args.slice_pA
    n_star  = args.slice_n
    pB_mesh = np.linspace(0.05, 1.0, 200)
    m_mesh = np.asarray(m_list, float)  # only trained m values (categorical)

    PP_gp = np.array([[gp_prob(gpr, enc, pA_star, pB, n_star, m)
                       for pB in pB_mesh] for m in m_mesh])
    PP_gp = gaussian_filter(PP_gp, sigma=1.0)

    suffix = f"_n{n_star}_pA{pA_star:.2f}".replace(".", "")

    # ---- GP contour & steepness -----------------------------------
    # ---- GP probability surface ------------------------------------
    plt.figure(figsize=(7, 4))
    extent = [pB_mesh[0], pB_mesh[-1], m_mesh[0], m_mesh[-1]]

    im = plt.imshow(PP_gp, origin="lower", extent=extent,
                    aspect="auto", cmap="viridis", vmin=0, vmax=1)

    plt.colorbar(im, label="failure prob.")  # <── use im, place *before* contour
    CS = plt.contour(pB_mesh, m_mesh, PP_gp, levels=[0.5],
                     colors="r", linewidths=2)

    plt.title(f"GP prob surface & 0.5 contour  (pA={pA_star}, n={n_star})")
    plt.xlabel(r"sparsity $p_B$")
    plt.ylabel("input dim $m$")
    plt.savefig(f"figures/gp_surface_slice{suffix}.png", dpi=300)

    # gradient magnitude
    dP_dpB = np.gradient(PP_gp, pB_mesh, axis=1)
    dP_dm  = np.gradient(PP_gp, m_mesh,  axis=0)
    gamma  = np.hypot(dP_dpB, dP_dm)

    plt.figure(figsize=(7, 4))
    im = plt.imshow(gamma, origin="lower", extent=extent,
                    aspect="auto", cmap="plasma")
    plt.colorbar(im, label=r"$\gamma$")
    plt.title(r"Steepness $\gamma=\|\nabla P\|$")
    plt.xlabel(r"sparsity $p_B$")
    plt.ylabel("input dim $m$")
    plt.savefig(f"figures/gp_steepness_slice{suffix}.png", dpi=300)

    plt.close("all")

    # ---------------------------------------------------------------
    print("All done.  Figures saved in ./figures, models in ./models.")

# ------------------------------------------------------------------
if __name__ == "__main__":
    main()
