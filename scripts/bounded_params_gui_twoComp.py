from core.utils.identifiability import verify_identifiability_criteria
from core.models.pkpd.bellman_unidf_twoComp import BellmanAstrom
from core.utils.hankel import build_LTI_hankel, H_wd
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import diffrax

T = 10.0 # trajectory length
dt = 0.1 # time step
n_samples = 50 # number of parameter samples


def sample_params(bounds, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    return [{k: float(rng.uniform(lo, hi)) for k, (lo, hi) in bounds.items()}
            for _ in range(n_samples)]

def simulate_trajs(params_list):
    solver = diffrax.Dopri5()
    X_T = []
    ts = np.arange(0, T + dt, dt)
    for p in params_list:
        sys = BellmanAstrom(p, solver=solver)
        sol = sys.simulate(ts)
        X_T.append(sol)
    return X_T

def getSigmaSVD(X_T, order_estimate: int):
    H = H_wd(X_T, order_estimate)
    _, S, _ = np.linalg.svd(H, full_matrices=False)
    return S

def plot_singular_values(sigma_i, sample_idx):
    plt.figure()
    plt.semilogy(sigma_i, 'o-')
    plt.title(f"Sample {sample_idx}: Hankel Singular Values")
    plt.xlabel("Index")
    plt.ylabel("Singular Value (log scale)")
    plt.grid(True)
    plt.show()

def main():
    param_bounds = {
        'k1': (0.1, 2.0),
        'k2': (0.1, 2.0),
        'k3': (0.5, 1.5),  # known bound
        'k4': (0.1, 1.0),  # known bound
        'u': (-1.0, 1.0)  # initial condition sampled
    }

    samples = sample_params(param_bounds)
    X_T = simulate_trajs(samples)

    for i, X_T in enumerate(X_T):

        crit = verify_identifiability_criteria(states, sys.B)
        print(f"Sample {i} identifiability criteria:")
        pprint(crit)

        # 7b. SVD analysis on data
        svals = getSigmaSVD(np.array(outputs), order=2)
        plot_singular_values(svals, i)

if __name__ == '__main__':
    main()
