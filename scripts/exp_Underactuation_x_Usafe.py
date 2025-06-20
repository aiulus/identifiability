import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import cont2discrete
from scipy.linalg import logm

from nfoursid.kalman import Kalman
from nfoursid.nfoursid import NFourSID

from core.models.linearCT import LinearSystem



# --- ALGORITHM IMPLEMENTATIONS ---

def identify_ls(y: np.ndarray, u: np.ndarray, dt: float) -> (np.ndarray, np.ndarray):
    """
    Identifies a discrete-time state-space model (Ad, Bd) using one-shot Least Squares.

    Args:
        y: Output data matrix (num_samples, p). This algorithm naively assumes y=x.
        u: Input data matrix (num_samples, m).
        dt: The sampling time step.

    Returns:
        A tuple of the estimated (A, B) continuous-time matrices.
    """
    x = y
    X_next = x[1:].T
    X_current = x[:-1].T
    U_current = u[:-1].T

    PHI = np.vstack([X_current, U_current])

    try:
        THETA = X_next @ np.linalg.pinv(PHI)
    except np.linalg.LinAlgError:
        return np.nan, np.nan

    n, m = x.shape[1], u.shape[1]
    Ad_hat = THETA[:, :n]
    Bd_hat = THETA[:, n:]

    try:
        A_hat = logm(Ad_hat).real / dt
        B_hat = np.linalg.pinv(A_hat) @ (Ad_hat - np.eye(n)) @ np.linalg.pinv(logm(Ad_hat)) @ Bd_hat * (1 / dt)
    except (ValueError, np.linalg.LinAlgError):
        return np.nan, np.nan

    return A_hat, B_hat


def identify_n4sid(y: np.ndarray, u: np.ndarray, dt: float, state_size: int) -> (np.ndarray, np.ndarray):
    """
    Identifies a state-space model (A, B) using the N4SID algorithm.
    """
    output_cols = [f'y{i}' for i in range(y.shape[1])]
    input_cols = [f'u{i}' for i in range(u.shape[1])]

    y_df = pd.DataFrame(y, columns=output_cols)
    u_df = pd.DataFrame(u, columns=input_cols)

    combined_df = pd.concat([y_df, u_df], axis=1)

    # Instantiate NFourSID with the data and column names
    model = NFourSID(
        combined_df,
        output_columns=output_cols,
        input_columns=input_cols
    )

    # --- FIX: Pass the state_size to the subspace_identification method ---
    # This is the crucial step that sets the model's state dimension.
    model.subspace_identification(state_size=state_size)

    output_size = y.shape[1]
    input_size = u.shape[1]
    qrs_initial = np.eye(output_size + input_size)
    kalman = Kalman(model, qrs_initial)
    kalman.kalman_filtering()

    Ad_hat, Bd_hat, _, _ = kalman.get_mats()

    try:
        A_hat = logm(Ad_hat).real / dt
        B_hat = np.linalg.pinv(A_hat) @ (Ad_hat - np.eye(state_size)) @ np.linalg.pinv(logm(Ad_hat)) @ Bd_hat * (1 / dt)
    except (ValueError, np.linalg.LinAlgError):
        return np.nan, np.nan

    return A_hat, B_hat


def calculate_relative_error(A_true: np.ndarray, A_est: np.ndarray) -> float:
    """Calculates the relative estimation error using the Frobenius norm."""
    if not isinstance(A_est, np.ndarray) or np.any(np.isnan(A_est)):
        return np.inf
    return np.linalg.norm(A_true - A_est, 'fro') / np.linalg.norm(A_true, 'fro')


def main():
    # --- SYSTEM DEFINITION ---
    N_STATES = 5
    A_true = np.diag(np.ones(N_STATES - 1), k=1)
    A_true[-1, -1] = -0.5
    A_true[-1, -2] = -2
    A_true[2, 0] = -0.1

    B_symmetric = np.zeros((N_STATES, 3));
    B_symmetric[0, 0] = 1;
    B_symmetric[2, 1] = 1.5;
    B_symmetric[4, 2] = 1
    B_asymmetric = np.zeros((N_STATES, 3));
    B_asymmetric[0, 0] = 2.0;
    B_asymmetric[1, 0] = 0.5;
    B_asymmetric[2, 1] = 0.2;
    B_asymmetric[4, 2] = 1.0

    C_true = np.zeros((3, N_STATES));
    C_true[0, 0] = 1;
    C_true[1, 2] = 1;
    C_true[2, 4] = 1

    # --- EXPERIMENT SETUP ---
    dt = 0.1
    t_span = np.arange(0, 100, dt)
    u1, u2, u3 = np.sin(0.1 * t_span) + np.cos(1 * t_span), np.sin(0.2 * t_span) + np.cos(0.5 * t_span), np.sin(
        0.5 * t_span) + np.cos(2 * t_span)
    u_base = np.vstack([u1, u2, u3]).T
    x0 = np.zeros(N_STATES)

    # --- RUN EXPERIMENTS ---
    results = {}

    def run_scenario(u_exp, model):
        """Helper to generate data and run both identification algorithms."""
        x_data = np.array(model.simulate(x0, t_span, u=u_exp))
        y_data = x_data @ model.C.T

        A_ls, _ = identify_ls(x_data, u_exp, dt)
        A_n4sid, _ = identify_n4sid(y_data, u_exp, dt, N_STATES)

        return (calculate_relative_error(A_true, A_ls),
                calculate_relative_error(A_true, A_n4sid))

    # Exp 1: Underactuation (Symmetric B)
    results['exp1'] = {'LS': [], 'N4SID': []}
    model_sym = LinearSystem(A=A_true, B=B_symmetric, C=C_true)
    for num_inputs in [3, 2, 1]:
        u_exp = u_base.copy()
        if num_inputs < 3: u_exp[:, num_inputs:] = 0
        err_ls, err_n4sid = run_scenario(u_exp, model_sym)
        results['exp1']['LS'].append(err_ls)
        results['exp1']['N4SID'].append(err_n4sid)

    # Exp 2: Underactuation (Asymmetric B)
    results['exp2'] = {'LS': [], 'N4SID': []}
    model_asym = LinearSystem(A=A_true, B=B_asymmetric, C=C_true)
    u_exp = u_base.copy();
    u_exp[:, 2] = 0
    err_ls, err_n4sid = run_scenario(u_exp, model_asym)
    results['exp2']['LS'].append(err_ls)
    results['exp2']['N4SID'].append(err_n4sid)

    # Exp 3: Actuation Constraints
    results['exp3'] = {'LS': [], 'N4SID': []}
    for limit in [np.inf, 0.75, 0.25]:
        u_exp = np.clip(u_base, -limit, limit)
        err_ls, err_n4sid = run_scenario(u_exp, model_sym)
        results['exp3']['LS'].append(err_ls)
        results['exp3']['N4SID'].append(err_n4sid)

    # Exp 4: Combined Constraints
    results['exp4'] = {'LS': [], 'N4SID': []}
    u_exp = np.clip(u_base, -0.75, 0.75);
    u_exp[:, 2] = 0
    err_ls, err_n4sid = run_scenario(u_exp, model_sym)
    results['exp4']['LS'].append(err_ls)
    results['exp4']['N4SID'].append(err_n4sid)

    # --- VISUALIZE RESULTS ---
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    fig.suptitle('System Identification Performance Under Constraints', fontsize=18)
    axes = axes.flatten()

    bar_width = 0.4
    # Plot Exp 1
    x1 = np.arange(3)
    axes[0].bar(x1 - bar_width / 2, results['exp1']['LS'], bar_width, label='Naive LS')
    axes[0].bar(x1 + bar_width / 2, results['exp1']['N4SID'], bar_width, label='Robust N4SID')
    axes[0].set_title('Experiment 1: Underactuation (Symmetric B)')
    axes[0].set_xticks(x1, ['3 Inputs', '2 Inputs', '1 Input'])

    # Plot Exp 2
    axes[1].bar(0 - bar_width / 2, results['exp2']['LS'], bar_width, label='Naive LS')
    axes[1].bar(0 + bar_width / 2, results['exp2']['N4SID'], bar_width, label='Robust N4SID')
    axes[1].set_title('Experiment 2: Underactuation (Asymmetric B)')
    axes[1].set_xticks([0], ['2 of 3 Inputs Active'])

    # Plot Exp 3
    x3 = np.arange(3)
    axes[2].bar(x3 - bar_width / 2, results['exp3']['LS'], bar_width, label='Naive LS')
    axes[2].bar(x3 + bar_width / 2, results['exp3']['N4SID'], bar_width, label='Robust N4SID')
    axes[2].set_title('Experiment 3: Actuation Saturation')
    axes[2].set_xticks(x3, ['Unconstrained', '75% Limit', '25% Limit'])

    # Plot Exp 4
    axes[3].bar(0 - bar_width / 2, results['exp4']['LS'], bar_width, label='Naive LS')
    axes[3].bar(0 + bar_width / 2, results['exp4']['N4SID'], bar_width, label='Robust N4SID')
    axes[3].set_title('Experiment 4: Combined Constraints')
    axes[3].set_xticks([0], ['2 Inputs w/ 75% Limit'])

    for ax in axes:
        ax.set_ylabel('Relative Estimation Error ||A - Â|| / ||A||')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.show()


if __name__ == '__main__':
    main()
