import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from typing import Callable
from jax import Array

from core.models.bio.SimpleGlucoseInsulinModel import SimpleGlucoseInsulinModel
from core.idp import IdentificationProblem

def create_cost_function(problem: IdentificationProblem) -> Callable[[np.ndarray], float]:
    """
    Creates the sum-of-squared-errors cost function for SciPy's optimizer.
    This function wraps the JAX-based simulation to be compatible with SciPy.
    """
    def cost_function(theta_np: np.ndarray) -> float:
        # Convert NumPy array from SciPy to a JAX array
        theta_jax = jnp.array(theta_np)
        
        # Update the model's parameters for this evaluation
        problem.sys.params = theta_jax
        
        # Simulate the model with the current parameter estimate
        x_sim = problem.sys.simulate(problem.initial_state, problem.time_steps, problem.u_meas)
        y_sim = problem.sys.observe(x_sim, problem.time_steps)
        
        # Compute sum of squared errors and return as a standard Python float
        sse = jnp.sum((y_sim - problem.y_meas)**2)
        return float(sse)
    
    return cost_function

def run_parameter_estimation(problem: IdentificationProblem, theta_guess: np.ndarray) -> np.ndarray:
    """
    Runs the parameter estimation for a given problem and returns the best-fit parameters.
    """
    print(f"Running estimation for problem with input signal of max amplitude {problem.u_meas.max():.2f}...")
    cost_func = create_cost_function(problem)
    
    # Use SciPy's optimizer to find the parameters that minimize the cost function
    fit_result = minimize(cost_func, theta_guess, method='Nelder-Mead', options={'disp': True})
    
    print(f"Estimation complete. Final SSE: {fit_result.fun:.4f}\n")
    return fit_result.x

def main():
    """
    Main function to design, run, and visualize the experiment.
    """
    # --- 1. Establish Ground Truth ---
    true_physio_params = {
        'p1': 0.028,  # Glucose effectiveness
        'p2': 0.025,  # Insulin sensitivity (glucose component)
        'p3': 1.3e-5, # Insulin sensitivity (action component)
        'n':  0.09,   # Insulin clearance rate
        'k':  0.03    # Rate of insulin action
    }
    # We will estimate all parameters except 'p1', which we assume is known.
    true_theta = jnp.array([
        true_physio_params['p2'], 
        true_physio_params['p3'], 
        true_physio_params['n'], 
        true_physio_params['k']
    ])
    
    def construct_model(theta: Array) -> SimpleGlucoseInsulinModel:
        """Helper function to build the model from the parameter vector theta."""
        params = {
            'p1': true_physio_params['p1'], # Assumed known
            'p2': theta[0], 'p3': theta[1], 'n': theta[2], 'k': theta[3]
        }
        # We can only measure glucose, making this a challenging estimation problem
        C_matrix = jnp.array([[1.0, 0.0, 0.0]])
        return SimpleGlucoseInsulinModel(params, C_matrix=C_matrix)

    true_model = construct_model(true_theta)

    # --- 2. Define Simulation and Data Generation Settings ---
    time_steps = jnp.linspace(0, 300, 151) # 5 hours of data
    x0 = jnp.array([0., 0., 0.]) # Start at baseline (no deviation)
    noise_level = 1.5 # Noise standard deviation in mg/dL
    key = jax.random.PRNGKey(42)

    # --- 3. Define Actuation Scenarios ---
    scenarios = {
        "Full Actuation (100%)": 1.0,
        "Medium Actuation (50%)": 0.5,
        "Low Actuation (10%)": 0.1,
        "Minimal Actuation (1%)": 0.01
    }
    estimation_results = {}

    # --- 4. Loop Through Scenarios, Generate Data, and Estimate Parameters ---
    for name, scale in scenarios.items():
        u_insulin = jnp.zeros_like(time_steps)
        u_insulin = u_insulin.at[(time_steps > 10) & (time_steps < 20)].set(20.0 * scale)
        u_exp = jnp.vstack([jnp.zeros_like(time_steps), u_insulin]).T # u_G is 0
        
        # Generate the "true" noise-free data for this input
        x_true = true_model.simulate(x0, time_steps, u=u_exp)
        y_true = true_model.observe(x_true, time_steps)
        
        # Add noise to create the synthetic experimental data
        noise = jax.random.normal(key, y_true.shape) * noise_level
        y_exp = y_true + noise
        
        # Create the model instance for the optimizer to use
        estimation_model = construct_model(true_theta) 
        
        # Set up the identification problem
        problem = IdentificationProblem(estimation_model, time_steps, x0, u_exp, y_exp)
        
        initial_guess = np.array([0.015, 2.0e-5, 0.15, 0.05])
        
        estimated_theta = run_parameter_estimation(problem, initial_guess)
        estimation_results[name] = estimated_theta

    param_names = ['p2', 'p3', 'n', 'k']
    n_params = len(param_names)
    bar_width = 0.15
    
    fig, axes = plt.subplots(n_params, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Parameter Estimation Performance vs. Actuation Constraint', fontsize=16)

    for i, p_name in enumerate(param_names):
        ax = axes[i]
        true_val = true_theta[i]
        
        index = np.arange(len(scenarios))
        
        # Get estimated values for this parameter across all scenarios
        estimated_vals = [estimation_results[name][i] for name in scenarios]
        
        ax.bar(index - bar_width/2, [true_val] * len(scenarios), bar_width, label='True Value', color='black', alpha=0.6)
        ax.bar(index + bar_width/2, estimated_vals, bar_width, label='Estimated Value', color='cornflowerblue')

        ax.set_ylabel(p_name)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.xticks(index, scenarios.keys(), rotation=15)
    plt.xlabel('Experimental Scenario (Input Strength)')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == '__main__':
    main()
