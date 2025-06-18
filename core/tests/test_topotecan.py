# file: identifiability/core/tests/test_topotecan.py

import pytest
import jax
import jax.numpy as jnp
from identifiability.core.models.pkpd.topotecan import TopotecanModel

# =============================================================================
# Test Fixtures and Constants
# =============================================================================

# Define a set of valid, base parameters for use in multiple tests
BASE_PARAMS = {
    "ko_m": 0.0289, "kc_m": 1.06e-4, "ki": 3.09e-4, "ke": 1.0140,
    "ko_c": 0.0265, "kc_c": 0.1863, "kb": 8.53e-4, "kd": 4.4489,
    "BT": 28.9, "v1": 100.0, "v2": 0.1
}

INITIAL_DOSE = 10.0

# =============================================================================
# Tests for TopotecanModel
# =============================================================================

class TestTopotecanModel:
    """
    Test suite for the specific implementation of the TopotecanModel.
    """

    # --- Initialization and Parameter Validation ---

    def test_initialization_and_dimensions(self):
        """Tests correct instantiation and dimension setting."""
        model = TopotecanModel(params=BASE_PARAMS, initial_dose=INITIAL_DOSE)
        assert model.n == 5, "Number of states (n) should be 5"
        assert model.p == 4, "Number of outputs (p) should be 4"
        assert model.m == 0, "Number of inputs (m) should be 0"

    def test_initial_state(self):
        """Tests that the initial state is set correctly."""
        model = TopotecanModel(params=BASE_PARAMS, initial_dose=INITIAL_DOSE)
        expected_initial_state = jnp.array([INITIAL_DOSE, 0.0, 0.0, 0.0, 0.0])
        assert jnp.allclose(model.initial_state, expected_initial_state)

    def test_parameter_validation(self):
        """Tests that parameter validation (non-negativity, presence) works."""
        # Test for missing key
        bad_params_missing = BASE_PARAMS.copy()
        del bad_params_missing["ki"]
        with pytest.raises(KeyError):
            TopotecanModel(params=bad_params_missing, initial_dose=INITIAL_DOSE)

        # Test for negative value
        bad_params_negative = BASE_PARAMS.copy()
        bad_params_negative["ki"] = -0.1
        with pytest.raises(AssertionError, match="must be ≥ 0"):
            TopotecanModel(params=bad_params_negative, initial_dose=INITIAL_DOSE)

    # --- f (Dynamics) Method Tests ---

    def test_f_dimensionality(self):
        """Ensures the dynamics function f returns an array of the correct shape."""
        model = TopotecanModel(params=BASE_PARAMS, initial_dose=INITIAL_DOSE)
        f_out = model.f(model.initial_state, u=jnp.array([]), params=model.params, t=0.0)
        assert f_out.shape == (model.n,)
    
    def test_f_qualitative_behavior_and_non_negativity(self):
        """
        Tests for biologically plausible behavior and ensures concentrations remain non-negative.
        """
        model = TopotecanModel(params=BASE_PARAMS, initial_dose=INITIAL_DOSE)
        time_steps = jnp.linspace(0, 50, 100)
        trajectory = model.simulate(model.initial_state, time_steps, u=None)

        # 1. Test for non-negativity
        assert jnp.all(trajectory >= 0), "Concentrations should never be negative"

        # 2. Test qualitative behavior (e.g., drug flows from medium to cells)
        # Initially, Lm should decrease, and cellular components (Lc, Ln) should increase.
        assert trajectory[1, 0] < trajectory[0, 0] # Lm decreases
        assert trajectory[1, 2] > trajectory[0, 2] # Lc increases
        assert trajectory[1, 4] > trajectory[0, 4] # Ln increases

    def test_no_cell_influx_scenario(self):
        """
        Tests that if ki=0, no drug appears in cellular compartments.
        """
        params_no_influx = BASE_PARAMS.copy()
        params_no_influx["ki"] = 0.0
        model = TopotecanModel(params=params_no_influx, initial_dose=INITIAL_DOSE)
        
        time_steps = jnp.linspace(0, 100, 50)
        trajectory = model.simulate(model.initial_state, time_steps, u=None)
        
        # Lc (idx 2), Hc (idx 3), and Ln (idx 4) should remain zero.
        assert jnp.allclose(trajectory[:, 2:], 0.0, atol=1e-9)

    # --- g (Observation) Method Tests ---
    
    def test_g_dimensionality(self):
        """Ensures the observation function g returns an array of the correct shape."""
        model = TopotecanModel(params=BASE_PARAMS, initial_dose=INITIAL_DOSE)
        g_out = model.g(model.initial_state, params=model.params, t=0.0)
        assert g_out.shape == (model.p,)

    def test_g_logical_consistency(self):
        """Verifies the weighted average logic of the observation function."""
        model = TopotecanModel(params=BASE_PARAMS, initial_dose=INITIAL_DOSE)
        v2 = model.params[-1] # v2 is the last parameter

        # Case 1: All drug is in cytoplasm (Lc)
        state1 = jnp.array([0.0, 0.0, 10.0, 0.0, 0.0])
        obs1 = model.g(state1, model.params, t=0.0)
        expected_li_1 = 10.0 / (1 + v2)
        assert jnp.allclose(obs1[2], expected_li_1)
        assert jnp.allclose(obs1[3], 0.0) # Hi

        # Case 2: All drug is in nucleus (Ln)
        state2 = jnp.array([0.0, 0.0, 0.0, 0.0, 20.0])
        obs2 = model.g(state2, model.params, t=0.0)
        expected_li_2 = (v2 * 20.0) / (1 + v2)
        assert jnp.allclose(obs2[2], expected_li_2)
        
    # --- End-to-End Simulation Tests ---
    
    def test_mass_conservation(self):
        """
        Tests the principle of drug mass conservation across compartments,
        accounting for volumes. Total drug amount should be constant.
        """
        model = TopotecanModel(params=BASE_PARAMS, initial_dose=INITIAL_DOSE)
        v1 = BASE_PARAMS["v1"]
        v2 = BASE_PARAMS["v2"]

        # Total mass = (Lm + Hm)*v_medium + (Lc + Hc)*v_cyto + Ln*v_nucleus
        # Assuming v_cyto=1, then v_medium=v1 and v_nucleus=v2
        def get_total_mass(state):
            Lm, Hm, Lc, Hc, Ln = state
            return (Lm + Hm) * v1 + (Lc + Hc) + Ln * v2

        time_steps = jnp.linspace(0, 100, 1001) # dt = 0.1
        trajectory = model.simulate(model.initial_state, time_steps, u=None)
        
        # Ensure no NaNs were produced
        assert not jnp.isnan(trajectory).any(), "Simulation produced NaN values, indicating instability."
        
        total_mass_over_time = jax.vmap(get_total_mass)(trajectory)
        
        initial_mass = get_total_mass(model.initial_state)
        assert jnp.allclose(total_mass_over_time, initial_mass, atol=1e-5)
