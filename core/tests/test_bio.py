# file: identifiability/core/tests/test_biological_models.py

import pytest
import jax.numpy as jnp
from identifiability.core.models.bio.goodwin_oscillator import GoodwinOscillator
from identifiability.core.models.bio.hiv_model import HIVModel


class TestBiologicalModels:

    def test_goodwin_oscillator(self):
        """Tests instantiation and simulation of the Goodwin Oscillator."""
        params = {
            "alpha1": 1, "alpha2": 1, "alpha3": 1,
            "beta1": 1, "beta2": 1, "beta3": 1, "n": 4
        }
        initial_state = jnp.array([0.1, 0.2, 0.3])
        model = GoodwinOscillator(params, initial_state)
        
        time_steps = jnp.linspace(0, 1, 10)
        trajectory = model.simulate(model.initial_state, time_steps)
        
        assert trajectory.shape == (10, 3)
        observations = model.observe(trajectory, time_steps)
        assert observations.shape == (10, 1)

    def test_hiv_model(self):
        """Tests instantiation and simulation of the HIV Dynamics model."""
        params = {
            "s": 10, "d": 0.01, "beta": 2.4e-5,
            "delta": 0.24, "p": 20, "c": 2.4
        }
        initial_state = jnp.array([1e6, 0, 1e-3])
        model = HIVModel(params, initial_state)
        
        time_steps = jnp.linspace(0, 1, 10)
        trajectory = model.simulate(model.initial_state, time_steps)
        
        assert trajectory.shape == (10, 3)
        observations = model.observe(trajectory, time_steps)
        assert observations.shape == (10, 1)

