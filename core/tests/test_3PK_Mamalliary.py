import pytest
import jax.numpy as jnp

from identifiability.core.models.pkpd.threePK_Mamalliary import threePKModel

@pytest.fixture
def propofol_params():
    """Provides a set of typical, realistic parameters for the Propofol model."""
    # These are illustrative values
    return {
        'k10': 0.118,
        'k12': 0.352,
        'k13': 0.108,
        'k21': 0.189,
        'k31': 0.005,
        'V1': 15.9
    }

class TestPropofolPKModel:

    def test_instantiation_and_matrices(self, propofol_params):
        """
        Tests that the model is created correctly and that its A, B, C
        matrices have the correct shapes and values.
        """
        model = threePKModel(propofol_params)

        # Check dimensions inherited from LinearSystem
        assert model.n == 3
        assert model.m == 1
        assert model.p == 1

        # Check a few values in the constructed A matrix
        k10, k12, k13 = propofol_params['k10'], propofol_params['k12'], propofol_params['k13']
        k21 = propofol_params['k21']
        assert jnp.isclose(model.A[0, 0], -(k10 + k12 + k13))
        assert jnp.isclose(model.A[1, 0], k12)
        assert jnp.isclose(model.A[0, 1], k21)
        
        # Check the C matrix
        assert jnp.isclose(model.C[0, 0], 1.0 / propofol_params['V1'])

    def test_inherited_analysis_methods(self, propofol_params):
        """
        Tests that the Propofol model correctly uses the analysis methods
        inherited from the LinearSystem base class.
        """
        model = threePKModel(propofol_params)

        # The standard Propofol model should be both controllable and observable
        assert model.is_controllable(method='kalman')
        assert model.is_controllable(method='pbh')
        assert model.is_observable(method='kalman')
        assert model.is_observable(method='pbh')

    def test_uncontrollable_scenario(self, propofol_params):
        """
        Tests that a modified, uncontrollable version of the model is
        correctly identified as uncontrollable.
        """
        # Simulate a scenario where the drug cannot enter the slow compartment (k13=0)
        uncontrollable_params = propofol_params.copy()
        uncontrollable_params['k13'] = 0.0
        
        model = threePKModel(uncontrollable_params)
        
        # The system is no longer controllable because state x3 is disconnected
        # from the input. However, since k31 != 0, it is still observable.
        assert not model.is_controllable()
        assert model.is_observable()

