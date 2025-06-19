import pytest
import jax.numpy as jnp

from identifiability.core.algs.QiuICIS import QiuICIS
from identifiability.core.models.linearCT import LinearSystem

# --- Test Fixtures ---

@pytest.fixture
def lti_system_distinct_eigs():
    """A simple 2D LTI system with distinct, real eigenvalues."""
    # Eigenvalues are -1 and -2. Eigenvectors are [1, 0] and [0, 1].
    A = jnp.array([[-1.0, 0.0], [0.0, -2.0]])
    # B and C are not used in this analysis but are required for instantiation.
    B = jnp.zeros((2, 1))
    C = jnp.zeros((1, 2))
    return LinearSystem(A, B, C)

@pytest.fixture
def lti_system_repeated_eigs_diagonalizable():
    """A diagonalizable system with a repeated eigenvalue of -1."""
    A = jnp.array([[-1.0, 0.0], [0.0, -1.0]])
    return LinearSystem(A, jnp.zeros((2,1)), jnp.zeros((1,2)))

@pytest.fixture
def lti_system_defective():
    """A non-diagonalizable (defective) system with a repeated eigenvalue of -1."""
    A = jnp.array([[-1.0, 1.0], [0.0, -1.0]])
    return LinearSystem(A, jnp.zeros((2,1)), jnp.zeros((1,2)))

class TestQiuICIS:

    def test_identifiable_case(self, lti_system_distinct_eigs):
        """
        Tests a system that should be identifiable because the initial condition
        excites all dynamic modes (has components along all eigenvectors).
        """
        model = lti_system_distinct_eigs
        # This initial condition has components in both the [1,0] and [0,1] eigenvector directions.
        x0 = jnp.array([1.0, 1.0])
        
        analyzer = QiuICIS()
        result = analyzer.analyze(model, x0)
        
        assert result["identifiable"]
        assert result["status"] == "Success"
        assert result["score"] > 1e-9

    def test_unidentifiable_case(self, lti_system_distinct_eigs):
        """
        Tests a system that should be unidentifiable because the initial
        condition lies along a single eigenvector, leaving the other mode unexcited.
        """
        model = lti_system_distinct_eigs
        # This initial condition only has a component in the [1,0] direction.
        # The mode corresponding to eigenvalue -2 will not appear in the trajectory.
        x0 = jnp.array([1.0, 0.0])
        
        analyzer = QiuICIS()
        result = analyzer.analyze(model, x0)

        assert not result["identifiable"]
        assert result["status"] == "Success"
        assert result["score"] < 1e-9
        
    def test_non_distinct_eigenvalues(self):
        """
        Tests that the analyzer correctly handles the case of non-distinct eigenvalues.
        """
        # This matrix has a repeated eigenvalue of -1.
        A = jnp.array([[-1., 1.], [0., -1.]])
        B = jnp.zeros((2, 1))
        C = jnp.zeros((1, 2))
        model = LinearSystem(A, B, C)
        x0 = jnp.array([1.0, 1.0])

        analyzer = QiuICIS()
        result = analyzer.analyze(model, x0)
        
        assert not result["identifiable"]
        assert result["status"] == "Failed"


    def test_identifiable_distinct_eigs(self, lti_system_distinct_eigs):
        """Tests an identifiable case with distinct eigenvalues."""
        x0 = jnp.array([1.0, 1.0])
        analyzer = QiuICIS()
        result = analyzer.analyze(lti_system_distinct_eigs, x0)
        
        assert result["identifiable"]
        assert result["status"] == "Success"

    def test_unidentifiable_distinct_eigs(self, lti_system_distinct_eigs):
        """Tests an unidentifiable case where one mode is not excited."""
        x0 = jnp.array([1.0, 0.0])
        analyzer = QiuICIS()
        result = analyzer.analyze(lti_system_distinct_eigs, x0,)

        assert not result["identifiable"]
        assert result["status"] == "Success"

    def test_identifiable_repeated_eigs(self, lti_system_repeated_eigs_diagonalizable):
        """Tests an identifiable case with repeated but diagonalizable eigenvalues."""
        x0 = jnp.array([1.0, 1.0])
        analyzer = QiuICIS()
        result = analyzer.analyze(lti_system_repeated_eigs_diagonalizable, x0)
        
        assert result["identifiable"]
        assert result["status"] == "Success"
        
    def test_defective_matrix_case(self, lti_system_defective):
        """Tests that a defective (non-diagonalizable) matrix is correctly flagged."""
        x0 = jnp.array([1.0, 1.0])
        analyzer = QiuICIS()
        result = analyzer.analyze(lti_system_defective, x0)

        assert not result["identifiable"]
        assert result["status"] == "Failed"

