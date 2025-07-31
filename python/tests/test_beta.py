"""
Unit tests for CPU beta coefficient estimation functions.
"""

import pytest
import numpy as np
from scipy import linalg
from unittest.mock import patch

from devil.beta import init_beta, fit_beta_coefficients


class TestInitBeta:
    """Test beta coefficient initialization."""
    
    @pytest.fixture
    def test_data(self):
        """Create test data for beta initialization."""
        np.random.seed(42)
        n_genes, n_samples, n_features = 20, 15, 3
        
        # Create count matrix
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        
        # Create design matrix
        design = np.column_stack([
            np.ones(n_samples),  # intercept
            np.random.binomial(1, 0.5, n_samples),  # binary condition
            np.random.normal(0, 1, n_samples)  # continuous covariate
        ])
        
        # Create offset vector
        offset = np.random.normal(0, 0.1, n_samples)
        
        return counts, design, offset
    
    def test_init_beta_basic(self, test_data):
        """Test basic beta initialization."""
        counts, design, offset = test_data
        
        beta_init = init_beta(counts, design, offset)
        
        # Check dimensions
        assert beta_init.shape == (20, 3)
        
        # Check that values are finite
        assert np.all(np.isfinite(beta_init))
        
        # Intercept terms should generally be positive (log scale)
        intercept_mean = np.mean(beta_init[:, 0])
        assert intercept_mean > 0, "Intercept should be positive on log scale"
        
    def test_init_beta_single_gene(self, test_data):
        """Test beta initialization for single gene."""
        counts, design, offset = test_data
        single_gene_counts = counts[0:1, :]  # Take first gene only
        
        beta_init = init_beta(single_gene_counts, design, offset)
        
        assert beta_init.shape == (1, 3)
        assert np.all(np.isfinite(beta_init))
        
    def test_init_beta_minimal_design(self):
        """Test beta initialization with minimal design (intercept only)."""
        np.random.seed(42)
        counts = np.random.negative_binomial(5, 0.3, size=(10, 12))
        design = np.ones((12, 1))  # Intercept only
        offset = np.zeros(12)
        
        beta_init = init_beta(counts, design, offset)
        
        assert beta_init.shape == (10, 1)
        assert np.all(np.isfinite(beta_init))
        
    def test_init_beta_zero_counts(self):
        """Test beta initialization with some zero counts."""
        np.random.seed(42)
        counts = np.random.negative_binomial(2, 0.8, size=(10, 12))  # Many zeros expected
        design = np.ones((12, 1))
        offset = np.zeros(12)
        
        beta_init = init_beta(counts, design, offset)
        
        assert beta_init.shape == (10, 1)
        assert np.all(np.isfinite(beta_init))
        # log1p should handle zeros gracefully
        
    def test_init_beta_qr_decomposition(self, test_data):
        """Test that QR decomposition is used correctly."""
        counts, design, offset = test_data
        
        # Manually perform QR decomposition to compare.  Use the economic
        # decomposition so that `R` is square and matches the implementation in
        # :func:`init_beta` which relies on a reduced QR factorisation.
        Q, R = linalg.qr(design, mode="economic")
        norm_log_counts = np.log1p((counts.T / np.exp(offset)[:, np.newaxis]))
        expected_beta = linalg.solve_triangular(R, Q.T @ norm_log_counts, lower=False).T
        
        beta_init = init_beta(counts, design, offset)
        
        # Should be approximately equal
        np.testing.assert_allclose(beta_init, expected_beta, rtol=1e-10)


class TestFitBetaCoefficients:
    """Test beta coefficient fitting for single genes."""
    
    @pytest.fixture
    def single_gene_data(self):
        """Create data for single gene fitting."""
        np.random.seed(42)
        n_samples, n_features = 20, 3
        
        # True beta coefficients
        true_beta = np.array([2.0, 1.5, -0.5])  # intercept, condition, covariate
        
        # Design matrix
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples),
            np.random.normal(0, 1, n_samples)
        ])
        
        # Generate counts based on true model
        offset = np.random.normal(0, 0.1, n_samples)
        mu = np.exp(design @ true_beta + offset)
        dispersion = 0.1
        counts = np.random.negative_binomial(n=1/dispersion, p=1/(1 + mu * dispersion))
        
        # Initial beta estimate
        beta_init = np.random.normal(0, 0.1, n_features) + true_beta
        
        return counts, design, beta_init, offset, dispersion, true_beta
    
    def test_fit_beta_basic(self, single_gene_data):
        """Test basic beta coefficient fitting."""
        counts, design, beta_init, offset, dispersion, true_beta = single_gene_data
        
        fitted_beta, n_iter, converged = fit_beta_coefficients(
            counts, design, beta_init, offset, dispersion,
            max_iter=50, tolerance=1e-4
        )
        
        # Check basic properties
        assert len(fitted_beta) == 3
        assert isinstance(n_iter, int)
        assert isinstance(converged, bool)
        assert n_iter >= 1
        assert np.all(np.isfinite(fitted_beta))
        
        # Should converge for reasonable data
        assert converged, "Should converge with reasonable parameters"
        
        # Should be reasonably close to true values (within 2 standard errors)
        # This is a stochastic test, so we use loose bounds
        for i in range(3):
            assert abs(fitted_beta[i] - true_beta[i]) < 2.0
    
    def test_fit_beta_convergence_strict(self, single_gene_data):
        """Test convergence with strict tolerance."""
        counts, design, beta_init, offset, dispersion, _ = single_gene_data
        
        fitted_beta, n_iter, converged = fit_beta_coefficients(
            counts, design, beta_init, offset, dispersion,
            max_iter=100, tolerance=1e-8
        )
        
        if converged:
            # If converged, should be very precise
            assert n_iter <= 100
        else:
            # If not converged, should have used all iterations
            assert n_iter == 100
    
    def test_fit_beta_convergence_loose(self, single_gene_data):
        """Test convergence with loose tolerance."""
        counts, design, beta_init, offset, dispersion, _ = single_gene_data
        
        fitted_beta, n_iter, converged = fit_beta_coefficients(
            counts, design, beta_init, offset, dispersion,
            max_iter=50, tolerance=1e-1  # Very loose
        )
        
        # Should converge quickly with loose tolerance
        assert converged
        assert n_iter < 20  # Should converge fast
    
    def test_fit_beta_zero_dispersion(self, single_gene_data):
        """Test fitting with zero dispersion (Poisson model)."""
        counts, design, beta_init, offset, _, _ = single_gene_data
        
        fitted_beta, n_iter, converged = fit_beta_coefficients(
            counts, design, beta_init, offset, dispersion=0.0,
            max_iter=50, tolerance=1e-4
        )
        
        # Should handle Poisson case (dispersion=0)
        assert len(fitted_beta) == 3
        assert np.all(np.isfinite(fitted_beta))
        
    def test_fit_beta_high_dispersion(self, single_gene_data):
        """Test fitting with high dispersion."""
        counts, design, beta_init, offset, _, _ = single_gene_data
        
        fitted_beta, n_iter, converged = fit_beta_coefficients(
            counts, design, beta_init, offset, dispersion=100.0,  # Very high
            max_iter=50, tolerance=1e-4
        )
        
        # Should handle high dispersion
        assert len(fitted_beta) == 3
        assert np.all(np.isfinite(fitted_beta))
    
    def test_fit_beta_all_zero_counts(self):
        """Test fitting with all zero counts."""
        n_samples = 15
        counts = np.zeros(n_samples)
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        beta_init = np.array([0.0, 0.0])
        offset = np.zeros(n_samples)
        
        fitted_beta, n_iter, converged = fit_beta_coefficients(
            counts, design, beta_init, offset, dispersion=1.0,
            max_iter=10, tolerance=1e-4
        )
        
        # Should handle all-zero case
        assert len(fitted_beta) == 2
        assert np.all(np.isfinite(fitted_beta))
        # Should return zeros or very small values
        assert np.all(np.abs(fitted_beta) < 10)
    
    def test_fit_beta_singular_design(self):
        """Test behavior with near-singular design matrix."""
        np.random.seed(42)
        n_samples = 20
        counts = np.random.negative_binomial(5, 0.3, n_samples)
        
        # Create nearly singular design matrix
        design = np.column_stack([
            np.ones(n_samples),
            np.random.normal(0, 1, n_samples),
            np.random.normal(0, 1, n_samples) * 1e-10  # Nearly zero column
        ])
        
        beta_init = np.array([1.0, 0.5, 0.1])
        offset = np.zeros(n_samples)
        
        # Should handle near-singularity gracefully
        fitted_beta, n_iter, converged = fit_beta_coefficients(
            counts, design, beta_init, offset, dispersion=1.0,
            max_iter=20, tolerance=1e-4
        )
        
        assert len(fitted_beta) == 3
        assert np.all(np.isfinite(fitted_beta))
    
    def test_fit_beta_extreme_initial_values(self, single_gene_data):
        """Test fitting with extreme initial values."""
        counts, design, _, offset, dispersion, _ = single_gene_data
        
        # Very extreme initial values
        beta_init = np.array([100.0, -100.0, 50.0])
        
        fitted_beta, n_iter, converged = fit_beta_coefficients(
            counts, design, beta_init, offset, dispersion,
            max_iter=100, tolerance=1e-4
        )
        
        # Should still produce reasonable results
        assert len(fitted_beta) == 3
        assert np.all(np.isfinite(fitted_beta))
        # Should move towards reasonable values
        assert np.all(np.abs(fitted_beta) < 50)


class TestBetaEdgeCases:
    """Test edge cases in beta coefficient estimation."""
    
    def test_single_sample(self):
        """Test beta fitting with single sample (should not work well)."""
        counts = np.array([5])
        design = np.array([[1.0, 1.0]])  # Single sample, two parameters
        beta_init = np.array([1.0, 0.5])
        offset = np.array([0.0])
        
        # This is underdetermined, but should not crash
        fitted_beta, n_iter, converged = fit_beta_coefficients(
            counts, design, beta_init, offset, dispersion=1.0,
            max_iter=10, tolerance=1e-4
        )
        
        assert len(fitted_beta) == 2
        assert np.all(np.isfinite(fitted_beta))
        
    def test_no_variation_in_design(self):
        """Test with design matrix having no variation in some columns."""
        np.random.seed(42)
        n_samples = 20
        counts = np.random.negative_binomial(5, 0.3, n_samples)
        
        design = np.column_stack([
            np.ones(n_samples),
            np.ones(n_samples) * 2,  # Constant (but not 1)
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        beta_init = np.array([1.0, 0.0, 0.5])
        offset = np.zeros(n_samples)
        
        # Should handle constant columns
        fitted_beta, n_iter, converged = fit_beta_coefficients(
            counts, design, beta_init, offset, dispersion=1.0,
            max_iter=20, tolerance=1e-4
        )
        
        assert len(fitted_beta) == 3
        assert np.all(np.isfinite(fitted_beta))
    
    def test_very_sparse_counts(self):
        """Test with very sparse count data (many zeros)."""
        np.random.seed(42)
        n_samples = 50
        # Generate very sparse data
        counts = np.random.negative_binomial(1, 0.9, n_samples)  # Many zeros
        
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        beta_init = np.array([0.0, 0.0])
        offset = np.zeros(n_samples)
        
        fitted_beta, n_iter, converged = fit_beta_coefficients(
            counts, design, beta_init, offset, dispersion=1.0,
            max_iter=50, tolerance=1e-4
        )
        
        assert len(fitted_beta) == 2
        assert np.all(np.isfinite(fitted_beta))
        
    def test_outlier_counts(self):
        """Test with outlier count values."""
        np.random.seed(42)
        n_samples = 30
        counts = np.random.negative_binomial(3, 0.5, n_samples)
        counts[0] = 1000  # Add outlier
        
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        beta_init = np.array([1.0, 0.0])
        offset = np.zeros(n_samples)
        
        fitted_beta, n_iter, converged = fit_beta_coefficients(
            counts, design, beta_init, offset, dispersion=1.0,
            max_iter=50, tolerance=1e-4
        )
        
        # Should handle outliers without crashing
        assert len(fitted_beta) == 2
        assert np.all(np.isfinite(fitted_beta))


class TestBetaNumericalStability:
    """Test numerical stability of beta fitting."""
    
    def test_large_offset_values(self):
        """Test stability with large offset values."""
        np.random.seed(42)
        n_samples = 20
        counts = np.random.negative_binomial(5, 0.3, n_samples)
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        beta_init = np.array([1.0, 0.5])
        offset = np.random.normal(10, 1, n_samples)  # Large offsets
        
        fitted_beta, n_iter, converged = fit_beta_coefficients(
            counts, design, beta_init, offset, dispersion=1.0,
            max_iter=50, tolerance=1e-4
        )
        
        assert len(fitted_beta) == 2
        assert np.all(np.isfinite(fitted_beta))
        
    def test_small_dispersion_values(self):
        """Test stability with very small dispersion."""
        np.random.seed(42)
        n_samples = 20
        counts = np.random.negative_binomial(5, 0.3, n_samples)
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        beta_init = np.array([1.0, 0.5])
        offset = np.zeros(n_samples)
        
        fitted_beta, n_iter, converged = fit_beta_coefficients(
            counts, design, beta_init, offset, dispersion=1e-8,  # Very small
            max_iter=50, tolerance=1e-4
        )
        
        assert len(fitted_beta) == 2
        assert np.all(np.isfinite(fitted_beta))
        
    def test_numerical_precision_consistency(self):
        """Test that results are consistent across runs with same seed."""
        np.random.seed(42)
        n_samples = 25
        counts = np.random.negative_binomial(5, 0.3, n_samples)
        design = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        beta_init = np.array([1.0, 0.5])
        offset = np.zeros(n_samples)
        
        # Fit twice with same parameters
        fitted_beta1, _, _ = fit_beta_coefficients(
            counts, design, beta_init, offset, dispersion=1.0,
            max_iter=50, tolerance=1e-6
        )
        
        fitted_beta2, _, _ = fit_beta_coefficients(
            counts, design, beta_init, offset, dispersion=1.0,
            max_iter=50, tolerance=1e-6
        )
        
        # Results should be identical
        np.testing.assert_allclose(fitted_beta1, fitted_beta2, rtol=1e-12)
