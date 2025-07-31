"""
Unit tests for CPU overdispersion estimation functions.
"""

import pytest
import numpy as np
from scipy import optimize
from unittest.mock import patch, MagicMock

from devil.overdispersion import (
    estimate_initial_dispersion, fit_dispersion,
    compute_nb_log_likelihood, compute_nb_score, compute_nb_hessian
)


class TestEstimateInitialDispersion:
    """Test initial dispersion estimation using method of moments."""
    
    @pytest.fixture
    def test_data(self):
        """Create test data for dispersion estimation."""
        np.random.seed(42)
        n_genes, n_samples = 50, 30
        
        # Create count matrix with known overdispersion
        true_dispersion = np.random.gamma(2, 0.5, n_genes)  # True dispersion values
        counts = np.zeros((n_genes, n_samples))
        
        for i in range(n_genes):
            mu = np.random.gamma(10, 1, n_samples)  # Mean counts
            # Generate NB counts with known dispersion
            p = 1 / (1 + mu * true_dispersion[i])
            n = 1 / true_dispersion[i]
            counts[i, :] = np.random.negative_binomial(n, p)
        
        # Create offset vector
        offset_vector = np.random.normal(0, 0.1, n_samples)
        
        return counts, offset_vector, true_dispersion
    
    def test_estimate_initial_dispersion_basic(self, test_data):
        """Test basic initial dispersion estimation."""
        counts, offset_vector, true_dispersion = test_data
        
        estimated_dispersion = estimate_initial_dispersion(counts, offset_vector)
        
        # Check basic properties
        assert len(estimated_dispersion) == 50
        assert np.all(estimated_dispersion > 0)  # Should be positive
        assert np.all(np.isfinite(estimated_dispersion))
        
        # Should be in reasonable range (not too extreme)
        assert np.all(estimated_dispersion < 1000)
        assert np.all(estimated_dispersion > 1e-6)
    
    def test_estimate_initial_dispersion_edge_cases(self):
        """Test initial dispersion estimation with edge cases."""
        n_genes, n_samples = 10, 20
        
        # Case 1: All zero counts
        zero_counts = np.zeros((n_genes, n_samples))
        offset_vector = np.zeros(n_samples)
        
        dispersion = estimate_initial_dispersion(zero_counts, offset_vector)
        assert len(dispersion) == n_genes
        assert np.all(dispersion == 100.0)  # Default high value
        
        # Case 2: Very low variance (underdispersed)
        constant_counts = np.ones((n_genes, n_samples)) * 5  # Same count everywhere
        dispersion = estimate_initial_dispersion(constant_counts, offset_vector)
        assert np.all(dispersion == 100.0)  # Should default to high value
        
        # Case 3: Very high variance (overdispersed)
        high_var_counts = np.random.negative_binomial(1, 0.1, size=(n_genes, n_samples))
        dispersion = estimate_initial_dispersion(high_var_counts, offset_vector)
        assert np.all(dispersion > 0)
        assert np.all(np.isfinite(dispersion))
    
    def test_estimate_initial_dispersion_single_gene(self):
        """Test initial dispersion estimation for single gene."""
        np.random.seed(42)
        n_samples = 25
        counts = np.random.negative_binomial(5, 0.3, size=(1, n_samples))
        offset_vector = np.zeros(n_samples)
        
        dispersion = estimate_initial_dispersion(counts, offset_vector)
        
        assert len(dispersion) == 1
        assert dispersion[0] > 0
        assert np.isfinite(dispersion[0])
    
    def test_estimate_initial_dispersion_offset_effects(self):
        """Test that offset vector affects dispersion estimation."""
        np.random.seed(42)
        n_genes, n_samples = 20, 30
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        
        # Test with different offset vectors
        offset1 = np.zeros(n_samples)
        offset2 = np.random.normal(1, 0.5, n_samples)  # Different offsets
        
        disp1 = estimate_initial_dispersion(counts, offset1)
        disp2 = estimate_initial_dispersion(counts, offset2)
        
        # Results should be different due to different normalization
        assert not np.allclose(disp1, disp2)


class TestFitDispersion:
    """Test dispersion parameter fitting."""
    
    @pytest.fixture
    def single_gene_data(self):
        """Create data for single gene dispersion fitting."""
        np.random.seed(42)
        n_samples, n_features = 25, 3
        
        # Design matrix
        design_matrix = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples),
            np.random.normal(0, 1, n_samples)
        ])
        
        # True parameters
        true_beta = np.array([2.0, 1.0, -0.5])
        true_dispersion = 0.5
        
        # Generate data
        offset_vector = np.random.normal(0, 0.1, n_samples)
        mu = np.exp(design_matrix @ true_beta + offset_vector)
        p = 1 / (1 + mu * true_dispersion)
        n = 1 / true_dispersion
        y = np.random.negative_binomial(n, p)
        
        return y, design_matrix, true_beta, offset_vector, true_dispersion
    
    def test_fit_dispersion_basic(self, single_gene_data):
        """Test basic dispersion fitting."""
        y, design_matrix, beta, offset_vector, true_dispersion = single_gene_data
        
        estimated_dispersion = fit_dispersion(
            beta, design_matrix, y, offset_vector,
            tolerance=1e-3, max_iter=50, do_cox_reid_adjustment=True
        )
        
        # Check basic properties
        assert isinstance(estimated_dispersion, float)
        assert estimated_dispersion > 0
        assert np.isfinite(estimated_dispersion)
        
        # Should be reasonably close to true value (within factor of 2-3)
        ratio = estimated_dispersion / true_dispersion
        assert 0.3 < ratio < 3.0, f"Dispersion ratio {ratio} outside reasonable range"
    
    def test_fit_dispersion_all_zero_counts(self):
        """Test dispersion fitting with all zero counts."""
        n_samples = 20
        y = np.zeros(n_samples)
        design_matrix = np.ones((n_samples, 1))
        beta = np.array([0.0])
        offset_vector = np.zeros(n_samples)
        
        dispersion = fit_dispersion(
            beta, design_matrix, y, offset_vector,
            tolerance=1e-3, max_iter=50
        )
        
        # Should return 0 for all-zero genes
        assert dispersion == 0.0
    
    def test_fit_dispersion_no_cox_reid(self, single_gene_data):
        """Test dispersion fitting without Cox-Reid adjustment."""
        y, design_matrix, beta, offset_vector, _ = single_gene_data
        
        dispersion = fit_dispersion(
            beta, design_matrix, y, offset_vector,
            tolerance=1e-3, max_iter=50, do_cox_reid_adjustment=False
        )
        
        assert isinstance(dispersion, float)
        assert dispersion > 0
        assert np.isfinite(dispersion)
    
    def test_fit_dispersion_convergence_failure(self, single_gene_data):
        """Test behavior when dispersion fitting fails to converge."""
        y, design_matrix, beta, offset_vector, _ = single_gene_data
        
        # Use very strict tolerance and few iterations to force failure
        dispersion = fit_dispersion(
            beta, design_matrix, y, offset_vector,
            tolerance=1e-10, max_iter=2, do_cox_reid_adjustment=True
        )
        
        # Should still return a reasonable value (initial estimate)
        assert isinstance(dispersion, float)
        assert dispersion > 0
        assert np.isfinite(dispersion)
    
    def test_fit_dispersion_extreme_data(self):
        """Test dispersion fitting with extreme count data."""
        np.random.seed(42)
        n_samples = 30
        design_matrix = np.ones((n_samples, 1))
        beta = np.array([5.0])  # High intercept
        offset_vector = np.zeros(n_samples)
        
        # Generate very high counts
        y = np.random.negative_binomial(10, 0.1, n_samples)  # High variance
        
        dispersion = fit_dispersion(
            beta, design_matrix, y, offset_vector,
            tolerance=1e-3, max_iter=100
        )
        
        assert isinstance(dispersion, float)
        assert dispersion > 0
        assert np.isfinite(dispersion)
    
    def test_fit_dispersion_optimization_fallback(self, single_gene_data):
        """Test that optimization fallback works."""
        y, design_matrix, beta, offset_vector, _ = single_gene_data
        
        # Mock the first optimization to fail
        with patch('devil.overdispersion.optimize.minimize') as mock_minimize:
            # First call fails, second succeeds
            mock_minimize.side_effect = [
                MagicMock(success=False),  # First attempt fails
                MagicMock(success=True, x=[0.5])  # Second attempt succeeds
            ]
            
            dispersion = fit_dispersion(
                beta, design_matrix, y, offset_vector,
                tolerance=1e-3, max_iter=50
            )
            
            # Should have tried twice
            assert mock_minimize.call_count == 2
            assert dispersion > 0


class TestNegativeBinomialFunctions:
    """Test negative binomial likelihood, score, and hessian functions."""
    
    @pytest.fixture
    def nb_test_data(self):
        """Create test data for NB function testing."""
        np.random.seed(42)
        n_samples = 20
        
        y = np.random.negative_binomial(5, 0.3, n_samples)
        mu = np.random.gamma(5, 1, n_samples)
        theta = 0.5
        design_matrix = np.ones((n_samples, 1))
        
        return y, mu, theta, design_matrix
    
    def test_compute_nb_log_likelihood(self, nb_test_data):
        """Test negative binomial log-likelihood computation."""
        y, mu, theta, design_matrix = nb_test_data
        
        ll = compute_nb_log_likelihood(y, mu, theta, design_matrix, do_cox_reid=False)
        
        assert isinstance(ll, float)
        assert np.isfinite(ll)
        # Log-likelihood should be negative for most real data
        assert ll < 100  # Should not be extremely positive
    
    def test_compute_nb_log_likelihood_cox_reid(self, nb_test_data):
        """Test NB log-likelihood with Cox-Reid adjustment."""
        y, mu, theta, design_matrix = nb_test_data
        
        ll_cr = compute_nb_log_likelihood(y, mu, theta, design_matrix, do_cox_reid=True)
        ll_no_cr = compute_nb_log_likelihood(y, mu, theta, design_matrix, do_cox_reid=False)
        
        assert isinstance(ll_cr, float)
        assert isinstance(ll_no_cr, float)
        assert np.isfinite(ll_cr)
        assert np.isfinite(ll_no_cr)
        
        # Cox-Reid adjustment should change the likelihood
        assert ll_cr != ll_no_cr
    
    def test_compute_nb_score(self, nb_test_data):
        """Test negative binomial score function."""
        y, mu, theta, design_matrix = nb_test_data
        
        score = compute_nb_score(y, mu, theta, design_matrix, do_cox_reid=False)
        
        assert isinstance(score, float)
        assert np.isfinite(score)
    
    def test_compute_nb_hessian(self, nb_test_data):
        """Test negative binomial hessian computation."""
        y, mu, theta, design_matrix = nb_test_data
        
        hessian = compute_nb_hessian(y, mu, theta, design_matrix, do_cox_reid=False)
        
        assert isinstance(hessian, float)
        assert np.isfinite(hessian)
        # Hessian should typically be negative (concave likelihood)
    
    def test_nb_functions_consistency(self, nb_test_data):
        """Test consistency between NB functions (numerical derivatives)."""
        y, mu, theta, design_matrix = nb_test_data
        
        # Test that numerical derivative of likelihood matches score
        epsilon = 1e-6
        theta1 = theta - epsilon
        theta2 = theta + epsilon
        
        ll1 = compute_nb_log_likelihood(y, mu, theta1, design_matrix, do_cox_reid=False)
        ll2 = compute_nb_log_likelihood(y, mu, theta2, design_matrix, do_cox_reid=False)
        numerical_score = (ll2 - ll1) / (2 * epsilon)
        
        analytical_score = compute_nb_score(y, mu, theta, design_matrix, do_cox_reid=False)
        
        # Should be approximately equal
        relative_error = abs(numerical_score - analytical_score) / (abs(analytical_score) + 1e-10)
        assert relative_error < 0.01, f"Score mismatch: numerical={numerical_score}, analytical={analytical_score}"
    
    def test_nb_functions_edge_cases(self):
        """Test NB functions with edge cases."""
        n_samples = 10
        design_matrix = np.ones((n_samples, 1))
        
        # Case 1: All zero counts
        y_zero = np.zeros(n_samples)
        mu = np.ones(n_samples) * 5
        theta = 1.0
        
        ll = compute_nb_log_likelihood(y_zero, mu, theta, design_matrix, do_cox_reid=False)
        score = compute_nb_score(y_zero, mu, theta, design_matrix, do_cox_reid=False)
        hessian = compute_nb_hessian(y_zero, mu, theta, design_matrix, do_cox_reid=False)
        
        assert np.isfinite(ll)
        assert np.isfinite(score)
        assert np.isfinite(hessian)
        
        # Case 2: Very small theta
        theta_small = 1e-8
        y = np.random.poisson(5, n_samples)  # Should approach Poisson
        
        ll = compute_nb_log_likelihood(y, mu, theta_small, design_matrix, do_cox_reid=False)
        assert np.isfinite(ll)
        
        # Case 3: Very large theta
        theta_large = 1e8
        ll = compute_nb_log_likelihood(y, mu, theta_large, design_matrix, do_cox_reid=False)
        assert np.isfinite(ll)


class TestDispersionEdgeCases:
    """Test edge cases in dispersion estimation."""
    
    def test_dispersion_perfect_fit(self):
        """Test dispersion estimation when model fits perfectly."""
        np.random.seed(42)
        n_samples = 20
        
        # Create design matrix
        design_matrix = np.ones((n_samples, 1))
        beta = np.array([2.0])
        offset_vector = np.zeros(n_samples)
        
        # Generate data with known dispersion
        mu = np.exp(design_matrix @ beta + offset_vector)
        true_dispersion = 0.1
        p = 1 / (1 + mu * true_dispersion)
        n = 1 / true_dispersion
        y = np.random.negative_binomial(n, p)
        
        estimated_dispersion = fit_dispersion(
            beta, design_matrix, y, offset_vector,
            tolerance=1e-6, max_iter=100
        )
        
        # Should be close to true dispersion
        assert 0.05 < estimated_dispersion < 0.5
    
    def test_dispersion_sparse_data(self):
        """Test dispersion estimation with very sparse data."""
        np.random.seed(42)
        n_samples = 50
        
        design_matrix = np.ones((n_samples, 1))
        beta = np.array([-2.0])  # Low mean -> sparse data
        offset_vector = np.zeros(n_samples)
        
        # Generate sparse data (many zeros)
        mu = np.exp(design_matrix @ beta + offset_vector)
        y = np.random.poisson(mu)  # Use Poisson for very sparse
        
        dispersion = fit_dispersion(
            beta, design_matrix, y, offset_vector,
            tolerance=1e-3, max_iter=50
        )
        
        assert dispersion > 0
        assert np.isfinite(dispersion)
    
    def test_dispersion_outliers(self):
        """Test dispersion estimation with outlier observations."""
        np.random.seed(42)
        n_samples = 25
        
        design_matrix = np.ones((n_samples, 1))
        beta = np.array([1.0])
        offset_vector = np.zeros(n_samples)
        
        # Generate normal data with outliers
        y = np.random.negative_binomial(5, 0.3, n_samples)
        y[0] = 1000  # Add extreme outlier
        y[1] = 0     # Add zero outlier
        
        dispersion = fit_dispersion(
            beta, design_matrix, y, offset_vector,
            tolerance=1e-3, max_iter=100
        )
        
        # Should handle outliers without crashing
        assert dispersion > 0
        assert np.isfinite(dispersion)
        # Outliers might increase dispersion estimate
        assert dispersion < 100  # But not too extreme
    
    def test_dispersion_complex_design(self):
        """Test dispersion estimation with complex design matrix."""
        np.random.seed(42)
        n_samples = 30
        
        # Complex design with interactions
        x1 = np.random.binomial(1, 0.5, n_samples)
        x2 = np.random.normal(0, 1, n_samples)
        design_matrix = np.column_stack([
            np.ones(n_samples),  # intercept
            x1,                  # binary
            x2,                  # continuous
            x1 * x2              # interaction
        ])
        
        beta = np.array([1.0, 0.5, -0.3, 0.2])
        offset_vector = np.zeros(n_samples)
        
        # Generate data
        mu = np.exp(design_matrix @ beta + offset_vector)
        y = np.random.negative_binomial(3, 3/(3+mu))
        
        dispersion = fit_dispersion(
            beta, design_matrix, y, offset_vector,
            tolerance=1e-3, max_iter=100, do_cox_reid_adjustment=True
        )
        
        assert dispersion > 0
        assert np.isfinite(dispersion)


class TestDispersionOptimizationRobustness:
    """Test robustness of dispersion optimization."""
    
    def test_multiple_starting_points(self):
        """Test that dispersion estimation is robust to starting points."""
        np.random.seed(42)
        n_samples = 25
        
        design_matrix = np.ones((n_samples, 1))
        beta = np.array([1.5])
        offset_vector = np.zeros(n_samples)
        
        # Generate data with known dispersion
        mu = np.exp(design_matrix @ beta + offset_vector)
        true_dispersion = 0.3
        y = np.random.negative_binomial(1/true_dispersion, 1/(1 + mu * true_dispersion))
        
        # Test different starting points by modifying the initial estimate
        dispersions = []
        for start_multiplier in [0.1, 1.0, 10.0]:
            with patch('devil.overdispersion.estimate_initial_dispersion') as mock_init:
                mock_init.return_value = np.array([true_dispersion * start_multiplier])
                
                dispersion = fit_dispersion(
                    beta, design_matrix, y, offset_vector,
                    tolerance=1e-4, max_iter=100
                )
                dispersions.append(dispersion)
        
        # Results should be similar regardless of starting point
        dispersions = np.array(dispersions)
        assert np.std(dispersions) / np.mean(dispersions) < 0.2  # CV < 20%
    
    def test_optimization_bounds(self):
        """Test that optimization respects reasonable bounds."""
        np.random.seed(42)
        n_samples = 20
        
        design_matrix = np.ones((n_samples, 1))
        beta = np.array([2.0])
        offset_vector = np.zeros(n_samples)
        
        # Generate data
        mu = np.exp(design_matrix @ beta + offset_vector)
        y = np.random.negative_binomial(5, 0.3, n_samples)
        
        dispersion = fit_dispersion(
            beta, design_matrix, y, offset_vector,
            tolerance=1e-3, max_iter=50
        )
        
        # Should be within reasonable bounds
        assert 1e-8 < dispersion < 1e8
        assert np.isfinite(dispersion)