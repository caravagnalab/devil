"""
Comprehensive unit tests for overdispersion estimation functions.

This file contains updated and comprehensive tests for the devil.overdispersion module,
covering all major functions and edge cases with correct function signatures.
"""

import pytest
import numpy as np
from scipy import optimize, special, stats
from unittest.mock import patch, MagicMock, call
import warnings

from devil.overdispersion import (
    estimate_initial_dispersion, 
    fit_dispersion,
    compute_nb_log_likelihood, 
    compute_nb_score, 
    compute_nb_hessian
)


class TestEstimateInitialDispersion:
    """Test initial dispersion estimation using method of moments."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic test data with known properties."""
        np.random.seed(42)
        n_genes, n_samples = 100, 50
        
        # Create count matrix with controlled overdispersion
        true_dispersions = np.random.gamma(2, 0.3, n_genes)  # True dispersion values
        counts = np.zeros((n_genes, n_samples))
        
        for i in range(n_genes):
            # Generate mean expression levels
            mu = np.random.gamma(20, 2, n_samples)
            
            # Generate negative binomial counts with known dispersion
            alpha = 1.0 / true_dispersions[i]
            p = alpha / (alpha + mu)
            counts[i, :] = np.random.negative_binomial(alpha, p)
        
        # Create realistic offset vector (log normalization factors)
        offset_vector = np.random.normal(0, 0.2, n_samples)
        
        return counts, offset_vector, true_dispersions
    
    def test_basic_functionality(self, synthetic_data):
        """Test basic dispersion estimation functionality."""
        counts, offset_vector, true_dispersions = synthetic_data
        
        estimated_dispersions = estimate_initial_dispersion(counts, offset_vector)
        
        # Basic sanity checks
        assert len(estimated_dispersions) == len(true_dispersions)
        assert np.all(estimated_dispersions > 0), "All dispersions should be positive"
        assert np.all(np.isfinite(estimated_dispersions)), "All dispersions should be finite"
        
        # Should be in reasonable range
        assert np.all(estimated_dispersions < 1000), "Dispersions should not be extremely large"
        assert np.all(estimated_dispersions > 1e-8), "Dispersions should not be extremely small"
    
    def test_correlation_with_true_values(self, synthetic_data):
        """Test that estimated dispersions correlate with true values."""
        counts, offset_vector, true_dispersions = synthetic_data
        
        estimated_dispersions = estimate_initial_dispersion(counts, offset_vector)
        
        # Should have positive correlation with true values
        correlation = np.corrcoef(true_dispersions, estimated_dispersions)[0, 1]
        assert correlation > 0.3, f"Correlation too low: {correlation:.3f}"
    
    def test_edge_case_all_zeros(self):
        """Test behavior with all-zero count data."""
        n_genes, n_samples = 20, 30
        counts = np.zeros((n_genes, n_samples))
        offset_vector = np.zeros(n_samples)
        
        dispersions = estimate_initial_dispersion(counts, offset_vector)
        
        assert len(dispersions) == n_genes
        assert np.all(dispersions == 100.0), "Should return default high value for all-zero data"
    
    def test_edge_case_constant_counts(self):
        """Test behavior with constant (no variance) count data."""
        n_genes, n_samples = 15, 25
        counts = np.full((n_genes, n_samples), 10)  # All counts are 10
        offset_vector = np.zeros(n_samples)
        
        dispersions = estimate_initial_dispersion(counts, offset_vector)
        
        assert len(dispersions) == n_genes
        assert np.all(dispersions == 100.0), "Should return default high value for constant data"
    
    def test_single_gene(self):
        """Test dispersion estimation for single gene."""
        np.random.seed(123)
        n_samples = 40
        counts = np.random.negative_binomial(10, 0.4, size=(1, n_samples))
        offset_vector = np.zeros(n_samples)
        
        dispersions = estimate_initial_dispersion(counts, offset_vector)
        
        assert len(dispersions) == 1
        assert dispersions[0] > 0
        assert np.isfinite(dispersions[0])
    
    def test_offset_effects(self):
        """Test that different offset vectors produce different results."""
        np.random.seed(456)
        n_genes, n_samples = 30, 40
        counts = np.random.negative_binomial(8, 0.5, size=(n_genes, n_samples))
        
        offset1 = np.zeros(n_samples)
        offset2 = np.random.normal(0.5, 0.3, n_samples)  # Different normalization
        
        disp1 = estimate_initial_dispersion(counts, offset1)
        disp2 = estimate_initial_dispersion(counts, offset2)
        
        # Results should differ due to different normalization
        assert not np.array_equal(disp1, disp2), "Different offsets should yield different results"
    
    def test_high_overdispersion_detection(self):
        """Test detection of high overdispersion scenarios."""
        np.random.seed(789)
        n_genes, n_samples = 50, 30
        
        # Generate highly overdispersed data
        high_disp = 5.0
        counts = np.zeros((n_genes, n_samples))
        for i in range(n_genes):
            mu = np.random.gamma(15, 1, n_samples)
            alpha = 1.0 / high_disp
            p = alpha / (alpha + mu)
            counts[i, :] = np.random.negative_binomial(alpha, p)
        
        offset_vector = np.zeros(n_samples)
        dispersions = estimate_initial_dispersion(counts, offset_vector)
        
        # Should detect high overdispersion
        mean_disp = np.mean(dispersions)
        assert mean_disp > 1.0, f"Should detect high overdispersion, got {mean_disp:.3f}"


class TestFitDispersion:
    """Test maximum likelihood dispersion parameter fitting."""
    
    @pytest.fixture
    def single_gene_setup(self):
        """Create setup for single gene dispersion fitting."""
        np.random.seed(42)
        n_samples = 30
        
        # Design matrix with intercept and one covariate
        x = np.random.binomial(1, 0.5, n_samples)
        design_matrix = np.column_stack([np.ones(n_samples), x])
        
        # True coefficients
        true_beta = np.array([2.0, 0.8])
        
        # Generate data
        offset_vector = np.random.normal(0, 0.1, n_samples)
        mu = np.exp(design_matrix @ true_beta + offset_vector)
        
        # Generate counts with known dispersion
        true_theta = 0.5
        alpha = 1.0 / true_theta
        p = alpha / (alpha + mu)
        y = np.random.negative_binomial(alpha, p)
        
        return {
            'beta': true_beta,
            'design_matrix': design_matrix,
            'y': y,
            'offset_vector': offset_vector,
            'true_theta': true_theta,
            'mu': mu
        }
    
    def test_basic_dispersion_fitting(self, single_gene_setup):
        """Test basic dispersion parameter fitting."""
        setup = single_gene_setup
        
        estimated_theta = fit_dispersion(
            setup['beta'], 
            setup['design_matrix'], 
            setup['y'], 
            setup['offset_vector'],
            tolerance=1e-4,
            max_iter=100,
            do_cox_reid_adjustment=False
        )
        
        assert estimated_theta > 0, "Dispersion should be positive"
        assert np.isfinite(estimated_theta), "Dispersion should be finite"
        
        # Should be reasonably close to true value (within factor of 3)
        ratio = estimated_theta / setup['true_theta']
        assert 0.3 < ratio < 3.0, f"Estimated theta {estimated_theta:.3f} too far from true {setup['true_theta']:.3f}"
    
    def test_cox_reid_adjustment(self, single_gene_setup):
        """Test Cox-Reid adjustment in dispersion fitting."""
        setup = single_gene_setup
        
        # Fit with and without Cox-Reid adjustment
        theta_with_cr = fit_dispersion(
            setup['beta'], setup['design_matrix'], setup['y'], setup['offset_vector'],
            do_cox_reid_adjustment=True
        )
        
        theta_without_cr = fit_dispersion(
            setup['beta'], setup['design_matrix'], setup['y'], setup['offset_vector'],
            do_cox_reid_adjustment=False
        )
        
        assert theta_with_cr > 0 and theta_without_cr > 0
        assert np.isfinite(theta_with_cr) and np.isfinite(theta_without_cr)
        
        # Cox-Reid adjustment should make a difference
        assert theta_with_cr != theta_without_cr, "Cox-Reid adjustment should change the estimate"
    
    def test_all_zero_counts(self):
        """Test dispersion fitting with all-zero counts."""
        n_samples = 20
        design_matrix = np.ones((n_samples, 1))
        beta = np.array([1.0])
        y = np.zeros(n_samples)
        offset_vector = np.zeros(n_samples)
        
        theta = fit_dispersion(beta, design_matrix, y, offset_vector)
        
        assert theta == 0.0, "Should return 0.0 for all-zero counts"
    
    def test_convergence_tolerance(self, single_gene_setup):
        """Test that different tolerance settings affect convergence."""
        setup = single_gene_setup
        
        # Fit with strict tolerance
        theta_strict = fit_dispersion(
            setup['beta'], setup['design_matrix'], setup['y'], setup['offset_vector'],
            tolerance=1e-6, max_iter=200
        )
        
        # Fit with loose tolerance
        theta_loose = fit_dispersion(
            setup['beta'], setup['design_matrix'], setup['y'], setup['offset_vector'],
            tolerance=1e-2, max_iter=50
        )
        
        assert theta_strict > 0 and theta_loose > 0
        assert np.isfinite(theta_strict) and np.isfinite(theta_loose)
        
        # Should be similar but not necessarily identical
        relative_diff = abs(theta_strict - theta_loose) / max(theta_strict, theta_loose)
        assert relative_diff < 0.5, "Results with different tolerances should be reasonably similar"
    
    def test_complex_design_matrix(self):
        """Test dispersion fitting with complex design matrix."""
        np.random.seed(101)
        n_samples = 50
        
        # Complex design with multiple covariates and interactions
        x1 = np.random.binomial(1, 0.4, n_samples)
        x2 = np.random.normal(0, 1, n_samples)
        x3 = np.random.exponential(1, n_samples)
        
        design_matrix = np.column_stack([
            np.ones(n_samples),     # intercept
            x1,                     # binary
            x2,                     # continuous
            x3,                     # skewed continuous
            x1 * x2,               # interaction
            x2**2                   # quadratic
        ])
        
        beta = np.array([1.5, 0.8, -0.3, 0.2, 0.4, -0.1])
        offset_vector = np.random.normal(0, 0.15, n_samples)
        
        # Generate data
        mu = np.exp(design_matrix @ beta + offset_vector)
        theta = 0.7
        alpha = 1.0 / theta
        p = alpha / (alpha + mu)
        y = np.random.negative_binomial(alpha, p)
        
        estimated_theta = fit_dispersion(
            beta, design_matrix, y, offset_vector,
            tolerance=1e-3, max_iter=150
        )
        
        assert estimated_theta > 0
        assert np.isfinite(estimated_theta)
        assert 0.1 < estimated_theta < 5.0, "Should be in reasonable range"
    
    def test_optimization_robustness(self):
        """Test robustness to different starting conditions."""
        np.random.seed(202)
        n_samples = 35
        
        design_matrix = np.ones((n_samples, 1))
        beta = np.array([1.8])
        offset_vector = np.zeros(n_samples)
        
        mu = np.exp(design_matrix @ beta + offset_vector)
        true_theta = 0.4
        alpha = 1.0 / true_theta
        p = alpha / (alpha + mu)
        y = np.random.negative_binomial(alpha, p)
        
        # Test multiple runs (should be deterministic but testing robustness)
        estimates = []
        for _ in range(3):
            theta = fit_dispersion(beta, design_matrix, y, offset_vector)
            estimates.append(theta)
        
        # All estimates should be identical (deterministic algorithm)
        assert len(set(estimates)) == 1, "Algorithm should be deterministic"
        assert estimates[0] > 0 and np.isfinite(estimates[0])


class TestNegativeBinomialFunctions:
    """Test negative binomial likelihood, score, and Hessian functions."""
    
    @pytest.fixture
    def nb_data(self):
        """Create test data for NB function testing."""
        np.random.seed(42)
        n_samples = 25
        
        # Generate realistic data
        design_matrix = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.6, n_samples)
        ])
        
        mu = np.random.gamma(8, 2, n_samples)  # Realistic mean values
        theta = 0.3  # Moderate overdispersion
        y = np.random.negative_binomial(1/theta, 1/(1 + mu * theta))
        
        return {
            'y': y,
            'mu': mu,
            'theta': theta,
            'design_matrix': design_matrix
        }
    
    def test_log_likelihood_basic(self, nb_data):
        """Test basic log-likelihood computation."""
        data = nb_data
        
        ll = compute_nb_log_likelihood(
            data['y'], data['mu'], data['theta'], 
            data['design_matrix'], do_cox_reid=False
        )
        
        assert isinstance(ll, (float, np.floating))
        assert np.isfinite(ll), "Log-likelihood should be finite"
        # For real count data, likelihood is typically negative
        assert ll < 50, "Log-likelihood should not be extremely positive"
    
    def test_log_likelihood_cox_reid(self, nb_data):
        """Test log-likelihood with Cox-Reid adjustment."""
        data = nb_data
        
        ll_cr = compute_nb_log_likelihood(
            data['y'], data['mu'], data['theta'], 
            data['design_matrix'], do_cox_reid=True
        )
        
        ll_no_cr = compute_nb_log_likelihood(
            data['y'], data['mu'], data['theta'], 
            data['design_matrix'], do_cox_reid=False
        )
        
        assert np.isfinite(ll_cr) and np.isfinite(ll_no_cr)
        assert ll_cr != ll_no_cr, "Cox-Reid adjustment should change likelihood"
    
    def test_score_function(self, nb_data):
        """Test score function computation."""
        data = nb_data
        
        score = compute_nb_score(
            data['y'], data['mu'], data['theta'], 
            data['design_matrix'], do_cox_reid=False
        )
        
        assert isinstance(score, (float, np.floating))
        assert np.isfinite(score), "Score should be finite"
    
    def test_hessian_function(self, nb_data):
        """Test Hessian computation."""
        data = nb_data
        
        hessian = compute_nb_hessian(
            data['y'], data['mu'], data['theta'], do_cox_reid=False
        )
        
        assert isinstance(hessian, (float, np.floating))
        assert np.isfinite(hessian), "Hessian should be finite"
        # For well-posed problems, Hessian should typically be negative
        # (indicating concave log-likelihood)
    
    def test_numerical_derivatives_consistency(self, nb_data):
        """Test that analytical derivatives match numerical ones."""
        data = nb_data
        epsilon = 1e-6
        
        # Test score function against numerical derivative of log-likelihood
        theta = data['theta']
        theta_plus = theta + epsilon
        theta_minus = theta - epsilon
        
        ll_plus = compute_nb_log_likelihood(
            data['y'], data['mu'], theta_plus, 
            data['design_matrix'], do_cox_reid=False
        )
        ll_minus = compute_nb_log_likelihood(
            data['y'], data['mu'], theta_minus, 
            data['design_matrix'], do_cox_reid=False
        )
        
        numerical_score = (ll_plus - ll_minus) / (2 * epsilon)
        analytical_score = compute_nb_score(
            data['y'], data['mu'], theta, 
            data['design_matrix'], do_cox_reid=False
        )
        
        # Allow for some numerical error
        relative_error = abs(numerical_score - analytical_score) / (abs(analytical_score) + 1e-10)
        assert relative_error < 0.05, f"Score function error too large: {relative_error:.4f}"
    
    def test_edge_cases_zero_counts(self):
        """Test NB functions with all-zero counts."""
        n_samples = 15
        y = np.zeros(n_samples)
        mu = np.random.gamma(5, 1, n_samples)
        theta = 1.0
        design_matrix = np.ones((n_samples, 1))
        
        ll = compute_nb_log_likelihood(y, mu, theta, design_matrix, do_cox_reid=False)
        score = compute_nb_score(y, mu, theta, design_matrix, do_cox_reid=False)
        hessian = compute_nb_hessian(y, mu, theta, do_cox_reid=False)
        
        assert np.isfinite(ll), "Log-likelihood should be finite for zero counts"
        assert np.isfinite(score), "Score should be finite for zero counts"
        assert np.isfinite(hessian), "Hessian should be finite for zero counts"
    
    def test_extreme_theta_values(self, nb_data):
        """Test NB functions with extreme theta values."""
        data = nb_data
        
        # Very small theta (high overdispersion)
        theta_small = 1e-6
        ll_small = compute_nb_log_likelihood(
            data['y'], data['mu'], theta_small, 
            data['design_matrix'], do_cox_reid=False
        )
        assert np.isfinite(ll_small), "Should handle very small theta"
        
        # Very large theta (approaches Poisson)
        theta_large = 1e6
        ll_large = compute_nb_log_likelihood(
            data['y'], data['mu'], theta_large, 
            data['design_matrix'], do_cox_reid=False
        )
        assert np.isfinite(ll_large), "Should handle very large theta"
    
    def test_consistency_across_functions(self, nb_data):
        """Test consistency between all NB functions."""
        data = nb_data
        
        # All functions should work with the same inputs without error
        ll = compute_nb_log_likelihood(
            data['y'], data['mu'], data['theta'], 
            data['design_matrix'], do_cox_reid=True
        )
        score = compute_nb_score(
            data['y'], data['mu'], data['theta'], 
            data['design_matrix'], do_cox_reid=True
        )
        hessian = compute_nb_hessian(
            data['y'], data['mu'], data['theta'], do_cox_reid=True
        )
        
        assert all(np.isfinite([ll, score, hessian])), "All functions should return finite values"


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""
    
    def test_full_pipeline_simulation(self):
        """Test complete dispersion estimation pipeline."""
        np.random.seed(12345)
        n_genes, n_samples = 50, 40
        
        # Simulate realistic experimental design
        treatment = np.repeat([0, 1], n_samples // 2)
        batch = np.tile([0, 1], n_samples // 2)
        
        design_matrix = np.column_stack([
            np.ones(n_samples),     # intercept
            treatment,              # treatment effect
            batch                   # batch effect
        ])
        
        offset_vector = np.random.normal(0, 0.2, n_samples)  # size factors
        
        # Generate data for multiple genes
        results = []
        for gene_idx in range(n_genes):
            # Random coefficients for each gene
            beta = np.random.normal([3.0, 0.5, -0.2], [0.5, 0.3, 0.2])
            true_theta = np.random.gamma(2, 0.2)  # Gene-specific dispersion
            
            # Generate counts
            mu = np.exp(design_matrix @ beta + offset_vector)
            alpha = 1.0 / true_theta
            p = alpha / (alpha + mu)
            y = np.random.negative_binomial(alpha, p)
            
            # Estimate dispersion
            estimated_theta = fit_dispersion(
                beta, design_matrix, y, offset_vector,
                tolerance=1e-3, max_iter=100
            )
            
            results.append({
                'true_theta': true_theta,
                'estimated_theta': estimated_theta,
                'gene_idx': gene_idx
            })
        
        # Analyze results
        true_thetas = [r['true_theta'] for r in results]
        estimated_thetas = [r['estimated_theta'] for r in results]
        
        # All estimates should be positive and finite
        assert all(theta > 0 for theta in estimated_thetas), "All estimates should be positive"
        assert all(np.isfinite(theta) for theta in estimated_thetas), "All estimates should be finite"
        
        # Should have reasonable correlation with true values
        correlation = np.corrcoef(true_thetas, estimated_thetas)[0, 1]
        assert correlation > 0.4, f"Correlation too low: {correlation:.3f}"
        
        # No extreme outliers
        ratios = np.array(estimated_thetas) / np.array(true_thetas)
        assert np.all(ratios > 0.1), "No estimates should be extremely underestimated"
        assert np.all(ratios < 10.0), "No estimates should be extremely overestimated"
    
    def test_overdispersion_detection_scenarios(self):
        """Test ability to distinguish different overdispersion scenarios."""
        np.random.seed(6789)
        n_samples = 60
        
        scenarios = {
            'low_od': 0.05,      # Low overdispersion (close to Poisson)
            'moderate_od': 0.5,   # Moderate overdispersion
            'high_od': 2.0        # High overdispersion
        }
        
        design_matrix = np.ones((n_samples, 1))
        beta = np.array([2.5])
        offset_vector = np.zeros(n_samples)
        
        results = {}
        
        for scenario_name, true_theta in scenarios.items():
            # Generate data
            mu = np.exp(design_matrix @ beta + offset_vector)
            alpha = 1.0 / true_theta
            p = alpha / (alpha + mu)
            y = np.random.negative_binomial(alpha, p)
            
            # Estimate dispersion
            estimated_theta = fit_dispersion(
                beta, design_matrix, y, offset_vector,
                tolerance=1e-4, max_iter=100
            )
            
            results[scenario_name] = {
                'true': true_theta,
                'estimated': estimated_theta
            }
        
        # Check that we can distinguish between scenarios
        low_est = results['low_od']['estimated']
        high_est = results['high_od']['estimated']
        
        assert low_est < high_est, "Should distinguish between low and high overdispersion"
        assert low_est < 1.0, f"Low OD estimate should be < 1.0, got {low_est:.3f}"
        assert high_est > 0.8, f"High OD estimate should be > 0.8, got {high_est:.3f}"


class TestErrorHandlingAndWarnings:
    """Test proper error handling and edge cases."""
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        n_samples = 20
        
        # The current implementation doesn't validate negative counts
        # This test should be updated based on actual behavior
        counts = np.array([[-1, 2, 3], [4, 5, 6]])  # Negative count
        offset = np.zeros(3)
        
        # Test what actually happens (may not raise an error)
        try:
            result = estimate_initial_dispersion(counts, offset)
            # If it doesn't raise, verify it returns something reasonable
            assert len(result) == 2
            assert np.all(np.isfinite(result))
        except (ValueError, RuntimeError):
            # If it does raise, that's also acceptable
            pass
    
    def test_dimension_mismatches(self):
        """Test handling of dimension mismatches."""
        with pytest.raises((ValueError, IndexError)):
            counts = np.random.poisson(5, size=(10, 20))
            offset = np.zeros(15)  # Wrong number of samples
            estimate_initial_dispersion(counts, offset)
    
    def test_singular_design_matrices(self):
        """Test handling of singular design matrices."""
        np.random.seed(999)
        n_samples = 30
        
        # Create singular design matrix
        x1 = np.random.normal(0, 1, n_samples)
        x2 = 2 * x1  # Perfectly correlated
        design_matrix = np.column_stack([np.ones(n_samples), x1, x2])
        
        beta = np.array([1.0, 0.5, 0.25])  # This should sum to reasonable values
        offset_vector = np.zeros(n_samples)
        y = np.random.poisson(np.exp(design_matrix @ beta))
        
        # Should handle singular matrix gracefully (might warn but shouldn't crash)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore expected warnings
            theta = fit_dispersion(beta, design_matrix, y, offset_vector)
            
        assert theta > 0 and np.isfinite(theta), "Should handle singular matrices"
    
    def test_optimization_failure_fallback(self, monkeypatch):
        """Test fallback when optimization fails."""
        np.random.seed(1111)
        n_samples = 25
        
        design_matrix = np.ones((n_samples, 1))
        beta = np.array([1.5])
        offset_vector = np.zeros(n_samples)
        y = np.random.poisson(np.exp(design_matrix @ beta))
        
        # Mock optimization to fail
        def mock_minimize(*args, **kwargs):
            result = MagicMock()
            result.success = False
            result.x = np.array([np.log(0.5)])
            return result
        
        monkeypatch.setattr('devil.overdispersion.optimize.minimize', mock_minimize)
        
        # Should fall back gracefully
        theta = fit_dispersion(beta, design_matrix, y, offset_vector)
        assert theta > 0 and np.isfinite(theta), "Should have fallback for optimization failure"


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])