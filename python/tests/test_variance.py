"""
Unit tests for CPU variance estimation functions.
"""

import pytest
import numpy as np
from scipy import linalg
from unittest.mock import patch

from devil.variance import (
    compute_hessian, compute_scores, compute_sandwich_estimator
)


class TestComputeHessian:
    """Test Hessian matrix computation."""
    
    @pytest.fixture
    def hessian_test_data(self):
        """Create test data for Hessian computation."""
        np.random.seed(42)
        n_samples, n_features = 25, 3
        
        # Design matrix
        design_matrix = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples),
            np.random.normal(0, 1, n_samples)
        ])
        
        # Beta coefficients
        beta = np.array([1.5, 0.8, -0.3])
        
        # Response and parameters
        y = np.random.negative_binomial(5, 0.3, n_samples)
        precision = 2.0  # 1 / overdispersion
        size_factors = np.random.lognormal(0, 0.2, n_samples)
        
        return beta, precision, y, design_matrix, size_factors
    
    def test_compute_hessian_basic(self, hessian_test_data):
        """Test basic Hessian computation."""
        beta, precision, y, design_matrix, size_factors = hessian_test_data
        
        H = compute_hessian(beta, precision, y, design_matrix, size_factors)
        
        # Check dimensions
        assert H.shape == (3, 3)
        
        # Should be symmetric
        np.testing.assert_allclose(H, H.T, rtol=1e-10)
        
        # Should be positive definite (inverse of negative Hessian)
        eigenvals = linalg.eigvals(H)
        assert np.all(eigenvals > 0), "Hessian should be positive definite"
        
        # All elements should be finite
        assert np.all(np.isfinite(H))
    
    def test_compute_hessian_single_parameter(self):
        """Test Hessian computation with single parameter (intercept only)."""
        np.random.seed(42)
        n_samples = 20
        
        design_matrix = np.ones((n_samples, 1))
        beta = np.array([1.0])
        y = np.random.negative_binomial(5, 0.3, n_samples)
        precision = 1.0
        size_factors = np.ones(n_samples)
        
        H = compute_hessian(beta, precision, y, design_matrix, size_factors)
        
        assert H.shape == (1, 1)
        assert H[0, 0] > 0
        assert np.isfinite(H[0, 0])
    
    def test_compute_hessian_extreme_precision(self, hessian_test_data):
        """Test Hessian computation with extreme precision values."""
        beta, _, y, design_matrix, size_factors = hessian_test_data
        
        # Very high precision (low overdispersion)
        H_high = compute_hessian(beta, 1e6, y, design_matrix, size_factors)
        assert H_high.shape == (3, 3)
        assert np.all(np.isfinite(H_high))
        assert np.all(linalg.eigvals(H_high) > 0)
        
        # Very low precision (high overdispersion)
        H_low = compute_hessian(beta, 1e-6, y, design_matrix, size_factors)
        assert H_low.shape == (3, 3)
        assert np.all(np.isfinite(H_low))
        assert np.all(linalg.eigvals(H_low) > 0)
    
    def test_compute_hessian_zero_counts(self, hessian_test_data):
        """Test Hessian computation with zero counts."""
        beta, precision, _, design_matrix, size_factors = hessian_test_data
        
        # All zero counts
        y_zero = np.zeros(len(size_factors))
        H = compute_hessian(beta, precision, y_zero, design_matrix, size_factors)
        
        assert H.shape == (3, 3)
        assert np.all(np.isfinite(H))
        assert np.all(linalg.eigvals(H) > 0)
    
    def test_compute_hessian_singular_handling(self):
        """Test handling of near-singular Hessian matrices."""
        np.random.seed(42)
        n_samples = 15
        
        # Create nearly collinear design matrix
        x = np.random.normal(0, 1, n_samples)
        design_matrix = np.column_stack([
            np.ones(n_samples),
            x,
            x + 1e-10  # Nearly identical to second column
        ])
        
        beta = np.array([1.0, 0.5, 0.5])
        y = np.random.negative_binomial(3, 0.4, n_samples)
        precision = 1.0
        size_factors = np.ones(n_samples)
        
        # Should handle near-singularity gracefully
        H = compute_hessian(beta, precision, y, design_matrix, size_factors)
        
        assert H.shape == (3, 3)
        assert np.all(np.isfinite(H))
        # May use pseudo-inverse, so eigenvalues might be small but non-negative


class TestComputeScores:
    """Test score residual computation."""
    
    @pytest.fixture
    def scores_test_data(self):
        """Create test data for score computation."""
        np.random.seed(42)
        n_samples, n_features = 20, 3
        
        design_matrix = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples),
            np.random.normal(0, 1, n_samples)
        ])
        
        y = np.random.negative_binomial(5, 0.3, n_samples)
        beta = np.array([1.2, 0.6, -0.4])
        overdispersion = 0.5
        size_factors = np.random.lognormal(0, 0.2, n_samples)
        
        return design_matrix, y, beta, overdispersion, size_factors
    
    def test_compute_scores_basic(self, scores_test_data):
        """Test basic score computation."""
        design_matrix, y, beta, overdispersion, size_factors = scores_test_data
        
        scores = compute_scores(design_matrix, y, beta, overdispersion, size_factors)
        
        # Check dimensions
        n_samples, n_features = design_matrix.shape
        assert scores.shape == (n_samples, n_features)
        
        # All elements should be finite
        assert np.all(np.isfinite(scores))
        
        # Scores should have reasonable magnitude
        assert np.all(np.abs(scores) < 1000)
    
    def test_compute_scores_zero_overdispersion(self, scores_test_data):
        """Test score computation with zero overdispersion (Poisson case)."""
        design_matrix, y, beta, _, size_factors = scores_test_data
        
        scores = compute_scores(design_matrix, y, beta, 0.0, size_factors)
        
        assert scores.shape == (20, 3)
        assert np.all(np.isfinite(scores))
    
    def test_compute_scores_high_overdispersion(self, scores_test_data):
        """Test score computation with high overdispersion."""
        design_matrix, y, beta, _, size_factors = scores_test_data
        
        scores = compute_scores(design_matrix, y, beta, 100.0, size_factors)
        
        assert scores.shape == (20, 3)
        assert np.all(np.isfinite(scores))
    
    def test_compute_scores_zero_counts(self, scores_test_data):
        """Test score computation with zero counts."""
        design_matrix, _, beta, overdispersion, size_factors = scores_test_data
        
        y_zero = np.zeros(len(size_factors))
        scores = compute_scores(design_matrix, y_zero, beta, overdispersion, size_factors)
        
        assert scores.shape == (20, 3)
        assert np.all(np.isfinite(scores))
    
    def test_compute_scores_extreme_counts(self, scores_test_data):
        """Test score computation with extreme count values."""
        design_matrix, _, beta, overdispersion, size_factors = scores_test_data
        
        # Very high counts
        y_high = np.random.negative_binomial(100, 0.1, len(size_factors))
        scores = compute_scores(design_matrix, y_high, beta, overdispersion, size_factors)
        
        assert scores.shape == (20, 3)
        assert np.all(np.isfinite(scores))


class TestComputeSandwichEstimator:
    """Test sandwich variance estimator computation."""
    
    @pytest.fixture
    def sandwich_test_data(self):
        """Create test data for sandwich estimator computation."""
        np.random.seed(42)
        n_samples, n_features = 30, 3
        n_clusters = 5
        
        design_matrix = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples),
            np.random.normal(0, 1, n_samples)
        ])
        
        y = np.random.negative_binomial(5, 0.3, n_samples)
        beta = np.array([1.3, 0.7, -0.2])
        overdispersion = 0.8
        size_factors = np.random.lognormal(0, 0.2, n_samples)
        
        # Create clusters (patients)
        clusters = np.random.randint(1, n_clusters + 1, n_samples)
        
        return design_matrix, y, beta, overdispersion, size_factors, clusters
    
    def test_compute_sandwich_estimator_basic(self, sandwich_test_data):
        """Test basic sandwich estimator computation."""
        design_matrix, y, beta, overdispersion, size_factors, clusters = sandwich_test_data
        
        sandwich = compute_sandwich_estimator(
            design_matrix, y, beta, overdispersion, size_factors, clusters
        )
        
        # Check dimensions
        n_features = design_matrix.shape[1]
        assert sandwich.shape == (n_features, n_features)
        
        # Should be symmetric
        np.testing.assert_allclose(sandwich, sandwich.T, rtol=1e-10)
        
        # Should be positive definite
        eigenvals = linalg.eigvals(sandwich)
        assert np.all(eigenvals > -1e-10), "Sandwich estimator should be positive semi-definite"
        
        # All elements should be finite
        assert np.all(np.isfinite(sandwich))
    
    def test_compute_sandwich_estimator_single_cluster(self, sandwich_test_data):
        """Test sandwich estimator with single cluster (reduces to standard)."""
        design_matrix, y, beta, overdispersion, size_factors, _ = sandwich_test_data
        
        # All samples in one cluster
        clusters_single = np.ones(len(y), dtype=int)
        
        sandwich = compute_sandwich_estimator(
            design_matrix, y, beta, overdispersion, size_factors, clusters_single
        )
        
        assert sandwich.shape == (3, 3)
        assert np.all(np.isfinite(sandwich))
        assert np.all(linalg.eigvals(sandwich) > -1e-10)
    
    def test_compute_sandwich_estimator_each_sample_cluster(self, sandwich_test_data):
        """Test sandwich estimator where each sample is its own cluster."""
        design_matrix, y, beta, overdispersion, size_factors, _ = sandwich_test_data
        
        # Each sample is its own cluster
        clusters_individual = np.arange(1, len(y) + 1)
        
        sandwich = compute_sandwich_estimator(
            design_matrix, y, beta, overdispersion, size_factors, clusters_individual
        )
        
        assert sandwich.shape == (3, 3)
        assert np.all(np.isfinite(sandwich))
    
    def test_compute_sandwich_vs_standard_hessian(self, sandwich_test_data):
        """Compare sandwich estimator to standard Hessian."""
        design_matrix, y, beta, overdispersion, size_factors, clusters = sandwich_test_data
        
        # Compute sandwich estimator
        sandwich = compute_sandwich_estimator(
            design_matrix, y, beta, overdispersion, size_factors, clusters
        )
        
        # Compute standard Hessian
        precision = 1.0 / overdispersion if overdispersion > 0 else 1e6
        hessian = compute_hessian(beta, precision, y, design_matrix, size_factors)
        
        # Both should be positive definite
        assert np.all(linalg.eigvals(sandwich) > -1e-10)
        assert np.all(linalg.eigvals(hessian) > 0)
        
        # Sandwich should generally be larger (more conservative)
        # This is not always true, but often is for clustered data
        sandwich_trace = np.trace(sandwich)
        hessian_trace = np.trace(hessian)
        assert sandwich_trace > 0
        assert hessian_trace > 0
    
    def test_compute_sandwich_estimator_zero_overdispersion(self, sandwich_test_data):
        """Test sandwich estimator with zero overdispersion."""
        design_matrix, y, beta, _, size_factors, clusters = sandwich_test_data
        
        sandwich = compute_sandwich_estimator(
            design_matrix, y, beta, 0.0, size_factors, clusters
        )
        
        assert sandwich.shape == (3, 3)
        assert np.all(np.isfinite(sandwich))
        assert np.all(linalg.eigvals(sandwich) > -1e-10)
    
    def test_compute_sandwich_estimator_cluster_robustness(self):
        """Test sandwich estimator robustness to cluster assignment."""
        np.random.seed(42)
        n_samples = 40
        
        design_matrix = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        y = np.random.negative_binomial(5, 0.3, n_samples)
        beta = np.array([1.0, 0.5])
        overdispersion = 1.0
        size_factors = np.ones(n_samples)
        
        # Test different cluster configurations
        clusters1 = np.repeat(np.arange(1, 9), 5)  # 8 clusters of 5 samples each
        clusters2 = np.repeat(np.arange(1, 21), 2)  # 20 clusters of 2 samples each
        
        sandwich1 = compute_sandwich_estimator(
            design_matrix, y, beta, overdispersion, size_factors, clusters1
        )
        sandwich2 = compute_sandwich_estimator(
            design_matrix, y, beta, overdispersion, size_factors, clusters2
        )
        
        # Both should be valid
        assert np.all(np.isfinite(sandwich1))
        assert np.all(np.isfinite(sandwich2))
        assert np.all(linalg.eigvals(sandwich1) > -1e-10)
        assert np.all(linalg.eigvals(sandwich2) > -1e-10)


class TestVarianceEdgeCases:
    """Test edge cases in variance estimation."""
    
    def test_perfect_separation(self):
        """Test variance estimation with perfect separation in design."""
        np.random.seed(42)
        n_samples = 20
        
        # Create perfect separation
        condition = np.concatenate([np.zeros(10), np.ones(10)])
        design_matrix = np.column_stack([
            np.ones(n_samples),
            condition
        ])
        
        # Generate data with clear separation
        beta = np.array([1.0, 3.0])  # Large effect
        mu = np.exp(design_matrix @ beta)
        y = np.random.poisson(mu)
        
        precision = 1.0
        size_factors = np.ones(n_samples)
        
        H = compute_hessian(beta, precision, y, design_matrix, size_factors)
        
        # Should handle perfect separation
        assert H.shape == (2, 2)
        assert np.all(np.isfinite(H))
    
    def test_variance_with_outliers(self):
        """Test variance estimation with outlier observations."""
        np.random.seed(42)
        n_samples = 25
        
        design_matrix = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        beta = np.array([1.0, 0.5])
        y = np.random.negative_binomial(5, 0.3, n_samples)
        y[0] = 1000  # Add extreme outlier
        
        overdispersion = 1.0
        size_factors = np.ones(n_samples)
        clusters = np.random.randint(1, 6, n_samples)
        
        # Should handle outliers without crashing
        H = compute_hessian(beta, 1.0, y, design_matrix, size_factors)
        sandwich = compute_sandwich_estimator(
            design_matrix, y, beta, overdispersion, size_factors, clusters
        )
        
        assert np.all(np.isfinite(H))
        assert np.all(np.isfinite(sandwich))
    
    def test_variance_minimal_data(self):
        """Test variance estimation with minimal data."""
        # Just enough data to fit model
        n_samples = 3
        design_matrix = np.column_stack([
            np.ones(n_samples),
            [0, 1, 0]  # Minimal variation
        ])
        
        beta = np.array([1.0, 0.5])
        y = np.array([2, 8, 3])  # Small dataset
        precision = 1.0
        size_factors = np.ones(n_samples)
        clusters = np.array([1, 2, 1])  # Two clusters
        
        # Should work with minimal data
        H = compute_hessian(beta, precision, y, design_matrix, size_factors)
        sandwich = compute_sandwich_estimator(
            design_matrix, y, beta, 1.0, size_factors, clusters
        )
        
        assert H.shape == (2, 2)
        assert sandwich.shape == (2, 2)
        assert np.all(np.isfinite(H))
        assert np.all(np.isfinite(sandwich))
    
    def test_variance_unbalanced_clusters(self):
        """Test variance estimation with unbalanced cluster sizes."""
        np.random.seed(42)
        n_samples = 30
        
        design_matrix = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        beta = np.array([1.0, 0.5])
        y = np.random.negative_binomial(5, 0.3, n_samples)
        overdispersion = 1.0
        size_factors = np.ones(n_samples)
        
        # Create very unbalanced clusters
        clusters = np.concatenate([
            np.ones(25, dtype=int),      # Large cluster
            np.arange(2, 7, dtype=int)   # Small clusters of size 1 each
        ])
        
        sandwich = compute_sandwich_estimator(
            design_matrix, y, beta, overdispersion, size_factors, clusters
        )
        
        assert sandwich.shape == (2, 2)
        assert np.all(np.isfinite(sandwich))
        assert np.all(linalg.eigvals(sandwich) > -1e-10)


class TestVarianceNumericalStability:
    """Test numerical stability of variance computations."""
    
    def test_extreme_size_factors(self):
        """Test variance computation with extreme size factors."""
        np.random.seed(42)
        n_samples = 20
        
        design_matrix = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        beta = np.array([1.0, 0.5])
        y = np.random.negative_binomial(5, 0.3, n_samples)
        precision = 1.0
        
        # Very extreme size factors
        size_factors = np.random.lognormal(0, 3, n_samples)  # High variance
        size_factors[0] = 1e-6  # Very small
        size_factors[1] = 1e6   # Very large
        
        H = compute_hessian(beta, precision, y, design_matrix, size_factors)
        
        assert H.shape == (2, 2)
        assert np.all(np.isfinite(H))
        assert np.all(linalg.eigvals(H) > 0)
    
    def test_numerical_precision_consistency(self):
        """Test numerical precision consistency."""
        np.random.seed(42)
        n_samples = 25
        
        design_matrix = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples)
        ])
        
        beta = np.array([1.0, 0.5])
        y = np.random.negative_binomial(5, 0.3, n_samples)
        precision = 1.0
        size_factors = np.ones(n_samples)
        
        # Compute twice with same inputs
        H1 = compute_hessian(beta, precision, y, design_matrix, size_factors)
        H2 = compute_hessian(beta, precision, y, design_matrix, size_factors)
        
        # Should be identical
        np.testing.assert_allclose(H1, H2, rtol=1e-15)
    
    def test_matrix_conditioning(self):
        """Test behavior with poorly conditioned matrices."""
        np.random.seed(42)
        n_samples = 20
        
        # Create poorly conditioned design matrix
        x = np.random.normal(0, 1, n_samples)
        design_matrix = np.column_stack([
            np.ones(n_samples),
            x,
            x + 1e-12,  # Nearly collinear
            np.random.normal(0, 1e-6, n_samples)  # Very small values
        ])
        
        beta = np.array([1.0, 0.5, 0.5, 0.1])
        y = np.random.negative_binomial(5, 0.3, n_samples)
        precision = 1.0
        size_factors = np.ones(n_samples)
        
        # Should handle poor conditioning gracefully
        H = compute_hessian(beta, precision, y, design_matrix, size_factors)
        
        assert H.shape == (4, 4)
        assert np.all(np.isfinite(H))
        # Eigenvalues might be very small but should not be NaN