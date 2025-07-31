"""
Unit tests for CPU size factor calculation functions.
"""

import pytest
import numpy as np
from unittest.mock import patch

from devil.size_factors import (
    calculate_size_factors, compute_offset_vector
)


class TestCalculateSizeFactors:
    """Test size factor calculation."""
    
    @pytest.fixture
    def test_count_matrix(self):
        """Create test count matrix with known properties."""
        np.random.seed(42)
        n_genes, n_samples = 100, 20
        
        # Create count matrix with different sequencing depths
        base_counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        
        # Simulate different sequencing depths by scaling columns
        depth_factors = np.random.lognormal(0, 0.5, n_samples)
        scaled_counts = base_counts * depth_factors[np.newaxis, :]
        
        return scaled_counts.astype(int), depth_factors
    
    def test_calculate_size_factors_basic(self, test_count_matrix):
        """Test basic size factor calculation."""
        counts, true_depths = test_count_matrix
        
        size_factors = calculate_size_factors(counts, verbose=False)
        
        # Check basic properties
        assert len(size_factors) == 20  # Number of samples
        assert np.all(size_factors > 0)  # Should be positive
        assert np.all(np.isfinite(size_factors))
        
        # Geometric mean should be 1
        geom_mean = np.exp(np.mean(np.log(size_factors)))
        np.testing.assert_allclose(geom_mean, 1.0, rtol=1e-10)
        
        # Should be correlated with true depth factors
        correlation = np.corrcoef(size_factors, true_depths)[0, 1]
        assert correlation > 0.5, f"Size factors should correlate with true depths: {correlation}"
    
    def test_calculate_size_factors_median_ratio_method(self):
        """Test size factors using median ratio method (default)."""
        np.random.seed(42)
        n_genes, n_samples = 50, 15
        
        # Create data with known scaling
        base_expression = np.random.gamma(2, 1, n_genes)
        true_size_factors = np.array([0.5, 0.8, 1.0, 1.2, 1.5] * 3)  # Known factors
        
        counts = np.outer(base_expression, true_size_factors)
        counts = np.random.poisson(counts)  # Add Poisson noise
        
        calculated_sf = calculate_size_factors(counts, verbose=False)
        
        # Should be correlated with true size factors
        correlation = np.corrcoef(calculated_sf, true_size_factors)[0, 1]
        assert correlation > 0.8, f"Poor correlation with true size factors: {correlation}"
    
    def test_calculate_size_factors_total_count_method(self):
        """Test size factors using total count method."""
        np.random.seed(42)
        n_genes, n_samples = 30, 12
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        
        # Test total count method (if implemented as option)
        # Note: Current implementation uses median ratio method
        size_factors = calculate_size_factors(counts, verbose=False)
        
        # Should still be valid size factors
        assert len(size_factors) == n_samples
        assert np.all(size_factors > 0)
        assert np.all(np.isfinite(size_factors))
        
        # Geometric mean should be 1
        geom_mean = np.exp(np.mean(np.log(size_factors)))
        np.testing.assert_allclose(geom_mean, 1.0, rtol=1e-10)
    
    def test_calculate_size_factors_single_sample(self):
        """Test size factor calculation with single sample."""
        np.random.seed(42)
        n_genes = 100
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, 1))
        
        size_factors = calculate_size_factors(counts, verbose=False)
        
        assert len(size_factors) == 1
        assert size_factors[0] == 1.0  # Should be 1 for single sample
    
    def test_calculate_size_factors_identical_samples(self):
        """Test size factor calculation with identical samples."""
        np.random.seed(42)
        n_genes, n_samples = 50, 10
        
        # All samples identical
        base_counts = np.random.negative_binomial(5, 0.3, n_genes)
        counts = np.tile(base_counts[:, np.newaxis], (1, n_samples))
        
        size_factors = calculate_size_factors(counts, verbose=False)
        
        # All size factors should be 1
        np.testing.assert_allclose(size_factors, 1.0, rtol=1e-10)
    
    def test_calculate_size_factors_zero_genes(self):
        """Test behavior with genes that have all zeros."""
        np.random.seed(42)
        n_genes, n_samples = 20, 8
        
        counts = np.random.negative_binomial(3, 0.4, size=(n_genes, n_samples))
        
        # Make some genes all zeros
        counts[0, :] = 0  # First gene all zeros
        counts[5, :] = 0  # Another gene all zeros
        
        size_factors = calculate_size_factors(counts, verbose=False)
        
        # Should still work (zero genes are ignored in median ratio calculation)
        assert len(size_factors) == n_samples
        assert np.all(size_factors > 0)
        assert np.all(np.isfinite(size_factors))
    
    def test_calculate_size_factors_all_zero_sample(self):
        """Test error handling with all-zero samples."""
        np.random.seed(42)
        n_genes, n_samples = 20, 8
        
        counts = np.random.negative_binomial(3, 0.4, size=(n_genes, n_samples))
        
        # Make one sample all zeros
        counts[:, 0] = 0
        
        # Should raise error for all-zero sample
        with pytest.raises(ValueError, match="all zeros"):
            calculate_size_factors(counts, verbose=False)
    
    def test_calculate_size_factors_single_gene_error(self):
        """Test error handling with single gene."""
        counts = np.random.negative_binomial(5, 0.3, size=(1, 10))
        
        # Should return all ones for single gene (can't compute ratios)
        size_factors = calculate_size_factors(counts, verbose=False)
        np.testing.assert_allclose(size_factors, 1.0, rtol=1e-10)
    
    def test_calculate_size_factors_verbose_output(self, test_count_matrix):
        """Test verbose output."""
        counts, _ = test_count_matrix
        
        # Test that verbose mode doesn't crash
        size_factors = calculate_size_factors(counts, verbose=True)
        
        assert len(size_factors) == 20
        assert np.all(size_factors > 0)
    
    def test_calculate_size_factors_extreme_counts(self):
        """Test size factor calculation with extreme count values."""
        np.random.seed(42)
        n_genes, n_samples = 30, 10
        
        counts = np.random.negative_binomial(3, 0.4, size=(n_genes, n_samples))
        
        # Add some extreme values
        counts[0, 0] = 10000  # Very high count
        counts[1, 1] = 0      # Zero count
        
        size_factors = calculate_size_factors(counts, verbose=False)
        
        # Should handle extremes gracefully
        assert len(size_factors) == n_samples
        assert np.all(size_factors > 0)
        assert np.all(np.isfinite(size_factors))
        assert np.all(size_factors < 1000)  # Shouldn't be too extreme


class TestComputeOffsetVector:
    """Test offset vector computation."""
    
    def test_compute_offset_vector_basic(self):
        """Test basic offset vector computation."""
        base_offset = 1e-6
        n_samples = 15
        size_factors = np.random.lognormal(0, 0.3, n_samples)
        
        offset_vector = compute_offset_vector(base_offset, n_samples, size_factors)
        
        # Check dimensions
        assert len(offset_vector) == n_samples
        
        # Should be base_offset + log(size_factors)
        expected = base_offset + np.log(size_factors)
        np.testing.assert_allclose(offset_vector, expected, rtol=1e-12)
        
        # All values should be finite
        assert np.all(np.isfinite(offset_vector))
    
    def test_compute_offset_vector_no_size_factors(self):
        """Test offset vector computation without size factors."""
        base_offset = 1e-5
        n_samples = 20
        
        offset_vector = compute_offset_vector(base_offset, n_samples, size_factors=None)
        
        # Should be all base_offset
        expected = np.full(n_samples, base_offset)
        np.testing.assert_allclose(offset_vector, expected)
    
    def test_compute_offset_vector_zero_offset(self):
        """Test offset vector computation with zero base offset."""
        base_offset = 0.0
        n_samples = 10
        size_factors = np.random.lognormal(0, 0.2, n_samples)
        
        offset_vector = compute_offset_vector(base_offset, n_samples, size_factors)
        
        # Should be just log(size_factors)
        expected = np.log(size_factors)
        np.testing.assert_allclose(offset_vector, expected, rtol=1e-12)
    
    def test_compute_offset_vector_extreme_size_factors(self):
        """Test offset vector with extreme size factors."""
        base_offset = 1e-6
        n_samples = 12
        
        size_factors = np.array([1e-8, 1e-4, 0.1, 1.0, 10.0, 1e4, 1e8] + [1.0] * 5)
        
        offset_vector = compute_offset_vector(base_offset, n_samples, size_factors)
        
        # Should handle extreme values
        assert len(offset_vector) == n_samples
        assert np.all(np.isfinite(offset_vector))
        
        # Check a few values manually
        np.testing.assert_allclose(
            offset_vector[3], base_offset + np.log(1.0), rtol=1e-12
        )
    
    def test_compute_offset_vector_single_sample(self):
        """Test offset vector computation with single sample."""
        base_offset = 1e-6
        n_samples = 1
        size_factors = np.array([2.5])
        
        offset_vector = compute_offset_vector(base_offset, n_samples, size_factors)
        
        assert len(offset_vector) == 1
        expected = base_offset + np.log(2.5)
        np.testing.assert_allclose(offset_vector, expected, rtol=1e-12)


class TestSizeFactorEdgeCases:
    """Test edge cases in size factor calculation."""
    
    def test_size_factors_sparse_data(self):
        """Test size factor calculation with very sparse data."""
        np.random.seed(42)
        n_genes, n_samples = 200, 25
        
        # Create very sparse data (many zeros)
        counts = np.random.negative_binomial(1, 0.9, size=(n_genes, n_samples))
        
        size_factors = calculate_size_factors(counts, verbose=False)
        
        assert len(size_factors) == n_samples
        assert np.all(size_factors > 0)
        assert np.all(np.isfinite(size_factors))
        
        # Geometric mean should still be 1
        geom_mean = np.exp(np.mean(np.log(size_factors)))
        np.testing.assert_allclose(geom_mean, 1.0, rtol=1e-10)
    
    def test_size_factors_high_variation(self):
        """Test size factor calculation with high between-sample variation."""
        np.random.seed(42)
        n_genes, n_samples = 80, 20
        
        # Create data with very different sequencing depths
        depths = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0] + [1.0] * 13)
        
        base_counts = np.random.negative_binomial(10, 0.5, n_genes)
        counts = np.outer(base_counts, depths)
        counts = np.random.poisson(counts)
        
        size_factors = calculate_size_factors(counts, verbose=False)
        
        # Should be correlated with true depths
        correlation = np.corrcoef(size_factors, depths)[0, 1]
        assert correlation > 0.8, f"Poor correlation with depths: {correlation}"
        
        # Should still have geometric mean of 1
        geom_mean = np.exp(np.mean(np.log(size_factors)))
        np.testing.assert_allclose(geom_mean, 1.0, rtol=1e-8)
    
    def test_size_factors_few_expressed_genes(self):
        """Test size factor calculation with few expressed genes."""
        np.random.seed(42)
        n_genes, n_samples = 10, 15  # Very few genes
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        
        size_factors = calculate_size_factors(counts, verbose=False)
        
        # Should still work with few genes
        assert len(size_factors) == n_samples
        assert np.all(size_factors > 0)
        assert np.all(np.isfinite(size_factors))
    
    def test_size_factors_constant_expression(self):
        """Test size factor behavior with constant expression levels."""
        n_genes, n_samples = 50, 12
        
        # All genes have same expression in all samples
        counts = np.full((n_genes, n_samples), 10)
        
        size_factors = calculate_size_factors(counts, verbose=False)
        
        # All size factors should be 1
        np.testing.assert_allclose(size_factors, 1.0, rtol=1e-10)
    
    def test_size_factors_outlier_genes(self):
        """Test size factor calculation with outlier genes."""
        np.random.seed(42)
        n_genes, n_samples = 100, 15
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        
        # Add outlier genes with very different expression patterns
        counts[0, :] = 1000  # Very highly expressed in all samples
        counts[1, 0] = 1000  # Very highly expressed in one sample only
        
        size_factors = calculate_size_factors(counts, verbose=False)
        
        # Should be robust to outliers
        assert len(size_factors) == n_samples
        assert np.all(size_factors > 0)
        assert np.all(np.isfinite(size_factors))
        
        # Size factors shouldn't be too extreme due to outliers
        assert np.all(size_factors < 50)  # Reasonable upper bound
        assert np.all(size_factors > 0.02)  # Reasonable lower bound
    
    def test_size_factors_bimodal_expression(self):
        """Test size factor calculation with bimodal expression patterns."""
        np.random.seed(42)
        n_genes, n_samples = 60, 18
        
        # Create bimodal expression: half samples have high expression
        # for half the genes, and vice versa
        counts = np.zeros((n_genes, n_samples))
        
        # First half of genes: high in first half of samples
        counts[:30, :9] = np.random.negative_binomial(20, 0.3, size=(30, 9))
        counts[:30, 9:] = np.random.negative_binomial(2, 0.7, size=(30, 9))
        
        # Second half of genes: high in second half of samples
        counts[30:, :9] = np.random.negative_binomial(2, 0.7, size=(30, 9))
        counts[30:, 9:] = np.random.negative_binomial(20, 0.3, size=(30, 9))
        
        size_factors = calculate_size_factors(counts, verbose=False)
        
        # Should handle bimodal patterns
        assert len(size_factors) == n_samples
        assert np.all(size_factors > 0)
        assert np.all(np.isfinite(size_factors))


class TestSizeFactorNumericalStability:
    """Test numerical stability of size factor calculations."""
    
    def test_size_factors_numerical_precision(self):
        """Test numerical precision in size factor calculations."""
        np.random.seed(42)
        n_genes, n_samples = 50, 15
        
        counts = np.random.negative_binomial(5, 0.3, size=(n_genes, n_samples))
        
        # Calculate twice
        sf1 = calculate_size_factors(counts, verbose=False)
        sf2 = calculate_size_factors(counts, verbose=False)
        
        # Should be identical
        np.testing.assert_allclose(sf1, sf2, rtol=1e-15)
    
    def test_size_factors_large_counts(self):
        """Test size factor calculation with large count values."""
        np.random.seed(42)
        n_genes, n_samples = 40, 12
        
        # Generate large counts
        counts = np.random.negative_binomial(100, 0.1, size=(n_genes, n_samples))
        
        size_factors = calculate_size_factors(counts, verbose=False)
        
        assert len(size_factors) == n_samples
        assert np.all(size_factors > 0)
        assert np.all(np.isfinite(size_factors))
        
        # Geometric mean should be 1
        geom_mean = np.exp(np.mean(np.log(size_factors)))
        np.testing.assert_allclose(geom_mean, 1.0, rtol=1e-10)
    
    def test_size_factors_small_counts(self):
        """Test size factor calculation with very small counts."""
        np.random.seed(42)
        n_genes, n_samples = 30, 10
        
        # Generate small counts (mostly 0, 1, 2)
        counts = np.random.negative_binomial(1, 0.5, size=(n_genes, n_samples))
        
        size_factors = calculate_size_factors(counts, verbose=False)
        
        assert len(size_factors) == n_samples
        assert np.all(size_factors > 0)
        assert np.all(np.isfinite(size_factors))
    
    def test_offset_vector_numerical_stability(self):
        """Test numerical stability of offset vector computation."""
        base_offset = 1e-10  # Very small offset
        n_samples = 20
        size_factors = np.random.lognormal(0, 2, n_samples)  # Wide range
        
        offset_vector = compute_offset_vector(base_offset, n_samples, size_factors)
        
        assert len(offset_vector) == n_samples
        assert np.all(np.isfinite(offset_vector))
        
        # Should be mathematically correct
        expected = base_offset + np.log(size_factors)
        np.testing.assert_allclose(offset_vector, expected, rtol=1e-12)
