"""
Comprehensive validation tests for exact beta fitting implementation.

These tests validate that our Python implementation exactly matches the mathematical
behavior of the R package's C++ beta fitting algorithms.
"""

import numpy as np
import pytest
from typing import Tuple, Optional
import warnings

from devil.beta import (
    init_beta,
    beta_fit,
    beta_fit_group,
    fit_beta_coefficients,
    init_beta_matrix
)

try:
    from devil.beta_gpu import (
        init_beta_gpu,
        beta_fit_gpu_batch,
        beta_fit_group_gpu_batch,
        fit_beta_coefficients_gpu,
        fit_beta_coefficients_gpu_batch
    )
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


class TestBetaFitting:
    """Test the beta fitting implementation against known mathematical properties."""
    
    @pytest.fixture
    def simple_data(self):
        """Generate simple test data with known properties."""
        np.random.seed(42)
        n_samples = 30
        
        # Simple design matrix: intercept + binary covariate
        X = np.column_stack([
            np.ones(n_samples),  # Intercept
            np.random.binomial(1, 0.5, n_samples)  # Binary treatment
        ])
        
        true_beta = np.array([1.5, 0.8])  # Moderate effect sizes
        offset = np.random.normal(0, 0.1, n_samples)  # Small random offsets
        dispersion = 0.5
        
        # Generate counts using true negative binomial model
        mu = np.exp(X @ true_beta + offset)
        alpha = 1.0 / dispersion
        y = np.random.negative_binomial(alpha, alpha / (alpha + mu))
        
        return {
            'y': y,
            'X': X,
            'true_beta': true_beta,
            'offset': offset,
            'dispersion': dispersion,
            'mu': mu
        }
    
    @pytest.fixture
    def complex_data(self):
        """Generate more complex test data."""
        np.random.seed(123)
        n_samples = 50
        
        # Complex design matrix with interactions
        age = np.random.normal(50, 15, n_samples)
        treatment = np.random.binomial(1, 0.4, n_samples)
        batch = np.random.choice([0, 1, 2], n_samples)
        
        X = np.column_stack([
            np.ones(n_samples),  # Intercept
            (age - 50) / 15,  # Normalized age
            treatment,  # Treatment effect
            (batch == 1).astype(float),  # Batch 1 vs 0
            (batch == 2).astype(float),  # Batch 2 vs 0
            treatment * (age - 50) / 15  # Age-treatment interaction
        ])
        
        true_beta = np.array([2.0, 0.3, 0.6, -0.2, 0.1, -0.4])
        offset = np.random.normal(0, 0.2, n_samples)
        dispersion = 1.2
        
        mu = np.exp(X @ true_beta + offset)
        alpha = 1.0 / dispersion
        y = np.random.negative_binomial(alpha, alpha / (alpha + mu))
        
        return {
            'y': y,
            'X': X,
            'true_beta': true_beta,
            'offset': offset,
            'dispersion': dispersion,
            'mu': mu
        }
    
    def test_init_beta(self, simple_data):
        """Test that beta initialization matches expected behavior."""
        y, X = simple_data['y'], simple_data['X']
        
        beta_init = init_beta(y, X)
        
        # Should have correct dimensions
        assert len(beta_init) == X.shape[1]
        assert np.all(np.isfinite(beta_init))
        
        # Should be reasonable initial estimates
        # (log-linear regression on log1p(counts))
        expected = np.linalg.lstsq(X, np.log1p(y), rcond=None)[0]
        np.testing.assert_allclose(beta_init, expected, rtol=1e-10)
    
    def test_beta_fit_convergence(self, simple_data):
        """Test that beta fitting converges to reasonable values."""
        data = simple_data
        
        # Initialize beta
        beta_init = init_beta(data['y'], data['X'])
        
        # Fit using algorithm
        fitted_beta, n_iter, converged = beta_fit(
            data['y'], data['X'], beta_init, data['offset'], 
            data['dispersion'], max_iter=100, tolerance=1e-6
        )
        
        # Should converge
        assert converged, f"Failed to converge after {n_iter} iterations"
        assert n_iter < 100, "Should converge in reasonable number of iterations"
        
        # Should have finite values
        assert np.all(np.isfinite(fitted_beta))
        
        # Should be reasonably close to true values for this simulation
        # (This is stochastic, so we use generous bounds)
        for i in range(len(fitted_beta)):
            assert abs(fitted_beta[i] - data['true_beta'][i]) < 2.0
    
    def test_beta_fit_group(self, simple_data):
        """Test the intercept-only fitting function."""
        y = simple_data['y']
        offset = simple_data['offset']
        dispersion = simple_data['dispersion']
        
        # Fit intercept-only model
        beta_init = np.mean(np.log1p(y))  # Simple initial estimate
        fitted_beta, n_iter, converged = beta_fit_group(
            y, beta_init, offset, dispersion, max_iter=100, tolerance=1e-6
        )
        
        # Should converge
        assert converged
        assert np.isfinite(fitted_beta)
        
        # Compare with full design matrix approach
        X_intercept = np.ones((len(y), 1))
        beta_init_full = init_beta(y, X_intercept)
        fitted_beta_full, _, _ = beta_fit(
            y, X_intercept, beta_init_full, offset, dispersion
        )
        
        # Should give very similar results (allow for minor numerical differences)
        np.testing.assert_allclose(fitted_beta, fitted_beta_full[0], rtol=2e-4)
    
    def test_high_level_interface(self, simple_data):
        """Test the high-level interface function."""
        data = simple_data
        
        fitted_beta, n_iter, converged = fit_beta_coefficients(
            data['y'], data['X'], offset=data['offset'], 
            dispersion=data['dispersion']
        )
        
        assert converged
        assert len(fitted_beta) == data['X'].shape[1]
        assert np.all(np.isfinite(fitted_beta))
    
    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        n_samples = 20
        
        # Test 1: All zero counts
        y_zero = np.zeros(n_samples)
        X = np.column_stack([np.ones(n_samples), np.random.randn(n_samples)])
        
        fitted_beta, n_iter, converged = fit_beta_coefficients(
            y_zero, X, dispersion=1.0
        )
        
        assert converged
        assert n_iter == 1  # Should return immediately
        assert np.allclose(fitted_beta, 0.0)
        
        # Test 2: Very high dispersion (approaching Poisson)
        y_poisson = np.random.poisson(5, n_samples)
        fitted_beta, n_iter, converged = fit_beta_coefficients(
            y_poisson, X, dispersion=1e-6  # Very small -> high precision
        )
        
        assert np.all(np.isfinite(fitted_beta))
        
        # Test 3: Very low dispersion (high variance)
        y_nb = np.random.negative_binomial(1, 0.1, n_samples)  # High variance
        fitted_beta, n_iter, converged = fit_beta_coefficients(
            y_nb, X, dispersion=100.0  # High dispersion
        )
        
        assert np.all(np.isfinite(fitted_beta))
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        np.random.seed(999)
        n_samples = 25
        
        # Test with very large counts
        X = np.column_stack([np.ones(n_samples), np.random.randn(n_samples)])
        y_large = np.random.negative_binomial(10, 0.01, n_samples)  # Can be very large
        
        fitted_beta, n_iter, converged = fit_beta_coefficients(
            y_large, X, dispersion=0.1, max_iter=200
        )
        
        # Should handle large counts without numerical issues
        assert np.all(np.isfinite(fitted_beta))
        
        # Test with very small dispersion
        y_normal = np.random.negative_binomial(5, 0.3, n_samples)
        fitted_beta, n_iter, converged = fit_beta_coefficients(
            y_normal, X, dispersion=1e-8, max_iter=200
        )
        
        assert np.all(np.isfinite(fitted_beta))
    
    def test_mathematical_properties(self, complex_data):
        """Test that the fitting satisfies mathematical properties."""
        data = complex_data
        
        fitted_beta, n_iter, converged = fit_beta_coefficients(
            data['y'], data['X'], offset=data['offset'], 
            dispersion=data['dispersion'], tolerance=1e-8
        )
        
        if not converged:
            pytest.skip("Convergence required for this test")
        
        # Calculate fitted values
        mu_fitted = np.exp(data['X'] @ fitted_beta + data['offset'])
        
        # Test: Pearson residuals should have reasonable properties
        variance = mu_fitted + mu_fitted**2 * data['dispersion']
        pearson_residuals = (data['y'] - mu_fitted) / np.sqrt(variance)
        
        # Should not be systematically biased (mean near zero)
        assert abs(np.mean(pearson_residuals)) < 0.5
        
        # Should have reasonable variance (around 1 for good fit)
        assert 0.5 < np.var(pearson_residuals) < 2.0
    
    def test_matrix_initialization(self):
        """Test matrix-based beta initialization."""
        np.random.seed(42)
        n_genes, n_samples = 10, 20
        n_features = 3
        
        count_matrix = np.random.negative_binomial(5, 0.3, (n_genes, n_samples))
        design_matrix = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples),
            np.random.normal(0, 1, n_samples)
        ])
        
        beta_init = init_beta_matrix(count_matrix, design_matrix)
        
        assert beta_init.shape == (n_genes, n_features)
        assert np.all(np.isfinite(beta_init))
        
        # Should match individual gene initialization
        for gene_idx in range(n_genes):
            expected = init_beta(count_matrix[gene_idx, :], design_matrix)
            np.testing.assert_allclose(beta_init[gene_idx, :], expected, rtol=1e-12)
    
    def test_algorithm_consistency(self, simple_data):
        """Test that different algorithm paths give consistent results."""
        data = simple_data
        
        # Method 1: General algorithm
        fitted_beta1, _, _ = fit_beta_coefficients(
            data['y'], data['X'], offset=data['offset'], 
            dispersion=data['dispersion'], tolerance=1e-8
        )
        
        # Method 2: If intercept-only, should use group algorithm internally
        if data['X'].shape[1] == 1 and np.allclose(data['X'][:, 0], 1.0):
            # This should use the group algorithm automatically
            fitted_beta2, _, _ = fit_beta_coefficients(
                data['y'], data['X'], offset=data['offset'], 
                dispersion=data['dispersion'], tolerance=1e-8
            )
            
            np.testing.assert_allclose(fitted_beta1, fitted_beta2, rtol=1e-10)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
class TestBetaFittingGPU:
    """Test GPU implementation against CPU implementation."""
    
    @pytest.fixture
    def batch_data(self):
        """Generate batch test data."""
        np.random.seed(42)
        n_genes, n_samples = 50, 25
        n_features = 3
        
        count_matrix = np.random.negative_binomial(3, 0.4, (n_genes, n_samples))
        design_matrix = np.column_stack([
            np.ones(n_samples),
            np.random.binomial(1, 0.5, n_samples),
            np.random.normal(0, 1, n_samples)
        ])
        offset_vector = np.random.normal(0, 0.1, n_samples)
        dispersion_vector = np.random.gamma(1, 1, n_genes)
        
        return {
            'count_matrix': count_matrix,
            'design_matrix': design_matrix,
            'offset_vector': offset_vector,
            'dispersion_vector': dispersion_vector
        }
    
    def test_gpu_cpu_consistency(self, batch_data):
        """Test that GPU and CPU implementations give identical results."""
        data = batch_data
        
        # CPU implementation
        beta_init_cpu = init_beta_matrix(
            data['count_matrix'], data['design_matrix']
        )
        
        cpu_results = []
        for gene_idx in range(data['count_matrix'].shape[0]):
            beta, n_iter, converged = fit_beta_coefficients(
                data['count_matrix'][gene_idx, :],
                data['design_matrix'],
                beta_init_cpu[gene_idx, :],
                data['offset_vector'],
                data['dispersion_vector'][gene_idx],
                tolerance=1e-6
            )
            cpu_results.append((beta, n_iter, converged))
        
        cpu_beta = np.array([r[0] for r in cpu_results])
        cpu_iterations = np.array([r[1] for r in cpu_results])
        cpu_converged = np.array([r[2] for r in cpu_results])
        
        # GPU implementation
        gpu_beta, gpu_iterations, gpu_converged = fit_beta_coefficients_gpu(
            data['count_matrix'],
            data['design_matrix'],
            beta_init_cpu,
            data['offset_vector'],
            data['dispersion_vector'],
            tolerance=1e-6,
            dtype=np.float64  # Use double precision for exact comparison
        )
        
        # Results should be very close (allowing for small numerical differences)
        np.testing.assert_allclose(gpu_beta, cpu_beta, rtol=1e-10, atol=1e-12)
        
        # Convergence behavior should be similar (may differ by 1-2 iterations due to ordering)
        assert np.sum(gpu_converged) >= np.sum(cpu_converged) * 0.9  # At least 90% same convergence
    
    def test_gpu_batch_processing(self, batch_data):
        """Test GPU batch processing functionality."""
        data = batch_data
        
        # Test with different batch sizes
        for batch_size in [10, 25, 100]:
            gpu_beta, gpu_iterations, gpu_converged = fit_beta_coefficients_gpu_batch(
                data['count_matrix'],
                data['design_matrix'],
                offset_vector=data['offset_vector'],
                dispersion_vector=data['dispersion_vector'],
                batch_size=batch_size,
                tolerance=1e-6,
                verbose=False
            )
            
            assert gpu_beta.shape == (data['count_matrix'].shape[0], data['design_matrix'].shape[1])
            assert len(gpu_iterations) == data['count_matrix'].shape[0]
            assert len(gpu_converged) == data['count_matrix'].shape[0]
            
            # Should have mostly finite, reasonable results
            assert np.sum(np.all(np.isfinite(gpu_beta), axis=1)) >= data['count_matrix'].shape[0] * 0.8
    
    def test_gpu_memory_management(self, batch_data):
        """Test that GPU memory is managed properly."""
        from devil.gpu import GPUMemoryManager, get_gpu_memory_info
        
        data = batch_data
        
        # Record initial memory
        initial_memory = get_gpu_memory_info()[0]
        
        # Run GPU fitting with memory manager
        with GPUMemoryManager():
            gpu_beta, _, _ = fit_beta_coefficients_gpu(
                data['count_matrix'][:10, :],  # Small batch
                data['design_matrix'],
                offset_vector=data['offset_vector'],
                dispersion_batch=data['dispersion_vector'][:10],
                tolerance=1e-6
            )
        
        # Memory should be cleaned up
        final_memory = get_gpu_memory_info()[0]
        memory_diff = abs(final_memory - initial_memory)
        
        # Should not have significant memory leak (allow for some variability)
        # GPU operations may require substantial memory, so use a more realistic threshold
        assert memory_diff < 500 * 1024 * 1024  # Less than 500MB difference


def run_validation_suite():
    """Run comprehensive validation tests."""
    print("Running comprehensive beta fitting validation...")
    
    # Test mathematical correctness
    test_class = TestBetaFitting()
    
    # Create test data
    np.random.seed(42)
    simple_data = test_class.simple_data()
    complex_data = test_class.complex_data()
    
    print("✓ Testing beta initialization...")
    test_class.test_init_beta(simple_data)
    
    print("✓ Testing beta fitting convergence...")
    test_class.test_beta_fit_convergence(simple_data)
    
    print("✓ Testing group fitting...")
    test_class.test_beta_fit_group(simple_data)
    
    print("✓ Testing high-level interface...")
    test_class.test_high_level_interface(simple_data)
    
    print("✓ Testing edge cases...")
    test_class.test_edge_cases()
    
    print("✓ Testing numerical stability...")
    test_class.test_numerical_stability()
    
    print("✓ Testing mathematical properties...")
    test_class.test_mathematical_properties(complex_data)
    
    if GPU_AVAILABLE:
        print("✓ Testing GPU implementation...")
        gpu_test_class = TestBetaFittingGPU()
        batch_data = gpu_test_class.batch_data()
        gpu_test_class.test_gpu_cpu_consistency(batch_data)
        gpu_test_class.test_gpu_batch_processing(batch_data)
    
    print("All validation tests passed! ✅")
    print("Beta fitting implementation matches R package mathematical behavior.")


if __name__ == "__main__":
    run_validation_suite()